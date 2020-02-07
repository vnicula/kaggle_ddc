from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import cv2
import feature_extractor_models as featx
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import tensorflow as tf
import time
import threading

from scipy.interpolate import griddata

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.xception import Xception

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Bidirectional, BatchNormalization, Concatenate, Conv2D, Dense, Dropout
from tensorflow.keras.layers import Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Input, GRU, LeakyReLU, LSTM, Masking, MaxPooling2D, multiply, Reshape, TimeDistributed
# from tensorflow.keras.utils import multi_gpu_model

from keras_utils import ScaledDotProductAttention, SeqSelfAttention, SeqWeightedAttention, binary_focal_loss, save_loss
# from multi_head import Encoder, CustomSchedule

# Needed because keras model.fit shape checks are weak
# https://github.com/tensorflow/tensorflow/issues/24520
# tf.enable_eager_execution()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# TODO there's an input warning that input doesnt come from input layer
# TODO use parallel_interleave

SEQ_LEN = 30
# FEAT_SHAPE = (300,)
FEAT_SHAPE = (224, 224, 3)
D_MODEL = 784

def save_sample_img(name, label, values):
    IMG_SIZE = values[0].shape[0]    

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 4
    font_scale = 2

    # line_shape = (IMG_SIZE, max_elems*IMG_SIZE, 3)
    tile_shape = (IMG_SIZE, SEQ_LEN*IMG_SIZE, 3)
    tile_img = np.zeros(tile_shape, dtype=np.float32)
    for j in range(len(values)):
        color = (0, 255, 0) if label == 0 else (255, 0, 0)
        cv2.putText(tile_img, name, (10, 50),
                        font_face, font_scale,
                        color, thickness, 2)
        
        tile_img[:, j*IMG_SIZE:(j+1)*IMG_SIZE, :] = values[j]

    plt.imsave(name+'.jpg', tile_img)

    return tile_img


def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof


def get_simple_feature(face):
    t0 = time.time()

    N = 300
    img = np.dot(face[...,:3], [0.2989, 0.5870, 0.1140])    
    # Calculate FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    # Calculate the azimuthally averaged 1D power spectrum
    psd1D = azimuthalAverage(magnitude_spectrum)

    # Interpolation
    points = np.linspace(0,N,num=psd1D.size) 
    xi = np.linspace(0,N,num=N) 
    interpolated = griddata(points,psd1D,xi,method='cubic')
    
    # Normalization
    interpolated /= interpolated[0]
    # print('simple feature shape: {}, took {}'.format(interpolated.shape, time.time()-t0))

    return interpolated             


def get_image_feature(face):
    # I think this is what preprocess input does with 'tf' mode
    return (face.astype(np.float32) / 127.5) - 1.0


class CosineAnnealingScheduler(tf.keras.callbacks.Callback):
    """Cosine annealing scheduler.
    """

    def __init__(self, T_max, eta_max, eta_min=0, verbose=0):
        super(CosineAnnealingScheduler, self).__init__()
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: CosineAnnealingScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)


def rescale_list(input_list, size):
    """Given a list and a size, return a rescaled/samples list. For example,
    if we want a list of size 5 and we have a list of size 25, return a new
    list of size five which is every 5th element of the origina list."""
    assert len(input_list) >= size

    # Get the number to skip between iterations.
    skip = len(input_list) // size

    # Build our new output.
    output = [input_list[i] for i in range(0, len(input_list), skip)]

    # Cut off the last one if needed.
    return output[:size]


# TODO oversample REAL
def read_file(file_path):
    
    t0 = time.time()
    # names = []
    labels = []
    samples = []
    masks = []
    
    with open(file_path.numpy(), 'rb') as f_p:
        data = pickle.load(f_p)
        selected_keys = [k for k in data.keys() if data[k][0] == 1]
        initial_positives = len(selected_keys)
        if len(selected_keys) > 2:
            random.shuffle(selected_keys)
            selected_keys = selected_keys[:int(len(selected_keys) * 0.7)]
        print('Loaded {}, dropped {} positives.'.format(file_path, initial_positives-len(selected_keys)))
        selected_set = set(selected_keys)
        feature_func = get_image_feature
        # feature_func = get_simple_feature

        for key in data.keys():
            label = data[key][0]
            if label == 1 and key not in selected_set:
                continue
            # names.append(key)
            # sample = data[key][1][0]

            # feat_shape = feature_func(data[key][1][0]).shape
            feat_shape = FEAT_SHAPE
            my_seq_len = len(data[key][1])
            data_seq_len = min(my_seq_len, SEQ_LEN)
            sample = np.zeros((SEQ_LEN,) + feat_shape, dtype=np.float32)
            sample_f = np.zeros((SEQ_LEN,) + feat_shape, dtype=np.float32)
            mask = np.zeros(SEQ_LEN, dtype=np.float32)
            for indx in range(data_seq_len):
                sample[indx] = feature_func(data[key][1][indx])
                if label == 0:
                    sample_f[indx] = feature_func(np.fliplr(data[key][1][indx]))
                mask[indx] = 1.0
            
            # sample = preprocess_input(sample)
            # print(file_path, len(samples))
            samples.append(sample)
            masks.append(mask)
            labels.append(label)

            if label == 0:
                samples.append(sample_f)
                masks.append(mask)
                labels.append(0)
                # save_sample_img(key+'_o', 0, sample)
                # save_sample_img(key+'_f', 0, sample_f)
        
        del data
    # NOTE if one sample doesn't have enough frames Keras will error out here with 'assign a sequence'
    npsamples = np.array(samples, dtype=np.float32)
    npmasks = np.array(masks, dtype=np.float32)
    nplabels = np.array(labels, dtype=np.int32)

    print('file {} Shape samples {}, labels {} took {}'.format(file_path, npsamples.shape, nplabels.shape, time.time()-t0))
    # return tf.data.Dataset.from_tensor_slices((npsamples, npmasks, nplabels))
    return npsamples, npmasks, nplabels

"""
npsamples = np.zeros((64,) + (SEQ_LEN,) + FEAT_SHAPE, dtype=np.float32)
npmasks = np.ones((64,) + (SEQ_LEN,), dtype=np.float32)
nplabels = np.ones((64,), dtype = np.int32)
def fake_read_file(file_path):
    t0 = time.time()

    print('Fake read {}, thread {}, took {}'.format(file_path, threading.get_ident(), time.time()-t0))
    return npsamples, npmasks, nplabels
"""

# def input_dataset(input_dir):
#     print('Using dataset from: ', input_dir)
#     dataset = tf.data.Dataset.list_files(input_dir)
#     # f_list = os.listdir(input_dir)
#     # dataset_files = [os.path.join(input_dir, fn) for fn in f_list if fn.endswith('pkl')]
#     # dataset = tf.data.Dataset.from_tensor_slices((dataset_files))
    
#     dataset = dataset.flat_map(
#         lambda file_name: tf.data.Dataset.from_tensor_slices(
#             tuple(tf.py_func(read_file, [file_name], [tf.float32, tf.float32, tf.float32]))
#         )
#     )
#     def final_map(s, m, l):
#         return  {'input_1': tf.reshape(s, [-1, 224, 224, 3]), 'input_2': tf.reshape(m, [-1, 1])}, tf.reshape(l, [-1])
#     dataset = dataset.map(final_map)
#     return dataset

# TODO wip finish the optimal one
# def input_dataset(input_dir, is_training):
#     print('Using dataset from: ', input_dir)

#     dataset = tf.data.Dataset.list_files(input_dir).shuffle(1024)
#     # dataset = tf.data.Dataset.range(1, 2000)
#     def map_function_wrapper(filename):
#         features, masks, labels = tf.py_function(
#            read_file, [filename], (tf.float32, tf.float32, tf.int32))
        
#         return tf.data.Dataset.from_tensor_slices((features, masks, labels))
#         # return tf.data.Dataset.from_tensor_slices((npsamples, npmasks, nplabels))
    
#     dataset = dataset.map(
#         map_function_wrapper,
#         num_parallel_calls=8
#     ).prefetch(4)
#     dataset = dataset.interleave(
#         # lambda *x: tf.data.Dataset.from_tensor_slices(x).map(
#         lambda x: x.map(
#             lambda s, m, l: ({'input_1': tf.reshape(s, (-1,)+FEAT_SHAPE), 'input_2': tf.reshape(m, [-1])}, tf.reshape(l, [-1]))
#         ),
#         cycle_length=16,
#         num_parallel_calls=16
#     )

#     return dataset


def tfrecords_dataset(input_dir, is_training):
    print('Using tfrecords dataset from: ', input_dir)
    
    file_list = tf.data.Dataset.list_files(input_dir).shuffle(512)
    dataset = tf.data.TFRecordDataset(filenames=file_list, buffer_size=None, num_parallel_reads=40)

    feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'name': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'sample': tf.io.FixedLenFeature([30, 224, 224, 3], tf.float32),
        'mask': tf.io.FixedLenFeature([30], tf.float32),
    }

    def _parse_function(example_proto):
        example = tf.io.parse_single_example(example_proto, feature_description)
        sample = (example['sample'] + 1.0) / 2
        return {'input_1': sample, 'input_2': example['mask']}, example['label']

    dataset = dataset.map(map_func=_parse_function, num_parallel_calls=16)

    return dataset


def input_dataset(input_dir, is_training):
    print('Using dataset from: ', input_dir)

    dataset = tf.data.Dataset.list_files(input_dir).shuffle(1024)
    # options = tf.data.Options()
    # options.experimental_deterministic = False
    # dataset = dataset.with_options(options)
    # dataset = dataset.interleave(
    #     # map_func=lambda file_name: tf.py_func(read_file, [file_name], [tf.float32, tf.float32, tf.int32]),
    #     map_func=lambda file_name: tf.data.Dataset.from_tensor_slices(
    #         tuple(tf.py_function(read_file, [file_name], [tf.float32, tf.float32, tf.int32]))
    #     ),
    #     cycle_length=16, #tf.data.experimental.AUTOTUNE
    #     block_length=1,
    #     num_parallel_calls=16
    # )

    # dataset = dataset.map(
    #     lambda file_name: tf.py_func(read_file, [file_name], [tf.float32, tf.float32, tf.int32]),
    #     num_parallel_calls=32
    # ).prefetch(64) #.prefetch(tf.data.experimental.AUTOTUNE)
    # dataset = dataset.flat_map(
    #     lambda *x: tf.data.Dataset.from_tensor_slices(x)
    # )

    dataset = dataset.apply(tf.data.experimental.parallel_interleave(
        lambda file_name: tf.data.Dataset.from_tensor_slices(
            tuple(tf.py_function(read_file, [file_name], [tf.float32, tf.float32, tf.int32]))),
        cycle_length=8,
        block_length=1,
        sloppy=True,
        buffer_output_elements=4,
        )
    )

    def final_map(s, m, l):
        return  {'input_1': tf.reshape(s, (-1,)+FEAT_SHAPE), 'input_2': tf.reshape(m, [-1])}, tf.reshape(l, [-1])
    dataset = dataset.map(final_map, num_parallel_calls=16)

    # def class_func(sample, mask, label):
    #     return label

    # if is_training:
    #     resampler = tf.data.experimental.rejection_resample(class_func, target_dist=[0.3, 0.7])
    #     dataset = dataset.apply(resampler)

    return dataset


def fraction_positives(y_true, y_pred):
    return tf.keras.backend.mean(y_true)


# class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
#     """
#     Args:
#       pos_weight: Scalar to affect the positive labels of the loss function.
#       weight: Scalar to affect the entirety of the loss function.
#       from_logits: Whether to compute loss form logits or the probability.
#       reduction: Type of tf.keras.losses.Reduction to apply to loss.
#       name: Name of the loss function.
#     """
#     def __init__(self, pos_weight, weight, from_logits=False,
#                  reduction=tf.keras.losses.Reduction.AUTO,
#                  name='weighted_binary_crossentropy'):
#         super(WeightedBinaryCrossEntropy, self).__init__(reduction=reduction,
#                                                          name=name)
#         self.pos_weight = pos_weight
#         self.weight = weight
#         self.from_logits = from_logits

#     def call(self, y_true, y_pred):
#         if not self.from_logits:
#             # Manually calculate the weighted cross entropy.
#             # Formula is qz * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
#             # where z are labels, x is logits, and q is the weight.
#             # Since the values passed are from sigmoid (assuming in this case)
#             # sigmoid(x) will be replaced by y_pred

#             # qz * -log(sigmoid(x)) 1e-6 is added as an epsilon to stop passing a zero into the log
#             x_1 = y_true * self.pos_weight * -tf.math.log(y_pred + 1e-6)

#             # (1 - z) * -log(1 - sigmoid(x)). Epsilon is added to prevent passing a zero into the log
#             x_2 = (1 - y_true) * -tf.math.log(1 - y_pred + 1e-6)

#             return tf.add(x_1, x_2) * self.weight 

#         # Use built in function
#         return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, self.pos_weight) * self.weight


def compile_model(model):

    optimizer = tf.keras.optimizers.Adam(lr=0.02) #(lr=0.025)
    # learning_rate=CustomSchedule(D_MODEL)
    # optimizer = tf.keras.optimizers.Adam(
    #     learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # TODO keras needs custom_objects when loading models with custom metrics
    # But this one needs the optimizer.
    # def get_lr_metric(optimizer):
    #     def lr(y_true, y_pred):
    #         return optimizer.lr
    #     return lr
    # lr_metric = get_lr_metric(optimizer)

    thresh = 0.5
    METRICS = [
        tf.keras.metrics.TruePositives(name='tp', thresholds=thresh),
        tf.keras.metrics.FalsePositives(name='fp', thresholds=thresh),
        tf.keras.metrics.TrueNegatives(name='tn', thresholds=thresh),
        tf.keras.metrics.FalseNegatives(name='fn', thresholds=thresh), 
        tf.keras.metrics.BinaryAccuracy(name='acc', threshold=thresh),
        tf.keras.metrics.Precision(name='precision', thresholds=thresh),
        tf.keras.metrics.Recall(name='recall', thresholds=thresh),
        tf.keras.metrics.AUC(name='auc'),
        # tf.keras.metrics.BinaryCrossentropy(from_logits=True),
        tf.keras.metrics.BinaryCrossentropy(),
        fraction_positives,
        # lr_metric,
    ]
    # my_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)
    # my_loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.025)
    my_loss = binary_focal_loss(alpha=0.7)
    # parallel_model.compile(loss=my_loss, optimizer=optimizer, metrics=METRICS)
    # print(parallel_model.summary())
    model.compile(loss=my_loss, optimizer=optimizer, metrics=METRICS)

    return model


def weighted_ce_logits(y_true, y_pred):
    return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, 0.3, name='weighted_ce_logits')

# TODO: use Bidirectional and try narrower MobileNet
# TODO: explore removal of projection at the end of MobileNet to LSTM
# TODO: resolve from logits for all metrics, perhaps do your own weighted BCE
# oh wait its already in TF doc
def create_model(input_shape):

    input_layer = Input(shape=input_shape)
    input_mask = Input(shape=(input_shape[0]))
    # reshape = Reshape([224, 224, 3])(input_layer)

    classifier = featx.MesoInception4()
    # print(classifier.model.summary())
    # classifier.model.load_weights('pretrained/Meso/raw/all/weights.h5')
    classifier.model.load_weights('best_meso_featx_weights.h5')

    for i, layer in enumerate(classifier.model.layers):
        print(i, layer.name, layer.trainable)
        if layer.name == 'max_pooling2d_3':
            output = layer.output
            print('output set to {} max_pooling2d_3'.format())
    feature_extractor = Model(inputs=classifier.model.input, outputs=output)

    # 'pretrained/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5'
    # 'pretrained/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224_no_top.h5',

    # feature_extractor = MobileNetV2(include_top=False,
    #     weights='pretrained/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224_no_top.h5',
    #     # weights='imagenet',
    #     alpha=0.5,
    #     input_shape=input_shape[-3:],
    #     pooling='avg'
    #     # pooling=None
    # )

    # feature_extractor = Xception(include_top=False, 
    #     weights='imagenet', 
    #     # input_tensor=None, 
    #     input_shape=None, 
    #     pooling='avg'
    # )

    for layer in feature_extractor.layers:
        layer.trainable = False
    print(feature_extractor.summary())

    # for layer in feature_extractor.layers:
    #     if (('block_16' not in layer.name) # and ('block_15' not in layer.name)
    #         # and ('block_14' not in layer.name) and ('block_13' not in layer.name)
    #         # and ('block_12' not in layer.name) and ('block_11' not in layer.name)
    #         # and ('block_10' not in layer.name) and ('block_9' not in layer.name)
    #     ):
    #         layer.trainable = False
    #     else:
    #         print('Layer {} trainable {}'.format(layer.name, layer.trainable))
    
    # for layer in feature_extractor.layers:
    #     if (('block_1_' in layer.name) or ('block_2_' in layer.name) 
    #         or ('block_3_' in layer.name) or ('block_4_' in layer.name)
    #         or ('block_5_' in layer.name) or ('block_6_' in layer.name)
    #         or ('block_7_' in layer.name) or ('block_8_' in layer.name)
    #     ):
    #         layer.trainable = False
    #         print('Layer {} trainable {}'.format(layer.name, layer.trainable))

    # features = mobilenet.output
    # features = Conv2D(512, (1, 1), strides=(1, 1), padding='valid', activation='relu')(features)
    # features = Conv2D(256, (3, 3), strides=(2, 2), padding='valid', activation='relu')(features)
    # # # features = MaxPooling2D(pool_size=(2, 2))(features)
    # features = GlobalMaxPooling2D()(features)
    # features = Flatten()(features)
    # features = Dense(128, activation='elu')(features)
    # features = Dropout(0.25)(features)

    # feature_extractor = Model(inputs=mobilenet.input, outputs=features)
    # # print(feature_extractor.summary())
    
    # net = input_layer
    net = TimeDistributed(feature_extractor)(input_layer)
    net = TimeDistributed(Flatten())(net)
    # net = TimeDistributed(Conv2D(256, (1, 1), strides=(1, 1), padding='valid', activation='relu'))(net)
    # net = TimeDistributed(Conv2D(D_MODEL, (3, 3), strides=(2, 2), padding='valid', activation='relu'))(net)
    # net = TimeDistributed(BatchNormalization())(net)
    # net = TimeDistributed(GlobalMaxPooling2D())(net)
    # net = Encoder(num_layers=1, d_model=D_MODEL, num_heads=2, dff=256,
    #     maximum_position_encoding=1000)(net, mask=input_mask)
    # net = multiply([net, input_mask])
    # net = Masking(mask_value = 0.0)(net)
    net = Bidirectional(GRU(256, return_sequences=True))(net, mask=input_mask)
    # net = SeqSelfAttention(attention_type='additive', attention_activation='sigmoid')(net, mask=input_mask)
    # net = Bidirectional(GRU(256, return_sequences=True))(net, mask=input_mask)
    # net = ScaledDotProductAttention()(net, mask=input_mask)

    net = SeqWeightedAttention()(net, mask=input_mask)
    # net = Bidirectional(GRU(128, return_sequences=False))(net, mask=input_mask)
    
    # net = Dense(256, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(net)
    # net = Dropout(0.25)(net)
    out = Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.002),
        bias_initializer=tf.keras.initializers.Constant(np.log([1.5])))(net)
    # out = Dense(1, activation='sigmoid')(net)

    model = Model(inputs=[input_layer, input_mask], outputs=out)

    # This saturates GPU:0 for large models
    # parallel_model = multi_gpu_model(model, cpu_merge=False, gpus=4)
    # parallel_model = multi_gpu_model(model, gpus=2)
    # print("Training using multiple GPUs..")
    
    print(model.summary())
    return model


def step_decay(epoch):
    initial_lrate = 0.045
    drop = 0.9
    epochs_drop = 2.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    print('learning rate ', lrate)
    return lrate


if __name__ == '__main__':

    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--eval_dir', type=str)
    parser.add_argument('--load', type=str, default=None)
    args = parser.parse_args()

    # train_dataset = input_dataset(args.train_dir, is_training=True)
    # eval_dataset = input_dataset(args.eval_dir, is_training=False)
    train_dataset = tfrecords_dataset(args.train_dir, is_training=True)
    eval_dataset = tfrecords_dataset(args.eval_dir, is_training=False)
 
    # elem = dataset.make_one_shot_iterator().get_next()
    # with tf.Session() as session:
    #     npelem = session.run(elem)
    #     print(npelem[0], npelem[1])
    #     print(npelem[0].shape)

    in_shape = (SEQ_LEN,) + FEAT_SHAPE

    custom_objs = {
        'fraction_positives':fraction_positives,
        'SeqWeightedAttention':SeqWeightedAttention,
        # 'SeqSelfAttention':SeqSelfAttention,
        # 'weighted_ce_logits':weighted_ce_logits,
    }

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model = create_model(in_shape)
        if args.load is not None:
            print('Loading model and weights from: ', args.load)
            # model = tf.keras.models.load_model(args.load, custom_objects=custom_objs)
            model.load_weights(args.load)
        else:
            print('Training model from scratch.')
        compile_model(model)

    num_epochs = 1000
    # validation_steps = 32
    batch_size = 64

    # Cached for small datasets
    # train_dataset = train_dataset.shuffle(buffer_size=256).cache().batch(batch_size).prefetch(2)
    # eval_dataset = eval_dataset.take(validation_steps * (batch_size + 1)).cache().batch(batch_size).prefetch(1)
    train_dataset = train_dataset.batch(batch_size).prefetch(4)
    eval_dataset = eval_dataset.batch(batch_size).prefetch(4)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='fattw_{epoch}.h5',
            save_best_only=True,
            monitor='val_binary_crossentropy',
            # save_format='tf',
            save_weights_only=True,
            verbose=1),
        tf.keras.callbacks.EarlyStopping(
            # Stop training when `val_loss` is no longer improving
            # monitor='val_loss', # watch out for reg losses
            monitor='val_binary_crossentropy',
            min_delta=1e-4,
            patience=50,
            verbose=1),
        tf.keras.callbacks.CSVLogger('training_log.csv'),
        # tf.keras.callbacks.LearningRateScheduler(step_decay),
        # CosineAnnealingScheduler(T_max=num_epochs, eta_max=0.02, eta_min=1e-5),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_binary_crossentropy', 
            factor=0.95, patience=1, min_lr=5e-6, verbose=1, mode='min')
    ]
    
    # class_weight={0: 0.65, 1: 0.35}
    # class_weight=[0.99, 0.01]
    history = model.fit(train_dataset, epochs=num_epochs, #class_weight=class_weight, 
        validation_data=eval_dataset, #validation_steps=validation_steps, 
        callbacks=callbacks)
    
    model.save('final_model.h5')
    # new_model = tf.keras.models.load_model('my_model')
    # new_model.summary()

    save_loss(history, 'final_model')
    t1 = time.time()

    print("Execution took: {}".format(t1-t0))

