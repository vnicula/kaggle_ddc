from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import tensorflow as tf
import time

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.xception import Xception

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Bidirectional, BatchNormalization, Concatenate, Conv2D, Dense, Dropout
from tensorflow.keras.layers import Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Input, GRU, LeakyReLU, Masking, MaxPooling2D, multiply, Reshape, TimeDistributed
from tensorflow.keras.utils import multi_gpu_model

from keras_utils import ScaledDotProductAttention, SeqSelfAttention, SeqWeightedAttention

# Needed because keras model.fit shape checks are weak
# https://github.com/tensorflow/tensorflow/issues/24520
# tf.enable_eager_execution()

# TODO there's an input warning that input doesnt come from input layer
# TODO use parallel_interleave

SEQ_LEN = 16


class MesoInception4():
    def __init__(self, learning_rate = 0.001):
        self.model = self.init_model()
        # optimizer = Adam(lr = learning_rate)
        # self.model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])
    
    def InceptionLayer(self, a, b, c, d):
        def func(x):
            x1 = Conv2D(a, (1, 1), padding='same', activation='relu')(x)
            
            x2 = Conv2D(b, (1, 1), padding='same', activation='relu')(x)
            x2 = Conv2D(b, (3, 3), padding='same', activation='relu')(x2)
            
            x3 = Conv2D(c, (1, 1), padding='same', activation='relu')(x)
            x3 = Conv2D(c, (3, 3), dilation_rate = 2, strides = 1, padding='same', activation='relu')(x3)
            
            x4 = Conv2D(d, (1, 1), padding='same', activation='relu')(x)
            x4 = Conv2D(d, (3, 3), dilation_rate = 3, strides = 1, padding='same', activation='relu')(x4)

            y = Concatenate(axis = -1)([x1, x2, x3, x4])
            
            return y
        return func
    
    def init_model(self):
        x = Input(shape = (IMGWIDTH, IMGWIDTH, 3))
        
        x1 = self.InceptionLayer(1, 4, 4, 2)(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
        x2 = self.InceptionLayer(2, 4, 4, 2)(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)        
        
        x3 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
        
        x4 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
        
        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation = 'sigmoid')(y)

        return Model(inputs = x, outputs = y)


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
    
    names = []
    labels = []
    samples = []
    masks = []
    
    with open(file_path, 'rb') as f_p:
        data = pickle.load(f_p)
        selected_keys = [k for k in data.keys() if data[k][0] == 1]
        initial_positives = len(selected_keys)
        if len(selected_keys) > 1:
            random.shuffle(selected_keys)
            selected_keys = selected_keys[:len(selected_keys) // 2]
        print('Loaded {}, dropped {} positives.'.format(file_path, initial_positives-len(selected_keys)))
        selected_set = set(selected_keys)

        for key in data.keys():
            label = data[key][0]
            if label == 1 and key not in selected_set:
                continue
            names.append(key)
            labels.append(label)
            # labels.append(int(data[key][0] == 'FAKE'))
            # sample = data[key][1][0]

            img_size = data[key][1][0].shape[0]
            my_seq_len = len(data[key][1])
            data_seq_len = min(my_seq_len, SEQ_LEN)
            sample = np.zeros((SEQ_LEN, img_size, img_size, 3), dtype=np.float32)
            mask = np.zeros(SEQ_LEN, dtype=np.float32)
            for indx in range(data_seq_len):
                sample[indx] = data[key][1][indx]
                mask[indx] = 1.0
            # sample = preprocess_input(sample)
            # I think this is what preprocess input does with 'tf' mode
            sample /= 127.5
            sample -= 1.

            if my_seq_len < SEQ_LEN:
                sample[my_seq_len:] = np.zeros((SEQ_LEN-my_seq_len, img_size, img_size, 3), dtype=np.float32)
            # print(sample.shape)
            samples.append(sample)
            masks.append(mask)

            if label == 0:
                samples.append(np.fliplr(sample))
                masks.append(mask)
                labels.append(0)

    # NOTE if one sample doesn't have enough frames Keras will error out here with 'assign a sequence'
    npsamples = np.array(samples, dtype=np.float32)
    npmasks = np.array(masks, dtype=np.float32)
    nplabels = np.array(labels, dtype=np.int32)

    print('file {} Shape samples {}, labels {}'.format(file_path, npsamples.shape, nplabels.shape))
    # return tf.data.Dataset.from_tensor_slices((npsamples, npmasks, nplabels))
    return npsamples, npmasks, nplabels

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

def input_dataset(input_dir, is_training):
    print('Using dataset from: ', input_dir)
    dataset = tf.data.Dataset.list_files(input_dir).shuffle(8192)
    
    # dataset = dataset.interleave(
    #     # map_func=lambda file_name: tf.py_func(read_file, [file_name], [tf.float32, tf.float32, tf.int32]),
    #     map_func=lambda file_name: tf.data.Dataset.from_tensor_slices(
    #         tuple(tf.py_func(read_file, [file_name], [tf.float32, tf.float32, tf.int32]))
    #     ),
    #     cycle_length=32, #tf.data.experimental.AUTOTUNE
    #     block_length=64,
    #     num_parallel_calls=16
    #     )      

    # dataset = dataset.map(
    #     lambda file_name: tf.py_func(read_file, [file_name], [tf.float32, tf.float32, tf.int32]),
    #     num_parallel_calls=32
    # ).prefetch(64) #.prefetch(tf.data.experimental.AUTOTUNE)
    # dataset = dataset.flat_map(
    #     lambda *x: tf.data.Dataset.from_tensor_slices(x)
    # )

    dataset = dataset.apply(tf.data.experimental.parallel_interleave(
        lambda file_name: tf.data.Dataset.from_tensor_slices(
            tuple(tf.py_func(read_file, [file_name], [tf.float32, tf.float32, tf.int32]))),
        cycle_length=8,
        block_length=8,
        sloppy=True,
        buffer_output_elements=4,
        )
    )

    def final_map(s, m, l):
        return  {'input_1': tf.reshape(s, [-1, 224, 224, 3]), 'input_2': tf.reshape(m, [-1])}, tf.reshape(l, [-1])
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


def weighted_ce_logits(y_true, y_pred):
    return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, 0.3, name='weighted_ce_logits')

# TODO: use Bidirectional and try narrower MobileNet
# TODO: explore removal of projection at the end of MobileNet to LSTM
# TODO: resolve from logits for all metrics, perhaps do your own weighted BCE
# oh wait its already in TF doc
def create_model(input_shape, weights):

    # with tf.device('/cpu:0'):
    # strategy = tf.distribute.MirroredStrategy()
    # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    # with strategy.scope():

    input_layer = Input(shape=input_shape)
    input_mask = Input(shape=(input_shape[0]))
    # reshape = Reshape([224, 224, 3])(input_layer)

    feature_extractor = MobileNetV2(include_top=False,
        weights=weights,
        # weights='imagenet',
        alpha=0.5,
        input_shape=input_shape[-3:],
        pooling='max'
        # pooling=None
    )

    # feature_extractor = Xception(include_top=False, 
    #     weights='imagenet', 
    #     # input_tensor=None, 
    #     input_shape=None, 
    #     pooling='avg'
    # )

    # for layer in mobilenet.layers:
    #     if (('block_16' not in layer.name) and ('block_15' not in layer.name)
    #         and ('block_14' not in layer.name) and ('block_13' not in layer.name)
    #         and ('block_12' not in layer.name) and ('block_11' not in layer.name)
    #         and ('block_10' not in layer.name) and ('block_9' not in layer.name)
    #     ):
    #         layer.trainable = False
    #     else:
    #         print('Layer {} trainable {}'.format(layer.name, layer.trainable))
    
    # for layer in mobilenet.layers:
    #     if (('block_1_' in layer.name) or ('block_2_' in layer.name)
    #         # and ('block_14' not in layer.name) and ('block_13' not in layer.name)
    #         # and ('block_12' not in layer.name) and ('block_11' not in layer.name)
    #         # and ('block_10' not in layer.name) and ('block_9' not in layer.name)
    #     ):
    #         layer.trainable = False
    #         print('Layer {} trainable {}'.format(layer.name, layer.trainable))

    # features = mobilenet.output
    # # features = Conv2D(512, (1, 1), strides=(1, 1), padding='valid', activation='relu')(features)
    # # features = Conv2D(256, (3, 3), strides=(2, 2), padding='valid', activation='relu')(features)
    # # # features = MaxPooling2D(pool_size=(2, 2))(features)
    # # features = GlobalMaxPooling2D()(features)
    # features = Flatten()(features)
    # features = Dense(128, activation='elu')(features)
    # # features = Dropout(0.25)(features)

    # feature_extractor = Model(inputs=mobilenet.input, outputs=features)
    # # print(feature_extractor.summary())
    
    net = TimeDistributed(feature_extractor)(input_layer)
    # net = multiply([net, input_mask])
    # net = Masking(mask_value = 0.0)(net)
    net = Bidirectional(GRU(256, return_sequences=True))(net, mask=input_mask)
    net = SeqSelfAttention(attention_type='multiplicative', attention_activation='sigmoid')(net, mask=input_mask)
    net = Bidirectional(GRU(256, return_sequences=False))(net, mask=input_mask)
    # net = SeqWeightedAttention()(net, mask=input_mask)
    # net = ScaledDotProductAttention()(net, mask=input_mask)
    # net = Bidirectional(GRU(128, return_sequences=False))(net, mask=input_mask)
    # net = Flatten()(net)
    # net = Dense(256, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(net)
    # net = Dropout(0.25)(net)
    out = Dense(1, activation='sigmoid', 
        bias_initializer=tf.keras.initializers.Constant(np.log([1.5])))(net)
    # out = Dense(1, activation='sigmoid')(net)

    model = Model(inputs=[input_layer, input_mask], outputs=out)
    
    # This saturates GPU:0 for large models
    # parallel_model = multi_gpu_model(model, cpu_merge=False, gpus=4)
    # parallel_model = multi_gpu_model(model, gpus=2)
    # print("Training using multiple GPUs..")

    # def get_lr_metric(optimizer):
    #     def lr(y_true, y_pred):
    #         return optimizer.lr
    #     return lr

    optimizer = tf.keras.optimizers.Adam(lr=0.01)
    # TODO keras needs custom_objects when loading models with custom metrics
    # But this one needs the optimizer.
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
    my_loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1)
    # parallel_model.compile(loss=my_loss, optimizer=optimizer, metrics=METRICS)
    # print(parallel_model.summary())
    model.compile(loss=my_loss, optimizer=optimizer, metrics=METRICS)
    print(model.summary())

    return model


def save_loss(H):
    plt.figure()
    N = len(H.history["loss"])
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("loss.jpg")


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

    train_dataset = input_dataset(args.train_dir, is_training=True)
    eval_dataset = input_dataset(args.eval_dir, is_training=False)
 
    # elem = dataset.make_one_shot_iterator().get_next()
    # with tf.Session() as session:
    #     npelem = session.run(elem)
    #     print(npelem[0], npelem[1])
    #     print(npelem[0].shape)

    in_shape = (SEQ_LEN, 224, 224, 3)

    # in_shape = (224, 224, 3)
    if args.load is not None:
        print('Loading model and weights from: ', args.load)
        custom_objs = {
            'fraction_positives':fraction_positives,
            'SeqWeightedAttention':SeqWeightedAttention,
            # 'SeqSelfAttention':SeqSelfAttention,
            # 'weighted_ce_logits':weighted_ce_logits,
        }
        model = tf.keras.models.load_model(args.load, custom_objects=custom_objs)
    else:
        print('Training model from scratch.')
        model = create_model(in_shape,
            # 'pretrained/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5'
            'pretrained/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224_no_top.h5'
        )

    num_epochs = 100
    validation_steps = 128
    batch_size = 16

    train_dataset = train_dataset.shuffle(buffer_size=512).batch(batch_size).prefetch(2)
    eval_dataset = eval_dataset.take(validation_steps * (batch_size + 1)).batch(batch_size).prefetch(2)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='mobgru_{epoch}.h5',
            save_best_only=True,
            monitor='val_binary_crossentropy',
            verbose=1),
        tf.keras.callbacks.EarlyStopping(
            # Stop training when `val_loss` is no longer improving
            # monitor='val_loss', # watch out for reg losses
            monitor='val_binary_crossentropy',
            # "no longer improving" being defined as "no better than 1e-2 less"
            min_delta=1e-3,
            # "no longer improving" being further defined as "for at least 2 epochs"
            patience=10,
            verbose=1),
        tf.keras.callbacks.CSVLogger('mobgru_log.csv'),
        # tf.keras.callbacks.LearningRateScheduler(step_decay),
        # CosineAnnealingScheduler(T_max=num_epochs, eta_max=0.02, eta_min=1e-5),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_binary_crossentropy', 
            factor=0.9, patience=2, min_lr=1e-5, verbose=1, mode='min')
    ]
    
    class_weight={0: 0.6, 1: 0.4}
    # class_weight=[0.99, 0.01]
    history = model.fit(train_dataset, epochs=num_epochs, class_weight=class_weight, 
        validation_data=eval_dataset, validation_steps=validation_steps, callbacks=callbacks)
    save_loss(history)

    t1 = time.time()
    print("Execution took: {}".format(t1-t0))

