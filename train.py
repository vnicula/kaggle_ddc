from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import tensorflow as tf
import time

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Input, GRU, Masking, MaxPooling2D, multiply, Reshape, TimeDistributed
from tensorflow.keras.utils import multi_gpu_model

# Needed because keras model.fit shape checks are weak
# https://github.com/tensorflow/tensorflow/issues/24520
# tf.enable_eager_execution()

# TODO there's an input warning that input doesnt come from input layer
# TODO use parallel_interleave

SEQ_LEN = 30

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
        print('Loaded ', file_path)
        for key in data.keys():
            names.append(key)
            labels.append(data[key][0])
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

            if data[key][0] == 0:
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

def input_dataset(input_dir):
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

    dataset = dataset.apply(tf.data.experimental.parallel_interleave(
        lambda file_name: tf.data.Dataset.from_tensor_slices(
            tuple(tf.py_func(read_file, [file_name], [tf.float32, tf.float32, tf.int32]))),
        cycle_length=8,
        block_length=8,
        sloppy=True,
        buffer_output_elements=8,
        )
    )

    # dataset = dataset.map(
    #     lambda file_name: tf.py_func(read_file, [file_name], [tf.float32, tf.float32, tf.int32]),
    #     num_parallel_calls=32
    # ).prefetch(64) #.prefetch(tf.data.experimental.AUTOTUNE)
    # dataset = dataset.flat_map(
    #     lambda *x: tf.data.Dataset.from_tensor_slices(x)
    # )

    def final_map(s, m, l):
        return  {'input_1': tf.reshape(s, [-1, 224, 224, 3]), 'input_2': tf.reshape(m, [-1, 1])}, tf.reshape(l, [-1])
    dataset = dataset.map(final_map, num_parallel_calls=16)
    return dataset


def fraction_positives(y_true, y_pred):
    return tf.keras.backend.mean(y_true)


def create_model(input_shape, weights):

    # with tf.device('/cpu:0'):
    input_layer = Input(shape=input_shape)
    input_mask = Input(shape=(input_shape[0], 1))
    # reshape = Reshape([224, 224, 3])(input_layer)
    mobilenet = MobileNetV2(include_top=False, weights=weights,
        input_shape=input_shape[-3:],
        pooling='avg'
        # pooling=None
    )

    # for layer in mobilenet.layers:
    #     if ('block_16' not in layer.name): # and ('block_15' not in layer.name): # and ('block_14' not in layer.name):
    #         layer.trainable = False
    #     else:
    #         print('Layer {} trainable {}'.format(layer.name, layer.trainable))
    
    # features = Conv2D(512, (1, 1), strides=(1, 1), padding='valid', activation='relu')(mobilenet.output)
    # features = Conv2D(256, (3, 3), strides=(2, 2), padding='valid', activation='relu')(features)
    # # features = MaxPooling2D(pool_size=(2, 2))(features)
    # features = GlobalMaxPooling2D()(features)
    # features = Flatten()(features)
    # # features = Dropout(0.25)(features)

    # feature_extractor = Model(inputs=mobilenet.input, outputs=features)
    # print(feature_extractor.summary())
    
    # cnn_output = TimeDistributed(feature_extractor)(input_layer)
    cnn_output = TimeDistributed(mobilenet)(input_layer)
    net = multiply([cnn_output, input_mask])
    net = Masking(mask_value = 0.0)(net)
    net = GRU(256, return_sequences=False)(net)
    net = Dense(512, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(net)
    net = Dropout(0.25)(net)
    out = Dense(1, #activation='sigmoid', 
        bias_initializer=tf.keras.initializers.Constant(0.83))(net)

    model = Model(inputs=[input_layer, input_mask], outputs=out)
    
    # This saturates GPU:0 for large models
    # parallel_model = multi_gpu_model(model, cpu_merge=False, gpus=4)
    parallel_model = multi_gpu_model(model, gpus=4)
    print("Training using multiple GPUs..")

    # def get_lr_metric(optimizer):
    #     def lr(y_true, y_pred):
    #         return optimizer.lr
    #     return lr

    optimizer = tf.keras.optimizers.Adam(lr=0.01)
    # TODO keras needs custom_objects when loading models with custom metrics
    # But this one needs the optimizer.
    # lr_metric = get_lr_metric(optimizer)
    METRICS = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'), 
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        'binary_crossentropy',
        fraction_positives,
        # lr_metric,
    ]
    my_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)
    parallel_model.compile(loss=my_loss, optimizer=optimizer, metrics=METRICS)
    print(parallel_model.summary())

    return parallel_model

    # model.compile(loss='binary_crossentropy', optimizer='adam', 
    #     metrics=['accuracy', tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(),
    #          tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()])
    # print(model.summary())

    # return model


def save_loss(H):
    plt.figure()
    N = len(H.history["loss"])
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("loss.jpg")


def step_decay(epoch):
    initial_lrate = 0.02 # 0.045
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

    train_dataset = input_dataset(args.train_dir)
    eval_dataset = input_dataset(args.eval_dir)
 
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
        }
        model = tf.keras.models.load_model(args.load, custom_objects=custom_objs)
    else:
        print('Training model from scratch.')
        model = create_model(in_shape,
            'pretrained/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5')

    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(16).prefetch(2)
    eval_dataset = eval_dataset.batch(4).prefetch(2)
    
    num_epochs = 5

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
            patience=3,
            verbose=1),
        tf.keras.callbacks.CSVLogger('mobgru_log.csv'),
        tf.keras.callbacks.LearningRateScheduler(step_decay),
        # CosineAnnealingScheduler(T_max=num_epochs, eta_max=0.045, eta_min=1e-4),
    ]
    
    class_weight={0: 0.999, 1: 0.001}
    # class_weight=[0.99, 0.01]
    history = model.fit(train_dataset, epochs=num_epochs, class_weight=class_weight, 
        validation_data=eval_dataset, validation_steps=40, callbacks=callbacks)
    save_loss(history)

    t1 = time.time()
    print("Execution took: {}".format(t1-t0))

