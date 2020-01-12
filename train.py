from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import tensorflow as tf
import time

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Input, GRU, Reshape, TimeDistributed

# Needed because keras model.fit shape checks are weak
# https://github.com/tensorflow/tensorflow/issues/24520
# tf.enable_eager_execution()

SEQ_LEN = 3

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


# TODO investigate preprocessing of pixel values for pretrained MobileNet
# TODO do padding with mask (and masking layer in the model)
# TODO do sample weighting and oversample REAL
def read_file(file_path):
    
    names = []
    labels = []
    samples = []
    
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f_p:
            data = pickle.load(f_p)
            for key in data.keys():
                names.append(key)
                labels.append(data[key][0])
                # labels.append(int(data[key][0] == 'FAKE'))
                # sample = data[key][1][0]
                sample = np.array(data[key][1][:SEQ_LEN])
                sample = preprocess_input(sample.astype(np.float32))
                # print(sample.shape)
                samples.append(sample)

    # Not sure why can't I do this here instead of py func
    # dataset = tf.data.Dataset.from_tensor_slices((names, labels, samples))
    npsamples = np.array(samples, dtype=np.float32)
    nplabels = np.array(labels, dtype=np.float32)
    print('file {} Shape samples {}, labels {}'.format(file_path, npsamples.shape, nplabels.shape))
    return npsamples, nplabels

def input_dataset(input_dir):
    # dataset = tf.data.Dataset.list_files(input_dir)
    f_list = os.listdir(input_dir)
    dataset_files = [os.path.join(input_dir, fn) for fn in f_list if fn.endswith('pkl')]
    dataset = tf.data.Dataset.from_tensor_slices((dataset_files))
    
    dataset = dataset.flat_map(
        lambda file_name: tf.data.Dataset.from_tensor_slices(
            tuple(tf.py_func(read_file, [file_name], [tf.float32, tf.float32]))
        )
    )
    dataset = dataset.map(lambda s, l: (tf.reshape(s, [-1, 224, 224, 3]), tf.reshape(l, [-1])))
    return dataset

def create_model(input_shape, weights):

    input_layer = Input(shape=input_shape)
    # reshape = Reshape([224, 224, 3])(input_layer)
    mobilenet = MobileNetV2(include_top=False, weights=weights,
        input_shape=input_shape[-3:], pooling='avg')
    for layer in mobilenet.layers:
        layer.trainable = False
    # net = mobilenet(input_layer)
    net = TimeDistributed(mobilenet)(input_layer)
    net = GRU(128, return_sequences=False)(net)
    out = Dense(1, activation='sigmoid')(net)

    model = Model(inputs=input_layer, outputs=out)
    model.compile(loss='binary_crossentropy', optimizer='adam', 
        metrics=['accuracy', #tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(),
             tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()])
    print(model.summary())

    return model


def save_loss(H, N):
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    # plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    # plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("loss.jpg")


if __name__ == '__main__':

    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    args = parser.parse_args()

    dataset = input_dataset(args.input_dir)
 
    # elem = dataset.make_one_shot_iterator().get_next()
    # with tf.Session() as session:
    #     npelem = session.run(elem)
    #     print(npelem[0], npelem[1])
    #     print(npelem[0].shape)

    in_shape = (SEQ_LEN, 224, 224, 3)

    # in_shape = (224, 224, 3)
    model = create_model(in_shape,
        'pretrained/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5')

    dataset = dataset.shuffle(buffer_size=256).batch(16)
    num_epochs = 2
    history = model.fit(dataset, epochs=num_epochs)
    save_loss(history, num_epochs)

    t1 = time.time()
    print("Execution took: {}".format(t1-t0))

