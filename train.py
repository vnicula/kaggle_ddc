from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import numpy as np
import os
import pickle
import tensorflow as tf
import time

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Input, GRU, Reshape, TimeDistributed

# Needed because keras model.fit shape checks are weak
# https://github.com/tensorflow/tensorflow/issues/24520
tf.enable_eager_execution()

def read_file(file_path):
    
    names = []
    labels = []
    samples = []
    
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f_p:
            data = pickle.load(f_p)
            for key in data.keys():
                names.append(key)
                # labels.append(data[key][0])
                labels.append(int(data[key][0] == 'FAKE'))
                sample = np.array(data[key][1][:4])
                # sample = data[key][1][0]
                samples.append(sample)

    # Not sure why can't I do this here instead of py func
    # dataset = tf.data.Dataset.from_tensor_slices((names, labels, samples))
    return np.array(samples, dtype=np.int16), np.array(labels, dtype=np.int16)

def input_dataset(input_dir):
    f_list = os.listdir(input_dir)
    dataset_files = [os.path.join(input_dir, fn) for fn in f_list if fn.endswith('pkl')]
    dataset = tf.data.Dataset.from_tensor_slices((dataset_files))
    dataset = dataset.flat_map(
        lambda file_name: tf.data.Dataset.from_tensor_slices(
            tuple(tf.py_func(read_file, [file_name], [tf.int16, tf.int16]))
        )
    )
    return dataset

def create_model(input_shape, weights):

    input_layer = Input(shape=input_shape)
    # reshape = Reshape([224, 224, 3])(input_layer)
    mobilenet = MobileNetV2(include_top=False, weights=weights, pooling='max')
    for layer in mobilenet.layers:
        layer.trainable = False
    # net = mobilenet(input_layer)
    net = TimeDistributed(mobilenet)(input_layer)
    net = GRU(128, return_sequences=False)(net)
    out = Dense(1, activation='sigmoid')(net)

    model = Model(inputs=input_layer, outputs=out)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    return model



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

    in_shape = (4, 224, 224, 3)
    # in_shape = (224, 224, 3)
    model = create_model(in_shape,
        'pretrained/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5')

    dataset = dataset.shuffle(buffer_size=256).batch(4)
    history = model.fit(dataset, epochs=2)

    t1 = time.time()
    print("Execution took: {}".format(t1-t0))

