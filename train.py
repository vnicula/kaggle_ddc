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
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Input, GRU, Masking, MaxPooling2D, multiply, Reshape, TimeDistributed

# Needed because keras model.fit shape checks are weak
# https://github.com/tensorflow/tensorflow/issues/24520
# tf.enable_eager_execution()

SEQ_LEN = 30

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
# TODO oversample REAL
def read_file(file_path):
    
    names = []
    labels = []
    samples = []
    masks = []
    
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f_p:
            data = pickle.load(f_p)
            for key in data.keys():
                names.append(key)
                labels.append(data[key][0])
                # labels.append(int(data[key][0] == 'FAKE'))
                # sample = data[key][1][0]

                img_size = data[key][1][0].shape[0]
                my_seq_len = len(data[key][1])
                data_seq_len = min(my_seq_len, SEQ_LEN)
                sample = np.zeros((SEQ_LEN, img_size, img_size, 3), dtype=np.float32)
                mask = np.ones(SEQ_LEN, dtype=np.float32)
                for indx in range(data_seq_len):
                    sample[indx] = np.array(data[key][1])[indx]
                sample = preprocess_input(sample)
                if my_seq_len < SEQ_LEN:
                    sample[my_seq_len:] = np.zeros((SEQ_LEN-my_seq_len, img_size, img_size, 3), dtype=np.float32)
                    mask[my_seq_len:] = np.zeros((SEQ_LEN-my_seq_len), dtype=np.float32)
                # print(sample.shape)
                samples.append(sample)
                masks.append(mask)

    # Not sure why can't I do this here instead of py func
    # dataset = tf.data.Dataset.from_tensor_slices((names, labels, samples))
    # NOTE if one sample doesn't have enough frames Keras will error out here with 'assign a sequence'
    npsamples = np.array(samples, dtype=np.float32)
    npmasks = np.array(masks, dtype=np.float32)
    nplabels = np.array(labels, dtype=np.float32)

    print('file {} Shape samples {}, labels {}'.format(file_path, npsamples.shape, nplabels.shape))
    return npsamples, npmasks, nplabels

def input_dataset(input_dir):
    # dataset = tf.data.Dataset.list_files(input_dir)
    f_list = os.listdir(input_dir)
    dataset_files = [os.path.join(input_dir, fn) for fn in f_list if fn.endswith('pkl')]
    dataset = tf.data.Dataset.from_tensor_slices((dataset_files))
    
    dataset = dataset.flat_map(
        lambda file_name: tf.data.Dataset.from_tensor_slices(
            tuple(tf.py_func(read_file, [file_name], [tf.float32, tf.float32, tf.float32]))
        )
    )
    def final_map(s, m, l):
        return  {'input_1': tf.reshape(s, [-1, 224, 224, 3]), 'input_2': tf.reshape(m, [-1, 1])}, tf.reshape(l, [-1])
    dataset = dataset.map(final_map)
    return dataset

def create_model(input_shape, weights):

    input_layer = Input(shape=input_shape)
    input_mask = Input(shape=(input_shape[0], 1))
    # reshape = Reshape([224, 224, 3])(input_layer)
    mobilenet = MobileNetV2(include_top=False, weights=weights,
        input_shape=input_shape[-3:],
        # pooling='avg'
        pooling=None
    )
    for layer in mobilenet.layers:
        layer.trainable = False
    
    features = Conv2D(128, (3, 3), strides=(2, 2), padding='valid', activation='relu')(mobilenet.output)
    # features = MaxPooling2D(pool_size=(2, 2))(features)
    features = GlobalMaxPooling2D()(features)
    features = Flatten()(features)
    features = Dropout(0.25)(features)

    feature_extractor = Model(inputs=mobilenet.input, outputs=features)
    print(feature_extractor.summary())
    
    cnn_output = TimeDistributed(feature_extractor)(input_layer)
    net = multiply([cnn_output, input_mask])
    net = Masking(mask_value = 0.0)(net)
    net = GRU(256, return_sequences=False)(net)
    out = Dense(1, activation='sigmoid')(net)

    model = Model(inputs=[input_layer, input_mask], outputs=out)
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

    dataset = dataset.shuffle(buffer_size=).batch(16).prefetch(8)

    num_epochs = 2
    class_weight={0: 1., 1: 0.2}
    history = model.fit(dataset, epochs=num_epochs, class_weight=class_weight)
    save_loss(history, num_epochs)

    t1 = time.time()
    print("Execution took: {}".format(t1-t0))

