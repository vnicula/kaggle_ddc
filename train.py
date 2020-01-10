from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import numpy as np
import os
import pickle
import tensorflow as tf
import time

from tensorflow.keras.utils import Sequence


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
                sample = np.array(data[key][1][:10])
                samples.append(sample)

    # dataset = tf.data.Dataset.from_tensor_slices((names, labels, samples))
    # dataset = tf.data.Dataset.from_tensor_slices((labels, samples))
    return np.array(labels, dtype=np.int16), np.array(samples, dtype=np.int16)
    # return np.array(samples, dtype=np.int16)

def input_dataset(input_dir):
    f_list = os.listdir(input_dir)
    dataset_files = [os.path.join(input_dir, fn) for fn in f_list if fn.endswith('pkl')]
    dataset = tf.data.Dataset.from_tensor_slices((dataset_files))
    dataset = dataset.flat_map(lambda file_name: tf.data.Dataset.from_tensor_slices(
        tuple(tf.py_func(read_file, [file_name], [tf.int16, tf.int16]))))
    return dataset


class DataGenerator(Sequence):
    """
    Sequence based data generator.
    """
    def __init__(self, file_path, to_fit=True, batch_size=32, shuffle=True):
        """Initialization

        :param file_path: path to images location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.file_path = image_path
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch

        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data

        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self._generate_X(list_IDs_temp)

        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            return X, y
        else:
            return X

    def on_epoch_end(self):
        """Updates indexes after each epoch

        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images

        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self._load_grayscale_image(self.image_path + self.labels[ID])

        return X

    def _generate_y(self, list_IDs_temp):
        """Generates data containing batch_size masks

        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        """
        y = np.empty((self.batch_size, *self.dim), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            y[i,] = self._load_grayscale_image(self.mask_path + self.labels[ID])

        return y

    def _load_grayscale_image(self, image_path):
        """Load grayscale image

        :param image_path: path to image to load
        :return: loaded image
        """
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255
        return img


if __name__ == '__main__':

    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    args = parser.parse_args()

    dataset = input_dataset(args.input_dir)
 
    elem = dataset.make_one_shot_iterator().get_next()
    with tf.Session() as session:
        print(session.run(elem))
    
    t1 = time.time()
    print("Execution took: {}".format(t1-t0))

