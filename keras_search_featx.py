import argparse
import autokeras as ak
import constants

from kerastuner.applications import HyperResNet
from kerastuner import RandomSearch

import os
import tensorflow as tf
import time

from keras_utils import balance_dataset


def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    sparse_label = tf.strings.to_number(
        parts[-2],
        out_type=tf.int32,
        name=None
    )
    return sparse_label


@tf.function
def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)

    return img, label


def prepare_dataset(ds, is_training, cache):

    ds = balance_dataset(ds, is_training)

    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            print('Caching dataset is_training: %s' % is_training)
            ds = ds.cache()

    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds


def input_dataset(input_dir, is_training, cache):
    list_ds = tf.data.Dataset.list_files(input_dir)
    if is_training:
        list_ds = list_ds.shuffle(buffer_size=50000)

    labeled_ds = list_ds.map(
        process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    prepared_ds = prepare_dataset(labeled_ds, is_training, cache)

    def switch_to_onehot(features, label):
        return features, tf.one_hot(label, depth=2)

    prepared_ds = prepared_ds.map(switch_to_onehot, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    prepared_ds = prepared_ds.batch(64)
    prepared_ds = prepared_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return prepared_ds


if __name__ == '__main__':

    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--eval_dir', type=str)
    args = parser.parse_args()

    train_dataset = input_dataset(
        args.train_dir, 
        is_training=True, 
        cache=False)
    eval_dataset = input_dataset(
        args.eval_dir,
        is_training=False,
        cache=True)

    in_shape = (constants.MESO_INPUT_HEIGHT, constants.MESO_INPUT_WIDTH, 3)
    hypermodel = HyperResNet(
        input_shape=in_shape,
        classes=2)

    # Initialize the hypertuner: we should find the model that maximixes the
    # validation accuracy, using 40 trials in total.
    tuner = RandomSearch(
        hypermodel,
        objective='val_loss',
        max_trials=20,
        project_name='img256_resnet',
        directory='test_directory')

    # Display search overview.
    tuner.search_space_summary()

    # Performs the hypertuning.
    tuner.search(train_dataset, epochs=10, validation_data=eval_dataset)

    # Show the best models, their hyperparameters, and the resulting metrics.
    tuner.results_summary()

    # Retrieve the best model.
    best_model = tuner.get_best_models(num_models=1)[0]

    # Evaluate the best model.
    loss, accuracy = best_model.evaluate(eval_dataset)
    print('loss:', loss)
    print('accuracy:', accuracy)

    t1 = time.time()
    print("Execution took: {}".format(t1-t0))
