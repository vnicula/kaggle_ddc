import argparse

from keras_utils import SeqWeightedAttention
import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.models import load_model, Model

def fraction_positives(y_true, y_pred):
    return tf.keras.backend.mean(y_true)

if __name__ == '__main__':

    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', type=str)
    parser.add_argument('--load', type=str)
    args = parser.parse_args()

    custom_objs = {
        'fraction_positives':fraction_positives,
        'SeqWeightedAttention':SeqWeightedAttention,
    }

    model = load_model(args.load, custom_objects=custom_objs)

    # TODO use tf model maybe as it can be retrained
    # model.save('tf_model')
    # print("Loading tf SavedModel")
    # loaded_model = load_model('tf_model', custom_objects=custom_objs, compile=False)
    # loaded_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.025), 
    #     loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05))

    preds = model.predict([np.zeros((1, 30, 224, 224, 3), dtype=np.float32), 
        np.ones((1, 30), dtype=np.float32)], verbose=1)
    print(preds)

    t1 = time.time()
    print("Execution took: {}".format(t1-t0))
