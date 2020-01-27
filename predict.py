import argparse
import gc
import glob

import numpy as np
import pickle
import time
import tensorflow as tf
from keras_utils import SeqWeightedAttention


SEQ_LEN = 30


def fraction_positives(y_true, y_pred):
    return tf.keras.backend.mean(y_true)


def read_file(file_path):
    
    names = []
    labels = []
    samples = []
    masks = []
    
    with open(file_path, 'rb') as f_p:
        data = pickle.load(f_p)

        for key in data.keys():
            label = data[key][0]

            feat_shape = data[key][1][0].shape
            my_seq_len = len(data[key][1])
            data_seq_len = min(my_seq_len, SEQ_LEN)
            sample = np.zeros((SEQ_LEN,) + feat_shape, dtype=np.float32)
            mask = np.zeros(SEQ_LEN, dtype=np.float32)
            for indx in range(data_seq_len):
                sample[indx] = (data[key][1][indx].astype(np.float32) / 127.5) - 1.0
                mask[indx] = 1.0
            
            # print(file_path, len(samples))
            names.append(key)
            samples.append(sample)
            masks.append(mask)
            labels.append(label)

            # save_sample_img(key+'_o', 0, sample)
            # save_sample_img(key+'_f', 0, sample_f)
        
        del data
    # NOTE if one sample doesn't have enough frames Keras will error out here with 'assign a sequence'
    npsamples = np.array(samples, dtype=np.float32)
    npmasks = np.array(masks, dtype=np.float32)
    nplabels = np.array(labels, dtype=np.int32)

    print('file {} Shape samples {}, labels {}'.format(file_path, npsamples.shape, nplabels.shape))
    return names, npsamples, npmasks, nplabels


def save_predictions(predictions):
    with open('sample_submission.csv', 'w') as sf:
        sf.write('filename,label\n')
        for name, score in predictions:
            sf.write('%s,%1.6f\n' % (name, score))


if __name__ == '__main__':

    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', type=str)
    parser.add_argument('--model', type=str)
    args = parser.parse_args()

    custom_objs = {
        'fraction_positives':fraction_positives,
        'SeqWeightedAttention':SeqWeightedAttention,
    }

    model = tf.keras.models.load_model(args.model, custom_objects=custom_objs)
    print(model.summary())

    # TODO use tf model maybe as it can be retrained
    # model.save('tf_model')
    # print("Loading tf SavedModel")
    # loaded_model = load_model('tf_model', custom_objects=custom_objs, compile=False)
    # loaded_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.025), 
    #     loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05))

    predictions = []
    truths = []
    pkl_files = glob.glob(args.pkl)
    for pkl_file in pkl_files:
        print('Predicting on samples from {}'.format(pkl_file))
        names, npsamples, npmasks, nplabels = read_file(pkl_file)
        preds = model.predict([npsamples, npmasks], verbose=1)
        print(preds)
        predictions.extend(zip(names, preds))
        truths.extend(nplabels)
        del npsamples, npmasks
        gc.collect()

    print(predictions)
    save_predictions(predictions)
    t1 = time.time()
    print("Execution took: {}".format(t1-t0))
