import argparse
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import time
import tqdm
import tensorflow as tf

SEQ_LEN = 30
FEAT_SHAPE = (224, 224, 3)
MAX_RECORDS = 256


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))

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


def get_numpys(pkl_file):

    names = []
    samples = []
    masks = []
    labels = []

    with open(pkl_file, 'rb') as f_p:
        data = pickle.load(f_p)
        print('Loaded {}.'.format(pkl_file))

        for key in data.keys():
            label = data[key][0]

            feat_shape = FEAT_SHAPE
            my_seq_len = len(data[key][1])
            data_seq_len = min(my_seq_len, SEQ_LEN)
            sample = np.zeros((SEQ_LEN,) + feat_shape, dtype=np.float32)
            sample_f = np.zeros((SEQ_LEN,) + feat_shape, dtype=np.float32)
            mask = np.zeros(SEQ_LEN, dtype=np.float32)
            for indx in range(data_seq_len):
                sample[indx] = data[key][1][indx].astype(np.float32) / 127.5 - 1.0
                if label == 0:
                    sample_f[indx] = np.fliplr(sample[indx])
                mask[indx] = 1.0

            # print(file_path, len(samples))
            names.append(key)
            samples.append(sample)
            masks.append(mask)
            labels.append(label)

            if label == 0:
                names.append(key+'_f')
                samples.append(sample_f)
                masks.append(mask)
                labels.append(0)
                # save_sample_img(key+'_o', 0, sample)
                # save_sample_img(key+'_f', 0, sample_f)

    print('file {} samples {}, labels {}, sample shape {}.'.format(
        pkl_file, len(samples), len(labels), samples[0].shape))
    return names, samples, masks, labels


def save_numpy_to_tfrecords(names, samples, masks, labels, filename):
    """Converts numpys into tfrecords.
    """

    num_videos = len(samples)
    assert num_videos < MAX_RECORDS+1
    feature = {}
    print('Writing', filename)
    writer = tf.io.TFRecordWriter(filename)

    for i in tqdm.tqdm(range(num_videos)):

        sample = samples[i]
        mask = masks[i]
        label = labels[i]
        name = names[i].encode()
        assert sample.shape == (SEQ_LEN, ) + FEAT_SHAPE
        assert mask.shape == (SEQ_LEN, )

        feature['sample'] = _floats_feature(sample)
        feature['mask'] = _floats_feature(mask)
        feature['label'] = _int64_feature(label)
        feature['name'] = _bytes_feature(name)

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    if writer is not None:
        writer.close()


if __name__ == '__main__':

    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_dir', type=str)
    args = parser.parse_args()

    pkl_file_list = glob.glob(args.pkl_dir)
    print(pkl_file_list)

    names, samples, masks, labels = [], [], [], []
    records_batch = 0
    for pkl_file in tqdm.tqdm(pkl_file_list):
        name, sample, mask, label = get_numpys(pkl_file)
        names.extend(name)
        samples.extend(sample)
        masks.extend(mask)
        labels.extend(label)

        if len(samples) >= MAX_RECORDS:
            folder = os.path.dirname(pkl_file)
            folder_index = folder.split('_')[-1]
            tf_file = os.path.join(folder, folder_index + '_%d.tfrecord'%records_batch)
            records_batch += 1
            save_numpy_to_tfrecords(names[:MAX_RECORDS], samples[:MAX_RECORDS], masks[:MAX_RECORDS], 
                labels[:MAX_RECORDS], tf_file)
            names = names[MAX_RECORDS:]
            samples = samples[MAX_RECORDS:]
            masks = masks[MAX_RECORDS:]
            labels = labels[MAX_RECORDS:]

    if len(names) > 0:
        folder = os.path.dirname(pkl_file)
        folder_index = folder.split('_')[-1]
        tf_file = os.path.join(folder, folder_index + '_%d.tfrecord'%records_batch)
        save_numpy_to_tfrecords(names, samples, masks, labels, tf_file)

    t1 = time.time()
    print("Execution took: {}".format(t1-t0))
