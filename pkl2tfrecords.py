import argparse

import glob
import numpy as np
import pickle
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
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))


def get_numpys(pkl_file):

    names = []
    labels = []
    samples = []
    masks = []

    with open(pkl_file, 'rb') as f_p:
        data = pickle.load(f_p)
        print('Loaded {}.'.format(file_path))

        for key in data.keys():
            label = data[key][0]
            # names.append(key)
            # sample = data[key][1][0]

            feat_shape = FEAT_SHAPE
            my_seq_len = len(data[key][1])
            data_seq_len = min(my_seq_len, SEQ_LEN)
            sample = np.zeros((SEQ_LEN,) + feat_shape, dtype=np.float32)
            sample_f = np.zeros((SEQ_LEN,) + feat_shape, dtype=np.float32)
            mask = np.zeros(SEQ_LEN, dtype=np.float32)
            for indx in range(data_seq_len):
                sample[indx] = data[key][1][indx].astype(
                    np.float32) / 127.5 - 1.0
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


def save_numpy_to_tfrecords(samples, masks, labels, names, destination_path, name):
    """Converts an entire dataset into x tfrecords where x=videos/fragmentSize.
    """

    num_videos = len(samples)

    writer = None
    feature = {}

    filename = os.path.join(destination_path,
                            name + str(current_batch_number) + '_of_' + str(
                                total_batch_number) + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)

    for video_count in range((num_videos)):

        sample = samples[video_count]
        mask = masks[video_count]
        label = labels[video_count]
        name = names[video_count]

        feature['sample'] = _floats_feature(images)
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
    for pkl_file in tqdm(pkl_file_list):
        name, sample, mask, label = get_numpys(pkl_file)
        names.extend(name)
        samples.extend(sample)
        masks.extend(mask)
        labels.extend(label)

        if len(samples) >= MAX_RECORDS:
            save_numpy_to_tfrecords(samples[:MAX_RECORDS], masks[:MAX_RECORDS, labels[:MAX_RECORDS], names[:MAX_RECORDS],
            TODO)
