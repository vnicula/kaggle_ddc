import argparse
import constants
import cv2
from facenet_pytorch import MTCNN
import json
import math
import numpy as np
from matplotlib import pyplot as plt
import os
import pickle
import process_utils
import tensorflow as tf
import time
import torch
import tqdm


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
detector = MTCNN(device=device, margin=constants.MARGIN, min_face_size=20, post_process=False, keep_all=False, select_largest=False)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


def extract_one_sample_bbox(video_path, label, max_detection_size, max_frame_count, face_size, keep_tracks):
    """Returns a 4d numpy with the face sequence"""
    start = time.time()
    imgs, imrs, img_scale = process_utils.parse_vid(video_path, max_detection_size, 
        max_frame_count, constants.TRAIN_FPS, constants.SKIP_INITIAL_SEC)
    parsing = time.time() - start
    # faces = detect_facenet_pytorch(detector, imgs, 256)
    faces, _ = process_utils.detect_faces_bbox(detector, label, imgs, imrs, 256, img_scale, face_size, keep_tracks)
    # print('faces: ', faces)
    detection = time.time() - start - parsing
    print('parsing: %.3f scale %f, detection: %.3f seconds' %(parsing, img_scale, detection))
    return faces


def get_numpys(data):

    names = []
    samples = []
    masks = []
    labels = []

    for key in data.keys():
        label = data[key][0]

        my_seq_len = len(data[key][1])
        data_seq_len = min(my_seq_len, constants.SEQ_LEN)
        sample = np.zeros((constants.SEQ_LEN,) + constants.FEAT_SHAPE, dtype=np.float32)
        sample_f = np.zeros((constants.SEQ_LEN,) + constants.FEAT_SHAPE, dtype=np.float32)
        mask = np.zeros(constants.SEQ_LEN, dtype=np.float32)
        for indx in range(data_seq_len):
            sample[indx] = (data[key][1][indx].astype(np.float32) / 127.5) - 1.0
            if label == 0:
                sample_f[indx] = np.fliplr(sample[indx])
            mask[indx] = 1.0

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

    return names, samples, masks, labels

def save_numpy_to_tfrecords(names, samples, masks, labels, filename):
    """Converts numpys into tfrecords.
    """
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _floats_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))

    num_videos = len(samples)
    feature = {}
    print('Writing {} records to {}'.format(num_videos, filename))
    writer = tf.io.TFRecordWriter(filename)

    for i in tqdm.tqdm(range(num_videos)):

        sample = samples[i]
        mask = masks[i]
        label = labels[i]
        name = names[i].encode()
        assert sample.shape == (constants.SEQ_LEN, ) + constants.FEAT_SHAPE
        assert mask.shape == (constants.SEQ_LEN, )

        feature['sample'] = _floats_feature(sample)
        feature['mask'] = _floats_feature(mask)
        feature['label'] = _int64_feature(label)
        feature['name'] = _bytes_feature(name)

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    if writer is not None:
        writer.close()


def run(input_dir, slice_size, first_slice, keep_tracks):

    f_list = os.listdir(input_dir)
    slice_prefix = os.path.basename(input_dir).split('_')[-1]
    dataset_slice = {}
    slices = first_slice

    # with open(os.path.join(input_dir, constants.META_DATA)) as json_file:
    with open('all_metadata.json') as json_file:
        label_data = json.load(json_file)

    for i in tqdm.tqdm(range(len(f_list))):
        f_name = f_list[i]
        # Parse video
        f_path = os.path.join(input_dir, f_name)
        file_name = os.path.basename(f_path)
        
        print('Now processing: ' + f_path)
        suffix = f_path.split('.')[-1]

        if suffix.lower() in ['mp4', 'avi', 'mov']:
            # Parse video
            label = 1 if label_data[file_name]['label'] == 'FAKE' else 0
            # label = 0

            faces = extract_one_sample_bbox(f_path, label, max_detection_size=constants.MAX_DETECTION_SIZE, 
                max_frame_count=constants.TRAIN_FRAME_COUNT, face_size=constants.TRAIN_FACE_SIZE, keep_tracks=keep_tracks)
            if len(faces) > 0:
                for findex, face in enumerate(faces, start=1):
                    dataset_slice[str(findex) + file_name] = (label, face)
            print('name: {}, faces {}, label {}'.format(file_name, len(faces), label))
        
        if len(dataset_slice) >= slice_size:
            slice_name = os.path.join(input_dir, "%s_%s.tfrec" % (slice_prefix, slices))
            names, samples, masks, labels = get_numpys(dataset_slice)
            save_numpy_to_tfrecords(names, samples, masks, labels, slice_name)
            print('written file {} samples {}, labels {}, sample shape {}.'.format(
                slice_name, len(samples), len(labels), samples[0].shape))

            dataset_slice = {}
            slices += 1
    
    # Write the last incomplete slice    
    if len(dataset_slice) > 0:
        slice_name = os.path.join(input_dir, "%s_%s.tfrec" % (slice_prefix, slices))
        names, samples, masks, labels = get_numpys(dataset_slice)
        save_numpy_to_tfrecords(names, samples, masks, labels, slice_name)
        print('written file {} samples {}, labels {}, sample shape {}.'.format(
            slice_name, len(samples), len(labels), samples[0].shape))
        slices += 1

    return slices-first_slice


#TODO: detect on larger image up to original size
#return larger face area with larger margin
if __name__ == '__main__':

    # faces = extract_one_sample_bbox(
    #     'test_videos/aljjmeqszq.mp4', label=0,
    #     # 'H:/Downloads/dfdc_train_part_15/ajquhoecmv.mp4',
    #     # 'H:/Downloads/dfdc_train_part_15\gobvnzkjaf.mp4',
    #     max_detection_size=MAX_DETECTION_SIZE, max_frame_count=TRAIN_FRAME_COUNT, face_size=TRAIN_FACE_SIZE)
    # print('Faces detected: {}'.format(len(faces)))
    # for face in faces:
    #     for i in range(min(10, len(face))):
    #         plt.imshow(face[i])
    #         plt.show()

    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    args = parser.parse_args()

    run(args.input_dir, slice_size=256, first_slice=0, keep_tracks=1)

    t1 = time.time()
    print("Execution took: {}".format(t1-t0))

    del detector
    torch.cuda.empty_cache()
