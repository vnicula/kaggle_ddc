import argparse
import cv2
from facenet_pytorch import MTCNN
import json
import iou_tracker
import math
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os
import pickle
import tensorflow as tf
import time
import torch
import tqdm

META_DATA = "metadata.json"
MARGIN = 16
MAX_DETECTION_SIZE = 960
TRAIN_FACE_SIZE = 224
TRAIN_FRAME_COUNT = 32
TRAIN_FPS = 3
# SKIP_INITIAL_SEC = 8
SKIP_INITIAL_SEC = 0

SEQ_LEN = 30
FEAT_SHAPE = (224, 224, 3)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
detector = MTCNN(device=device, margin=MARGIN, min_face_size=20, post_process=False, keep_all=False, select_largest=False)

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


def parse_vid(video_path, max_detection_size, max_frame_count, sample_fps):
    vidcap = cv2.VideoCapture(video_path)
    frame_num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = np.int32(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)) # float
    height = np.int32(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
    print('cv2.FRAME_COUNT {}, cv2.PROP_FPS {}, cv2.FRAME_WIDTH {}, cv2.FRAME_HEIGHT {}'.format(frame_num, fps, width, height))
    
    skip_n = max(math.floor(fps / sample_fps), 0)
    max_dimension = max(width, height)
    img_scale = 1.0
    if max_dimension > max_detection_size:
        img_scale = max_detection_size / max_dimension
    print('Skipping %1.1f frames, scaling: %1.4f' % (skip_n, img_scale))

    imrs = []
    imgs = []
    count = 0

    #TODO make this robust to video reading errors
    for i in range(frame_num):
        success = vidcap.grab()
            
        if success:
            if i < SKIP_INITIAL_SEC * fps:
                continue
            if i % (skip_n+1) == 0:
                success, im = vidcap.retrieve()
                if success:
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    if img_scale < 1.0:
                        imr = cv2.resize(im, (int(im.shape[1] * img_scale), int(im.shape[0] * img_scale)))
                    else:
                        imr = im
                    imgs.append(im)
                    imrs.append(imr)
                    count += 1
                    if count >= max_frame_count:
                        break
        else:
            break

    vidcap.release()
    return imgs, imrs, img_scale


def detect_faces_bbox(detector, label, originals, images, batch_size, img_scale, face_size):
    faces = []
    detections = []

    for lb in np.arange(0, len(images), batch_size):
        imgs_pil = [Image.fromarray(image) for image in images[lb:lb+batch_size]]
        frames_boxes, frames_confidences = detector.detect(imgs_pil, landmarks=False)
        if (frames_boxes is not None) and (len(frames_boxes) > 0):
            print(frames_boxes, frames_confidences)
            for i in range(len(frames_boxes)):
                if frames_boxes[i] is not None:
                    boxes = []
                    for box, confidence in zip(frames_boxes[i], frames_confidences[i]):
                        boxes.append({'bbox': box, 'score':confidence})
                    detections.append(boxes)
    
    tracks = iou_tracker.track_iou(detections, 0.8, 0.9, 0.1, 10)

    # Can't use anything since it's multitrack fake
    if label == 1 and len(tracks) > 1:
        return faces

    tracks.sort(key = lambda x:x['max_score'], reverse=True)
    print(tracks)
    for track in tracks[:2]:
        track_faces = []
        for i, bbox in enumerate(track['bboxes']):
            original = originals[track['start_frame'] + i - 1]
            (x,y,w,h) = (
                max(int(bbox[0] / img_scale) - MARGIN, 0),
                max(int(bbox[1] / img_scale) - MARGIN, 0),
                int((bbox[2]-bbox[0]) / img_scale) + 2*MARGIN,
                int((bbox[3]-bbox[1]) / img_scale) + 2*MARGIN
            )
            face_extract = original[y:y+h, x:x+w].copy() # Without copy() memory leak with GPU
            face_extract = cv2.resize(face_extract, (face_size, face_size))
            track_faces.append(face_extract)
        faces.append(track_faces)

    return faces


def extract_one_sample_bbox(video_path, label, max_detection_size, max_frame_count, face_size):
    """Returns a 4d numpy with the face sequence"""
    start = time.time()
    imgs, imrs, img_scale = parse_vid(video_path, max_detection_size, max_frame_count, TRAIN_FPS)
    parsing = time.time() - start
    # faces = detect_facenet_pytorch(detector, imgs, 256)
    faces = detect_faces_bbox(detector, label, imgs, imrs, 256, img_scale, face_size)
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
        data_seq_len = min(my_seq_len, SEQ_LEN)
        sample = np.zeros((SEQ_LEN,) + FEAT_SHAPE, dtype=np.float32)
        sample_f = np.zeros((SEQ_LEN,) + FEAT_SHAPE, dtype=np.float32)
        mask = np.zeros(SEQ_LEN, dtype=np.float32)
        for indx in range(data_seq_len):
            sample[indx] = data[key][1][indx].astype(np.float32) / 127.5 - 1.0
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


def run(input_dir, slice_size, already_processed, first_slice):

    f_list = os.listdir(input_dir)
    slice_prefix = os.path.basename(input_dir).split('_')[-1]
    dataset_slice = {}
    slices = first_slice
    with open(os.path.join(input_dir, META_DATA)) as json_file:
        label_data = json.load(json_file)

    for i in tqdm.tqdm(range(len(f_list))):
        f_name = f_list[i]
        # Parse video
        f_path = os.path.join(input_dir, f_name)
        file_name = os.path.basename(f_path)
        
        if file_name in already_processed:
            print('Skipping already processed {}'.format(file_name))
            continue
        
        print('Now processing: ' + f_path)
        suffix = f_path.split('.')[-1]

        if suffix.lower() in ['mp4', 'avi', 'mov']:
            # Parse video
            label = 1 if label_data[file_name]['label'] == 'FAKE' else 0
            # label = 0

            faces = extract_one_sample_bbox(f_path, label, max_detection_size=MAX_DETECTION_SIZE, 
                max_frame_count=TRAIN_FRAME_COUNT, face_size=TRAIN_FACE_SIZE)
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

def get_processed_keys(input_dir):
    processed_files = set()
    f_list = os.listdir(input_dir)
    dataset_files = [fn for fn in f_list if fn.endswith('pkl')]
    last_slice = 0
    for data_file in dataset_files:
        suffix = int(data_file.split('_')[-1].split('.')[0])
        if suffix > last_slice:
            last_slice = suffix
        with open(os.path.join(input_dir, data_file), 'rb') as f_p:
            data = pickle.load(f_p)
            processed_files |= data.keys()
    
    return processed_files, last_slice


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

    already_processed, last_slice = get_processed_keys(args.input_dir)
    # print(already_processed, last_slice)
    run(args.input_dir, 128, already_processed, last_slice+1)

    t1 = time.time()
    print("Execution took: {}".format(t1-t0))

    del detector
    torch.cuda.empty_cache()
