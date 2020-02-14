# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import cv2
import gc
import glob
import joblib
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time
import tensorflow as tf
import torch

from facenet_pytorch import MTCNN
from PIL import Image
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import load_model
import tensorflow.keras as keras
import tensorflow.keras.backend as K

print('TF version:', tf.__version__)
print('Keras version:', tf.keras.__version__)
print('Torch version:', torch.__version__)

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

META_DATA = "metadata.json"
MARGIN = 16
MAX_DETECTION_SIZE = 960
SEQ_LEN = 30
TRAIN_FACE_SIZE = 256
TRAIN_FRAME_COUNT = 32
TRAIN_FPS = 3

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
detector = MTCNN(device=device, margin=MARGIN, min_face_size=20, post_process=False, keep_all=False, select_largest=False)

INPUT_DIR = '/kaggle/input/deepfake-detection-challenge'
input_dir = INPUT_DIR

def parse_vid(video_path, max_detection_size, max_frame_count, sample_fps):
    vidcap = cv2.VideoCapture(video_path)
    frame_num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = np.int32(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)) # float
    height = np.int32(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float

    skip_n = max(math.floor(fps / sample_fps), 0)
    max_dimension = max(width, height)
    img_scale = 1.0
    if max_dimension > max_detection_size:
        img_scale = max_detection_size / max_dimension
#     print('Frame count %d, skipping %1.1f frames, scaling: %1.4f' % (frame_num, skip_n, img_scale))

    imrs = []
    imgs = []
    count = 0

    for i in range(frame_num):
        success = vidcap.grab()            
        if success:
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


def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.

    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.

    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """

    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union


def track_iou(detections, sigma_l, sigma_h, sigma_iou, t_min):
    """
    Simple IOU based tracker.
    See "High-Speed Tracking-by-Detection Without Using Image Information by E. Bochinski, V. Eiselein, T. Sikora" for
    more information.

    Args:
         detections (list): list of detections per frame, usually generated by util.load_mot
         sigma_l (float): low detection threshold.
         sigma_h (float): high detection threshold.
         sigma_iou (float): IOU threshold.
         t_min (float): minimum track length in frames.

    Returns:
        list: list of tracks.
    """

    tracks_active = []
    tracks_finished = []

    for frame_num, detections_frame in enumerate(detections, start=1):
        # apply low threshold to detections
        dets = [det for det in detections_frame if det['score'] >= sigma_l]

        updated_tracks = []
        for track in tracks_active:
            if len(dets) > 0:
                # get det with highest iou
                best_match_index = 0
                best_iou = 0
                for i, det in enumerate(dets):
                    candidate_iou = iou(track['bboxes'][-1], det['bbox'])
                    if candidate_iou > best_iou:
                        best_match_index = i
                        best_iou = candidate_iou
                if best_iou >= sigma_iou:
                    best_match = dets[best_match_index]
                    track['bboxes'].append(best_match['bbox'])
                    track['max_score'] = max(track['max_score'], best_match['score'])

                    updated_tracks.append(track)
                    # print('best match: ', best_match)
                    # remove from best matching detection from detections
                    del dets[best_match_index]

            # if track was not updated
            if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                # finish track when the conditions are met
                if track['max_score'] >= sigma_h and len(track['bboxes']) >= t_min:
                    tracks_finished.append(track)

        # create new tracks
        new_tracks = [{'bboxes': [det['bbox']], 'max_score': det['score'], 'start_frame': frame_num} for det in dets]
        tracks_active = updated_tracks + new_tracks

    # finish all remaining active tracks
    tracks_finished += [track for track in tracks_active
                        if track['max_score'] >= sigma_h and len(track['bboxes']) >= t_min]

    return tracks_finished


def detect_faces_bbox(detector, originals, images, batch_size, img_scale, face_size):
    faces = []
    detections = []

    for lb in np.arange(0, len(images), batch_size):
        imgs_pil = [Image.fromarray(image) for image in images[lb:lb+batch_size]]
        frames_boxes, frames_confidences = detector.detect(imgs_pil, landmarks=False)
        if (frames_boxes is not None) and (len(frames_boxes) > 0):
#             print(frames_boxes, frames_confidences)
            for i in range(len(frames_boxes)):
                if frames_boxes[i] is not None:
                    boxes = []
                    for box, confidence in zip(frames_boxes[i], frames_confidences[i]):
                        boxes.append({'bbox': box, 'score':confidence})
                    detections.append(boxes)
#     print(detections)
    tracks = track_iou(detections, 0.8, 0.9, 0.1, 10)
    tracks.sort(key = lambda x:x['max_score'], reverse=True)
#     print(tracks)
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

    del tracks
    del detections

    return faces


def extract_one_sample_bbox(video_path, max_detection_size, max_frame_count, face_size):
    start = time.time()
    imgs, imrs, img_scale = parse_vid(video_path, max_detection_size, max_frame_count, TRAIN_FPS)
    parsing = time.time() - start
    faces = detect_faces_bbox(detector, imgs, imrs, 128, img_scale, face_size)
    del imgs, imrs
    # print('faces: ', faces)
    detection = time.time() - start - parsing
#     print('parsing: %.3f scale %f, detection: %.3f seconds' %(parsing, img_scale, detection))
    return faces


def fraction_positives(y_true, y_pred):
    return tf.keras.backend.mean(y_true)


custom_objs = {
    'fraction_positives':fraction_positives,
}


def run(file_list):

    prediction_list = []
    # Note: Kaggle renames the model folder behind my back
    # model = load_model('/kaggle/input/featureextractormodel/one_model.h5', custom_objects=custom_objs)
    model = load_model('one_model.h5', custom_objects=custom_objs)
#     print(model.summary())
    # score_calibrator = joblib.load('/kaggle/input/featureextractormodel/score_calibration.pkl')
    len_file_list = len(file_list)
    for i in range(len_file_list):
        f_name = os.path.basename(file_list[i])
        prediction = 0.5
        try:
            faces = extract_one_sample_bbox(file_list[i], max_detection_size=MAX_DETECTION_SIZE, 
                max_frame_count=TRAIN_FRAME_COUNT, face_size=TRAIN_FACE_SIZE)
            print('Now processing: {}, faces {} progress {}/{}'.format(file_list[i], len(faces), i, len_file_list))
            if len(faces) > 0:
                prediction_faces = [item for sublist in faces for item in sublist]
                prediction_faces = np.array(prediction_faces, dtype=np.float32) / 255.0            
#                 prediction_faces = np.array(prediction_faces, dtype=np.float32) / 127.5 - 1.0            
                model_prediction = model.predict(prediction_faces).flatten()
#                 print('model preds: ', model_prediction)
                if len(model_prediction) > 0:
                    prediction = model_prediction.mean()
                    prediction = score_calibrator.predict_proba(prediction.reshape(-1, 1))[:,1]
#                     prediction = np.percentile(model_prediction, 60)
                else:
                    print('Model gave no prediction!')

    #                 fig = plt.figure(figsize=(20, 24))
    #                 for i, frame_face in enumerate(faces[0]):
    #                     ax = fig.add_subplot(5, 6, i+1)
    #                     ax.axis('off')
    #                     plt.imshow(frame_face)
    #                 plt.tight_layout()
    #                 plt.show()

                del prediction_faces
            del faces

        except Exception as e:
            print(e)

        print('file: {}, prediction: {}'.format(f_name, prediction))
        prediction_list.append([f_name, prediction])
        if i % 10 == 0:  
            gc.collect()
    
    del model
    return prediction_list


def save_predictions(predictions):
    with open('submission.csv', 'w') as sf:
        sf.write('filename,label\n')
        for name, score in predictions:
#             score = np.clip(score, 0.49, 0.999) # expected 0.482
            sf.write('%s,%1.6f\n' % (name, score))


if __name__ == '__main__':

    t0 = time.time()

    # KAGGLE
    # file_paths = glob.glob(os.path.join(input_dir, 'test_videos/*.mp4'))
    file_paths = glob.glob(os.path.join(input_dir, '/raid/scratch/tf_train/dset/test_videos/*.mp4'))
    test_files = [os.path.basename(x) for x in file_paths]

#     try:
#         submission = pd.read_csv(os.path.join(input_dir, 'sample_submission.csv'))
#         csvfileset = set(submission.filename)
#         listdirset = set(test_files)
#         print('Are identical filenames in csv and test dir? ', csvfileset == listdirset)
#         print('csvfileset - listdirset', csvfileset - listdirset)
#         print('listdirset - csvfileset', listdirset - csvfileset)
#         del submission, csvfileset, listdirset
#         gc.collect()
#     except:
#         pass

    predictions = run(file_paths)
#     print(predictions)
    save_predictions(predictions)
    t1 = time.time()
    print("Execution took: {}".format(t1-t0))

    torch.cuda.empty_cache()

