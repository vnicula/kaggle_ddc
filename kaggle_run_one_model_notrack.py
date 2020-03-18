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
import efficientnet.tfkeras
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
MARGIN = 28
MAX_DETECTION_SIZE = 1280
TRAIN_FACE_SIZE = 256
TRAIN_FRAME_COUNT = 31
TRAIN_FPS = 3

MIN_FACE_CONFIDENCE = 0.9
# FEAT_SHAPE = (TRAIN_FACE_SIZE, TRAIN_FACE_SIZE, 3)
# SEQ_LEN = 30
# MIN_TRACK_FACES = 5
MIN_FACE_SIZE = 32

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
detector = MTCNN(device=device, margin=MARGIN, min_face_size=MIN_FACE_SIZE, post_process=False, keep_all=True, select_largest=False)
INPUT_DIR = '/kaggle/input/deepfake-detection-challenge'
input_dir = INPUT_DIR


def parse_vid(video_path, max_detection_size, max_frame_count, sample_fps):
    vidcap = cv2.VideoCapture(video_path)
    frame_num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = np.int32(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)) # float
    height = np.int32(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float

#     NOTE: I think for some simple, non temporal models it's better to sample fixed number of frames.
#     skip_n = max(math.floor(fps / sample_fps), 0)
    skip_n = max(math.floor(frame_num / TRAIN_FRAME_COUNT), 0)
    
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


def detect_faces_bbox(detector, originals, images, batch_size, img_scale, face_size):
    faces = []

    for lb in np.arange(0, len(images), batch_size):
        imgs_pil = [Image.fromarray(image) for image in images[lb:lb+batch_size]]
        frames_boxes, frames_confidences = detector.detect(imgs_pil, landmarks=False)
        if (frames_boxes is not None) and (len(frames_boxes) > 0):
#             print(frames_boxes, frames_confidences)
            for i in range(len(frames_boxes)):
                if frames_boxes[i] is not None:
                    for box, confidence in zip(frames_boxes[i][:2], frames_confidences[i][:2]):
                        if confidence > MIN_FACE_CONFIDENCE:
                            (x,y,w,h) = (
                                max(int(box[0] / img_scale) - MARGIN, 0),
                                max(int(box[1] / img_scale) - MARGIN, 0),
                                int((box[2]-box[0]) / img_scale) + 2*MARGIN,
                                int((box[3]-box[1]) / img_scale) + 2*MARGIN
                            )
                            original = originals[i]
                            face_extract = original[y:y+h, x:x+w].copy() # Without copy() memory leak with GPU
                            face_extract = cv2.resize(face_extract, (face_size, face_size), interpolation=cv2.INTER_LINEAR)
#                             face_extract = square_resize(face_extract, face_size)
                            faces.append(face_extract)

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


def run(file_list, model_file):

    prediction_list = []
    # Note: Kaggle renames the model folder behind my back
    # model = load_model('/kaggle/input/featureextractormodel/one_model.h5', custom_objects=custom_objs)
    model = load_model(model_file, custom_objects=custom_objs)
#     print(model.summary())
#     score_calibrator = joblib.load('/kaggle/input/featureextractormodel/score_calibration.pkl')
    len_file_list = len(file_list)
    for i in range(len_file_list):
        f_name = os.path.basename(file_list[i])
        prediction = 0.5
        try:
            faces = extract_one_sample_bbox(file_list[i], max_detection_size=MAX_DETECTION_SIZE, 
                max_frame_count=TRAIN_FRAME_COUNT, face_size=TRAIN_FACE_SIZE)
            print('Now processing: {}, faces {} progress {}/{}'.format(file_list[i], len(faces), i, len_file_list))
            if len(faces) > 0:
                # prediction_faces = [item for sublist in faces for item in sublist]
                # prediction_faces = np.array(prediction_faces, dtype=np.float32) / 255.0
                prediction_faces = np.array(faces, dtype=np.float32)
                prediction_faces = efficientnet.tfkeras.preprocess_input(prediction_faces)
#                 prediction_faces = np.array(prediction_faces, dtype=np.float32) / 127.5 - 1.0            
                model_prediction = model.predict(prediction_faces).flatten()
#                 print('model preds: ', model_prediction)
                if len(model_prediction) > 0:
                    prediction = model_prediction.mean()
#                     prediction = score_calibrator.predict_proba(prediction.reshape(-1, 1))[:,1]
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
    import sys
    t0 = time.time()
    test_path = os.path.join(sys.argv[1], 'test_videos/*.mp4')
    print(test_path)
    file_paths = glob.glob(test_path)
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

    predictions = run(file_paths, sys.argv[2])
#     print(predictions)
    save_predictions(predictions)
    t1 = time.time()
    print("Execution took: {}".format(t1-t0))

    torch.cuda.empty_cache()
