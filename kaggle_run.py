import cv2
import glob
import math
import numpy as np
import os
import pandas as pd
import time
import tensorflow as tf
import torch

from keras_utils import SeqWeightedAttention
from facenet_pytorch import MTCNN
# from mtcnn import MTCNN
from PIL import Image
from tensorflow.keras.models import load_model

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
TRAIN_FACE_SIZE = 224
TRAIN_FRAME_COUNT = 32

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
detector = MTCNN(device=device, margin=MARGIN, min_face_size=20, post_process=False, keep_all=False, select_largest=False)

def parse_vid(video_path, max_detection_size, max_frame_count):
    vidcap = cv2.VideoCapture(video_path)
    frame_num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('cv2.CAP_PROP_FRAME_COUNT: {}'.format(frame_num))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = np.int32(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)) # float
    height = np.int32(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float

    skip_n = max(math.floor(frame_num / max_frame_count), 0)
    max_dimension = max(width, height)
    img_scale = 1.0
    if max_dimension > max_detection_size:
        img_scale = max_detection_size / max_dimension
    print('Skipping %1.1f frames, scaling: %1.4f' % (skip_n, img_scale))

    imrs = []
    imgs = []
    count = 0

    #TODO make this robust to video reading errors
    while True:
        # success, im = vidcap.read()
        success = vidcap.grab()
        if success:
            if count % (skip_n+1) == 0:
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
        else:
            break

    vidcap.release()
    return imgs, imrs, img_scale


def detect_faces_bbox(detector, originals, images, batch_size, img_scale, face_size):
    faces = []
    face_locations = []
    for lb in np.arange(0, len(images), batch_size):
        imgs_pil = [Image.fromarray(image) for image in images[lb:lb+batch_size]]
        frames_boxes, frames_confidences = detector.detect(imgs_pil, landmarks=False)

        #TODO figure out a way to use all faces.
        for batch_idx, (frame_boxes, frame_confidences) in enumerate(zip(frames_boxes, frames_confidences), 0):
            # print(frame_boxes)
            original = originals[lb + batch_idx]
            # print('Original size {}, used for detection size {}'.format(original.shape, images[lb+batch_idx].shape))
            if ((frame_boxes is not None) and (len(frame_boxes) > 0)):
                selected_face_box = frame_boxes[0]
                # Distance logic is needed to weed out face-like artifact detection
                if len(face_locations) > 0:
                    min_distance = float('inf')
                    last_face_box = face_locations[-1]
                    for j, (face_box, confidence) in enumerate(zip(frame_boxes, frame_confidences), 0):
                        distance = (face_box[0] - last_face_box[0])**2 + (face_box[1] - last_face_box[1])**2
                        # print(distance)
                        if distance < min_distance:
                            min_distance = distance
                            selected_face_box = face_box
                
                face_locations.append(selected_face_box)
                (x,y,w,h) = (
                    max(int(selected_face_box[0] / img_scale) - MARGIN, 0),
                    max(int(selected_face_box[1] / img_scale) - MARGIN, 0),
                    int((selected_face_box[2]-selected_face_box[0]) / img_scale) + 2*MARGIN,
                    int((selected_face_box[3]-selected_face_box[1]) / img_scale) + 2*MARGIN
                )
                face_extract = original[y:y+h, x:x+w].copy() # Without copy() memory leak with GPU
                face_extract = cv2.resize(face_extract, (face_size, face_size))
                faces.append(face_extract)

    return faces


def extract_one_sample_bbox(video_path, max_detection_size, max_frame_count, face_size):
    start = time.time()
    imgs, imrs, img_scale = parse_vid(video_path, max_detection_size, max_frame_count)
    parsing = time.time() - start
    faces = detect_faces_bbox(detector, imgs, imrs, 256, img_scale, face_size)
    # print('faces: ', faces)
    detection = time.time() - start - parsing
    print('parsing: %.3f scale %f, detection: %.3f seconds' %(parsing, img_scale, detection))
    return faces

def fraction_positives(y_true, y_pred):
    return tf.keras.backend.mean(y_true)


custom_objs = {
    'fraction_positives':fraction_positives,
    'SeqWeightedAttention':SeqWeightedAttention,
}


def run(file_list):

    prediction_list = []
    model = load_model('candidate.h5', custom_objects=custom_objs)
    len_file_list = len(file_list)
    for i in range(len_file_list):
        f_name = os.path.basename(file_list[i])
        print('Now processing: {} {}/{}'.format(file_list[i], i, len_file_list))
        faces = extract_one_sample_bbox(file_list[i], max_detection_size=MAX_DETECTION_SIZE, 
            max_frame_count=TRAIN_FRAME_COUNT, face_size=TRAIN_FACE_SIZE)
        if len(faces) > 0:
            pred_faces = np.zeros((SEQ_LEN, TRAIN_FACE_SIZE, TRAIN_FACE_SIZE, 3), dtype=np.float32)
            mask = np.zeros(SEQ_LEN, dtype=np.float32)
            for k in range(len(faces)):
                pred_faces[k] = faces[k]
                mask[k] = 1.0
            pred_faces = pred_faces / 127.5 - 1.0
            prediction = model.predict(
                [np.expand_dims(pred_faces, axis=0), np.expand_dims(mask, axis=0)])
        else:
            prediction = 0.5
        prediction_list.append([f_name, prediction])
    
    return prediction_list

def save_predictions(predictions):
    with open('sample_submission.csv', 'w') as sf:
        sf.write('filename,label\n')
        for name, score in predictions:
            sf.write('%s,%1.6f\n' % (name, score))

if __name__ == '__main__':

    # !!! REPLACE !!!
    # input_dir = 'C:/Downloads/deepfake-detection-challenge'
    input_dir = '.'
    file_paths = glob.glob(os.path.join(input_dir, 'test_videos/*.mp4'))
    test_files = [os.path.basename(x) for x in file_paths]

    try:
        submission = pd.read_csv(os.path.join(input_dir, 'sample_submission.csv'))
        csvfileset = set(submission.filename)
        listdirset = set(test_files)
        print('Are identical filenames in csv and test dir? ', csvfileset == listdirset)
        print('csvfileset - listdirset', csvfileset - listdirset)
        print('listdirset - csvfileset', listdirset - csvfileset)    
    except:
        pass

    t0 = time.time()
    predictions = run(file_paths)
    save_predictions(predictions)
    t1 = time.time()
    print("Execution took: {}".format(t1-t0))

    del detector
    torch.cuda.empty_cache()
