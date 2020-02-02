import argparse
import cv2
from facenet_pytorch import MTCNN
import json
import iou_tracker
import math
from multiprocessing import Pool, current_process, Queue
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os
import pickle
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

queue = Queue()

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


def detect_faces_bbox(detector, images, batch_size, img_scale, keep_tracks):
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

    tracks.sort(key = lambda x:x['max_score'], reverse=True)
    # print(tracks)
    track_faces_sizes = []
    for track in tracks[:keep_tracks]:
        for i, bbox in enumerate(track['bboxes']):
            face_size = int(bbox[2]-bbox[0]) / img_scale
            track_faces_sizes.extend(face_size)

    return track_faces_sizes


def extract_one_sample_bbox(detector, video_path, max_detection_size, max_frame_count, keep_tracks):
    """Returns a 4d numpy with the face sequence"""
    start = time.time()
    _, imrs, img_scale = parse_vid(video_path, max_detection_size, max_frame_count, TRAIN_FPS)
    parsing = time.time() - start
    tracks_sizes = detect_faces_bbox(detector, imrs, 256, img_scale, keep_tracks)
    # print('faces: ', faces)
    detection = time.time() - start - parsing
    print('parsing: %.3f scale %f, detection: %.3f seconds' %(parsing, img_scale, detection))
    return tracks_sizes


def run(file_list, keep_tracks):

    faces_sizes = []
    
    detector = queue.get()
    for f_path in tqdm.tqdm(file_list):        
        print('Now processing: ' + f_path)
        track_sizes = extract_one_sample_bbox(detector, f_path, max_detection_size=MAX_DETECTION_SIZE, 
            max_frame_count=TRAIN_FRAME_COUNT, keep_tracks=keep_tracks)
        if len(track_sizes) > 0:
            faces_sizes.extend(track_sizes)
        print('name: {}, faces {}, avg_face_size {}'.format(f_path, len(track_sizes), np.mean(track_sizes)))
    queue.put(detector)

    return faces_sizes


if __name__ == '__main__':

    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    args = parser.parse_args()

    file_list = glob.glob(args.input)
    mapped_list = []
    chunk_size = len(file_list) // 8
    for i in range(0, len(file_list), chunk_size):
        mapped_list.append(file_list[i:i+chunk_size])

    dcount = torch.cuda.device_count()
    for i in range (dcount):
        device = 'cuda:' + str(i)
        detector = MTCNN(device=device, margin=MARGIN, min_face_size=20, post_process=False, keep_all=False, select_largest=False)
        queue.put(detector)

    pool = Pool(processes=4)
    face_sizes = pool.map(partial(run, keep_tracks=2), mapped_list)
    pool.close()

    t1 = time.time()
    print("Execution took: {}".format(t1-t0))

    torch.cuda.empty_cache()
