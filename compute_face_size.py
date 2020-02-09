import argparse
import constants
import cv2
from facenet_pytorch import MTCNN
import functools
import glob
import json
import iou_tracker
import math
from concurrent import futures
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os
import queue
import pickle
import proceess_utils
import time
import torch
import tqdm


q = queue.Queue()


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
            track_faces_sizes.append(face_size)

    return track_faces_sizes


def extract_one_sample_bbox(video_path, max_detection_size, max_frame_count, keep_tracks):
    """Returns a 4d numpy with the face sequence"""
    start = time.time()
    _, imrs, img_scale = process_utils.parse_vid(video_path, max_detection_size, max_frame_count, TRAIN_FPS)
    parsing = time.time() - start

    detector = q.get()
    tracks_sizes = detect_faces_bbox(detector, imrs, 256, img_scale, keep_tracks)
    q.put(detector)
    # print('faces: ', faces)
    detection = time.time() - start - parsing
    print('parsing: %.3f scale %f, detection: %.3f seconds' %(parsing, img_scale, detection))
    return tracks_sizes


def run(file_list, keep_tracks):

    faces_sizes = []
    
    for f_path in tqdm.tqdm(file_list):        
        print('Now processing: ' + f_path)
        track_sizes = extract_one_sample_bbox(f_path, max_detection_size=MAX_DETECTION_SIZE, 
            max_frame_count=TRAIN_FRAME_COUNT, keep_tracks=keep_tracks)
        if len(track_sizes) > 0:
            faces_sizes.extend(track_sizes)
        print('name: {}, faces {}, avg_face_size {}'.format(f_path, len(track_sizes), np.mean(track_sizes)))

    return faces_sizes


if __name__ == '__main__':

    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    args = parser.parse_args()

    file_list = glob.glob(args.input)

    for i in range (torch.cuda.device_count()):
        device = 'cuda:' + str(i)
        detector = MTCNN(device=device, margin=MARGIN, min_face_size=20, post_process=False, keep_all=False, select_largest=False)
        q.put(detector)

    # pool = futures.ThreadPoolExecutor(max_workers=1)
    # face_sizes = pool.map(functools.partial(run, keep_tracks=2), file_list, chunksize=2)
    face_size = run(file_list, keep_tracks=2)
    print(face_size)

    t1 = time.time()
    print("Execution took: {}".format(t1-t0))

    torch.cuda.empty_cache()
