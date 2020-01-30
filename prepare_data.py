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
import torch
import tqdm
import time

META_DATA = "metadata.json"
MARGIN = 16
MAX_DETECTION_SIZE = 960
TRAIN_FACE_SIZE = 224
TRAIN_FRAME_COUNT = 32
TRAIN_FPS = 3
SKIP_INITIAL_SEC = 8

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
detector = MTCNN(device=device, margin=MARGIN, min_face_size=20, post_process=False, keep_all=False, select_largest=False)

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
        
        for frame_boxes, frame_confidences in zip(frames_boxes, frames_confidences):
            boxes = []
            for box, confidence in zip(frame_boxes, frame_confidences):
                boxes.append({'bbox': box, 'score':confidence})
            detections.append(boxes)
    
    tracks = iou_tracker.track_iou(detections, 0.75, 0.85, 0.1, 10)
    print(tracks)
    if label == 1 and len(tracks) > 1:
        return faces

    for track in tracks:
        track_faces = []
        for bbox in track['bboxes']:
            original = originals[track['start_frame']]
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

def run(input_dir, slice_size, already_processed, first_slice):

    f_list = os.listdir(input_dir)
    slice_prefix = os.path.basename(input_dir).split('_')[-1]
    dataset_slice = {}
    slices = first_slice
    # with open(os.path.join(input_dir, META_DATA)) as json_file:
    #     label_data = json.load(json_file)

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
            # faces = extract_one_sample(f_path, img_scale=0.5, skip_n=4)
            # label = 1 if label_data[file_name]['label'] == 'FAKE' else 0
            label = 0

            faces = extract_one_sample_bbox(f_path, label, max_detection_size=MAX_DETECTION_SIZE, 
                max_frame_count=TRAIN_FRAME_COUNT, face_size=TRAIN_FACE_SIZE)
            if len(faces) > 0:
                dataset_slice[file_name] = (label, faces)
            print('name: {}, faces {}, label {}'.format(file_name, len(faces), label))
        
        if len(dataset_slice) >= slice_size:
            slice_name = input_dir + "/" + "%s_%s.pkl" % (slice_prefix, slices)
            with open(slice_name, "wb") as file_slice:
                print('writing dataset slice %s' % slice_name)
                pickle.dump(dataset_slice, file_slice)
            dataset_slice = {}
            slices += 1
    
    # Write the last incomplete slice    
    if len(dataset_slice) > 0:
        slice_name = input_dir + "/" + "%s_%s.pkl" % (slice_prefix, slices)
        with open(slice_name, "wb") as file_slice:
            print('writing dataset slice %s' % slice_name)
            pickle.dump(dataset_slice, file_slice)
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
    faces = extract_one_sample_bbox(
        # 'test_videos/atxvxouljq.mp4',
        # 'H:/Downloads/dfdc_train_part_15/ajquhoecmv.mp4',
       'H:/Downloads/dfdc_train_part_15\gobvnzkjaf.mp4',
        max_detection_size=MAX_DETECTION_SIZE, max_frame_count=TRAIN_FRAME_COUNT, face_size=TRAIN_FACE_SIZE)
    print('Faces detected: {}'.format(len(faces)))
    for i in range(min(20, len(faces))):
        plt.imshow(faces[i])
        plt.savefig()

"""
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
