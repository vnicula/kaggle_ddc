import argparse
import cv2
from facenet_pytorch import MTCNN
import json
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


def detect_facenet_pytorch(detector, images, batch_size):
    faces = []
    for lb in np.arange(0, len(images), batch_size):
        imgs_pil = [Image.fromarray(image) for image in images[lb:lb+batch_size]]
        detected = detector(imgs_pil)
        if len(detected) > 0:
            faces.extend(detector(imgs_pil))
    return faces


def detect_faces_bbox(detector, originals, images, batch_size, img_scale, face_size):
    faces = []
    face_locations = []
    for lb in np.arange(0, len(images), batch_size):
        imgs_pil = [Image.fromarray(image) for image in images[lb:lb+batch_size]]
        frames_boxes, frames_confidences = detector.detect(imgs_pil, landmarks=False)
        # Hope for no artifacts in 1st frame
        # print(frames_boxes)
        skip_batch = True
        if (frames_boxes is not None) and (len(frames_boxes) > 0):
            # Find first frame with faces:
            for i in range(len(frames_boxes)):
                if (frames_boxes[i] is not None):
                    skip_batch = len(frames_boxes[i]) != 1
                    break
        if skip_batch:
            continue

        #TODO figure out a way to use all faces; for now I keep largest.
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


def extract_one_sample_faces(video_path, max_detection_size, max_frame_count, face_size=0):
    """Returns a 4d numpy with the face sequence"""
    start = time.time()
    _, imrs, img_scale = parse_vid(video_path, max_detection_size, max_frame_count)
    parsing = time.time() - start
    faces = detect_facenet_pytorch(detector, imrs, 256)
    faces = [i for i in faces if i is not None]
    # print('faces: ', faces)
    if len(faces) > 0:
        faces = torch.stack(faces).permute(0, 2, 3, 1).int().numpy()
    detection = time.time() - start - parsing
    print('parsing: %.3f scale %f, detection: %.3f seconds' %(parsing, img_scale, detection))
    return faces

def extract_one_sample_bbox(video_path, max_detection_size, max_frame_count, face_size):
    """Returns a 4d numpy with the face sequence"""
    start = time.time()
    imgs, imrs, img_scale = parse_vid(video_path, max_detection_size, max_frame_count)
    parsing = time.time() - start
    # faces = detect_facenet_pytorch(detector, imgs, 256)
    faces = detect_faces_bbox(detector, imgs, imrs, 256, img_scale, face_size)
    # print('faces: ', faces)
    detection = time.time() - start - parsing
    print('parsing: %.3f scale %f, detection: %.3f seconds' %(parsing, img_scale, detection))
    return faces

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
            # faces = extract_one_sample(f_path, img_scale=0.5, skip_n=4)
            faces = extract_one_sample_bbox(f_path, max_detection_size=960, 
                max_frame_count=TRAIN_FRAME_COUNT, face_size=TRAIN_FACE_SIZE)
            label = 1 if label_data[file_name]['label'] == 'FAKE' else 0
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
    # faces = extract_one_sample_bbox(
    #     # 'test_videos/atxvxouljq.mp4',
    #     # 'H:/Downloads/dfdc_train_part_15/ajquhoecmv.mp4',
    #    'H:/Downloads/dfdc_train_part_15\gobvnzkjaf.mp4',
    #     max_detection_size=960, max_frame_count=TRAIN_FRAME_COUNT, face_size=TRAIN_FACE_SIZE)
    # print('Faces detected: {}'.format(len(faces)))
    # for i in range(min(20, len(faces))):
    #     plt.imshow(faces[i])
    #     plt.show()

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
