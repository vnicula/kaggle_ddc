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
MARGIN = 20

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
detector = MTCNN(device=device, margin=MARGIN, min_face_size=20, post_process=False, keep_all=False, select_largest=True)

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
    for lb in np.arange(0, len(images), batch_size):
        imgs_pil = [Image.fromarray(image) for image in images[lb:lb+batch_size]]
        frames_boxes, frames_confidences = detector.detect(imgs_pil, landmarks=False)
        #TODO try to stay on one face by picking the min distance to previous
        for batch_idx, (frame_boxes, frame_confidences) in enumerate(zip(frames_boxes, frames_confidences), 0):
            original = originals[lb + batch_idx]
            # print('Original size {}, used for detection size {}'.format(original.shape, images[lb+batch_idx].shape))
            if ((frame_boxes is not None) and (len(frame_boxes) > 0)):
                # frame_locations = []
                for j, (face_box, confidence) in enumerate(zip(frame_boxes, frame_confidences), 0):
                    (x,y,w,h) = (
                        max(int(face_box[0] / img_scale) - MARGIN, 0),
                        max(int(face_box[1] / img_scale) - MARGIN, 0),
                        int((face_box[2]-face_box[0]) / img_scale) + 2*MARGIN,
                        int((face_box[3]-face_box[1]) / img_scale) + 2*MARGIN
                    )
                    face_extract = original[y:y+h, x:x+w].copy() # Without copy() memory leak with GPU
                    face_extract = cv2.resize(face_extract, (face_size, face_size))
                    faces.append(face_extract)
                    #TODO figure out a way to use all faces; for now I keep largest.
                    break

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

def run(input_dir, slice_size):

    f_list = os.listdir(input_dir)
    slice_prefix = os.path.basename(input_dir)
    dataset_slice = {}
    slices = 1
    with open(os.path.join(input_dir, META_DATA)) as json_file:
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
            # faces = extract_one_sample(f_path, img_scale=0.5, skip_n=4)
            faces = extract_one_sample_bbox(f_path, max_detection_size=960, max_frame_count=32, face_size=224)
            if len(faces) > 0:
                label = label_data[file_name]['label']
                dataset_slice[file_name] = (label, faces)
            print('name: {}, faces {}, label {}'.format(file_name, len(faces), label))
        
        if len(dataset_slice) >= slice_size:
            slice_name = input_dir + "/" + "%s_%s.pkl" % (slice_prefix, slices)
            with open(slice_name, "wb") as file_slice:
                print('writing dataset slice %s' % slice_name)
                pickle.dump(dataset_slice, file_slice)
            dataset_slice = {}
            slices += 1
    
    if len(dataset_slice) > 0 and len(dataset_slice) < slice_size:
        with open(input_dir + "/" + "%s_%s" %(slice_prefix, slices),"wb") as file_slice:
            pickle.dump(dataset_slice, file_slice)
            slices += 1
    
    return slices


#TODO: detect on larger image up to original size
#return larger face area with larger margin
if __name__ == '__main__':
    # faces = extract_one_sample_bbox(
    #     # 'test_videos/atxvxouljq.mp4',
    #     'H:/Downloads/dfdc_train_part_15/ajquhoecmv.mp4',
    #     max_detection_size=960, max_frame_count=32, face_size=224)
    # print('Faces detected: {}, face[0] shape: {}'.format(len(faces), faces[0].shape))
    # plt.imshow(faces[1])
    # plt.show()

    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    args = parser.parse_args()

    run(args.input_dir, 128)

    t1 = time.time()
    print("Execution took: {}".format(t1-t0))

    del detector
    torch.cuda.empty_cache()
