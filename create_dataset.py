import argparse
import cv2
from facenet_pytorch import MTCNN
import json
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os
import pickle
import torch
import tqdm
import time

META_DATA = "metadata.json"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
detector = MTCNN(device=device, margin=10, min_face_size=20, post_process=False)

def parse_vid(video_path, img_scale, skip_n):
    vidcap = cv2.VideoCapture(video_path)
    frame_num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('cv2.CAP_PROP_FRAME_COUNT: {}'.format(frame_num))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = np.int32(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)) # float
    height = np.int32(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
    imgs = []
    count = 0
    while True:
        # success, im = vidcap.read()
        success = vidcap.grab()
        if success:
            if count % (skip_n+1) == 0:
                success, im = vidcap.retrieve()
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                imr = cv2.resize(im, (int(im.shape[1] * img_scale), int(im.shape[0] * img_scale)))
                imgs.append(imr)
            count += 1
        else:
            break

    vidcap.release()
    return imgs, fps, width, height


def detect_facenet_pytorch(detector, images, batch_size):
    faces = []
    for lb in np.arange(0, len(images), batch_size):
        imgs_pil = [Image.fromarray(image) for image in images[lb:lb+batch_size]]
        detected = detector(imgs_pil)
        if len(detected) > 0:
            faces.extend(detector(imgs_pil))
    return faces


def extract_one_sample(video_path, img_scale, skip_n):
    """Returns a 4d numpy with the face sequence"""
    start = time.time()
    imgs, _, _, _ = parse_vid(video_path, img_scale, skip_n)
    parsing = time.time() - start
    faces = detect_facenet_pytorch(detector, imgs, 256)
    faces = [i for i in faces if i is not None]
    # print('faces: ', faces)
    if len(faces) > 0:
        faces = torch.stack(faces).permute(0, 2, 3, 1).int().numpy()
    detection = time.time() - start - parsing
    print('parsing: %.3f, detection: %.3f seconds' %(parsing, detection))
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
            faces = extract_one_sample(f_path, img_scale=0.5, skip_n=4)
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
    # print(extract_one_sample('C:/Downloads/deepfake-detection-challenge/train_sample_videos/abofeumbvv.mp4', img_scale=0.5, skip_n=4))

    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    args = parser.parse_args()

    run(args.input_dir, 10)

    t1 = time.time()
    print("Execution took: {}".format(t1-t0))

    del detector
    torch.cuda.empty_cache()
