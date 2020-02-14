import argparse
import constants
import cv2
import glob
import json
import numpy as np
import os
import process_utils
import random
import time
import torch
import tqdm

from facenet_pytorch import MTCNN

REAL_TO_FAKE_RATIO = 1

def process_pair(detector, real_vid_path, fake_vid_path, track_cache, max_fakes):
    real_imgs, real_imrs, real_scale = process_utils.parse_vid(real_vid_path, constants.MAX_DETECTION_SIZE,
        constants.TRAIN_FRAME_COUNT, constants.TRAIN_FPS, constants.SKIP_INITIAL_SEC)
    fake_imgs, fake_imrs, fake_scale = process_utils.parse_vid(fake_vid_path, constants.MAX_DETECTION_SIZE, 
        constants.TRAIN_FRAME_COUNT, constants.TRAIN_FPS, constants.SKIP_INITIAL_SEC)
    assert real_scale == fake_scale
    real_vid_name = os.path.basename(real_vid_path)
    if real_vid_name in track_cache:
        print('Found {} in track cache, skipping detection.'.format(real_vid_name))
        real_detection = False
        tracks = track_cache[real_vid_name]
        real_faces = process_utils.get_faces_from_tracks(real_imgs, tracks, real_scale, constants.TRAIN_FACE_SIZE)
    else:
        real_detection = True
        real_faces, tracks = process_utils.detect_faces_bbox(detector, 0, real_imgs, real_imrs, 256, 
            real_scale, constants.TRAIN_FACE_SIZE, keep_tracks=2)
        print('Adding {} to track cache.'.format(real_vid_name))
        track_cache[real_vid_name] = tracks
    
    real_faces = [item for sublist in real_faces for item in sublist]
    if len(tracks) > 1:
        # exclude fakes with two faces - there's no good way to identify the fake track
        # print(real_faces)
        return random.sample(real_faces, min(len(real_faces), REAL_TO_FAKE_RATIO*max_fakes)), [], real_detection
    
    if len(real_faces) > 0:
        fake_faces = process_utils.get_faces_from_tracks(fake_imgs, tracks, real_scale, constants.TRAIN_FACE_SIZE)
        fake_faces = [item for sublist in fake_faces for item in sublist]
        img_diffs = []
        for real_face, fake_face in zip(real_faces, fake_faces):
            img_diffs.append(np.mean(np.abs(real_face - fake_face)))

        real_faces, fake_faces, img_diffs = (list(x) for x in zip(*sorted(
            zip(real_faces, fake_faces, img_diffs),
            key=lambda pair: pair[2],
            reverse=True)
            )
        )

        selected_fake_faces = []
        for fake_face, face_diff in zip(fake_faces, img_diffs):
            print('face diff: ', face_diff)
            if face_diff > 80:
                selected_fake_faces.append(fake_face)
                if len(selected_fake_faces) >= max_fakes:
                    break

        return real_faces[:REAL_TO_FAKE_RATIO*max_fakes], selected_fake_faces, real_detection

    return [], [], real_detection    


def process_single(detector, vid_path, label, max_faces):
    imgs, imrs, scale = process_utils.parse_vid(vid_path, constants.MAX_DETECTION_SIZE,
        constants.TRAIN_FRAME_COUNT, constants.TRAIN_FPS, constants.SKIP_INITIAL_SEC)

    faces, tracks = process_utils.detect_faces_bbox(detector, label, imgs, imrs, 256, 
            scale, constants.TRAIN_FACE_SIZE, keep_tracks=1)
    
    faces = [item for sublist in faces for item in sublist]
    
    return random.sample(faces, min(len(faces), max_faces))


def imwrite_tiled_faces(real_faces, fake_faces):
    if len(fake_faces) > 0:
        assert len(real_faces) >= len(fake_faces)
        real_faces = real_faces[:len(fake_faces)]
        im_real = cv2.hconcat(real_faces)
        im_fake = cv2.hconcat(fake_faces)
        im_diff = cv2.absdiff(im_real, im_fake)
        mask = cv2.cvtColor(im_diff, cv2.COLOR_RGB2GRAY)
        imask =  mask > 5
        canvas = np.zeros_like(im_fake, np.uint8)
        canvas[imask] = im_fake[imask]
        im = cv2.vconcat([im_real, im_fake, canvas])
        cv2.imwrite('selected_faces.jpg', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))


def imwrite_faces(output_dir, vid_file, faces):    
    for i, face in enumerate(faces):
        file_name = os.path.join(output_dir, vid_file + '_' + str(i) + '.png')
        cv2.imwrite(file_name, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))


def run(detector, input_dir, max_fakes):
    # with open(os.path.join(input_dir, constants.META_DATA)) as json_file:
    with open(os.path.join('.', 'all_metadata.json')) as json_file:
        label_data = json.load(json_file)

    writing_dir_0 = os.path.join(input_dir, '0')
    os.makedirs(writing_dir_0, exist_ok=True)
    writing_dir_1 = os.path.join(input_dir, '1')
    os.makedirs(writing_dir_1, exist_ok=True)

    track_cache = {}
    for file_name in tqdm.tqdm(label_data):
        if label_data[file_name]['label'] == 'FAKE':
            real_file = label_data[file_name]['original']
            real_file_path = os.path.join(input_dir,real_file)
            fake_file_path = os.path.join(input_dir, file_name)
            if os.path.exists(real_file_path) and os.path.exists(fake_file_path):
                real_faces, selected_fake_faces, real_detection = process_pair(
                    detector, real_file_path, fake_file_path, track_cache, max_fakes)
                if real_detection:
                    imwrite_faces(writing_dir_0, real_file, real_faces)
                imwrite_faces(writing_dir_1, file_name, selected_fake_faces)


def run_label(detector, input_dir, max_faces, label):

    writing_dir = os.path.join(input_dir, label)
    os.makedirs(writing_dir, exist_ok=True)
    file_list = glob.glob(input_dir + '/*.mp4')

    for file_path in tqdm.tqdm(file_list):
        file_name = os.path.basename(file_path)
        faces = process_single(detector, file_path, int(label), max_faces)
        imwrite_faces(writing_dir, file_name, faces)


if __name__ == '__main__':

    t0 = time.time()

    track_cache = {}
    # real_faces, fake_faces, real_detection = process_pair(
    #     detector,
    #     "/raid/scratch/tf_train/dset/dfdc_train_part_0/wcqvzujamg.mp4",
    #     "/raid/scratch/tf_train/dset/dfdc_train_part_0/dtjcyzgdts.mp4",
    #     track_cache,
    #     5,
    # )
    # imwrite_tiled_faces(real_faces, fake_faces)

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dirs', type=str)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--label', type=str, default='json')
    args = parser.parse_args()

    detector = MTCNN(device=args.device, margin=constants.MARGIN, min_face_size=20, 
        post_process=False, keep_all=False, select_largest=False)

    faces, tracks = process_single(detector, '/raid/scratch/tf_train/dset/dfdc_train_part_58/549_531.mp4', 5, 1)
    print(tracks)

    # dirs = glob.glob(args.input_dirs)
    # label = args.label

    # for dir in dirs:
    #     if 'json' in label:
    #         run(detector, dir, 5)
    #     else:
    #         run_label(detector, dir, 10, label)

    t1 = time.time()
    print("Execution took: {}".format(t1-t0))

    del detector
    torch.cuda.empty_cache()
