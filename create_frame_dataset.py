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

REAL_TO_FAKE_RATIO = 3
KEEP_ASPECT = True
MIN_FACE_DIFF = 3.5

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
        real_faces = process_utils.get_faces_from_tracks(real_imgs, tracks, real_scale)
    else:
        real_detection = True
        real_faces, tracks = process_utils.detect_faces_bbox(detector, 0, real_imgs, real_imrs, 256, 
            real_scale, 0, keep_tracks=2)
        print('Adding {} to track cache.'.format(real_vid_name))
        track_cache[real_vid_name] = tracks
    
    real_faces = [item for sublist in real_faces for item in sublist]
    if len(tracks) > 1:
        # exclude fakes with two faces - there's no good way to identify the fake track
        # print(real_faces)
        return random.sample(real_faces, min(len(real_faces), REAL_TO_FAKE_RATIO*max_fakes)), [], real_detection
    
    if len(real_faces) > 0:
        fake_faces = process_utils.get_faces_from_tracks(fake_imgs, tracks, real_scale)
        fake_faces = [item for sublist in fake_faces for item in sublist]
        img_diffs = []
        for real_face, fake_face in zip(real_faces, fake_faces):
            img_diffs.append(np.mean(cv2.absdiff(real_face, fake_face)))

        real_faces, fake_faces, img_diffs = (list(x) for x in zip(*sorted(
            zip(real_faces, fake_faces, img_diffs),
            key=lambda pair: pair[2],
            reverse=True)
            )
        )

        selected_fake_faces = []
        for fake_face, face_diff in zip(fake_faces, img_diffs):
            print('face diff: ', face_diff)
            if len(selected_fake_faces) == 0:
                selected_fake_faces.append(fake_face)
            elif face_diff > MIN_FACE_DIFF:
                selected_fake_faces.append(fake_face)
            if len(selected_fake_faces) >= max_fakes or face_diff <= MIN_FACE_DIFF:
                break

        # NOTE these are not resized to same size
        return real_faces[:REAL_TO_FAKE_RATIO*max_fakes], selected_fake_faces, real_detection

    return [], [], real_detection    


def process_single(detector, vid_path, label, max_faces):
    imgs, imrs, scale = process_utils.parse_vid(vid_path, constants.MAX_DETECTION_SIZE,
        constants.TRAIN_FRAME_COUNT, constants.TRAIN_FPS, 3) #constants.SKIP_INITIAL_SEC)

    faces, tracks = process_utils.detect_faces_bbox(detector, label, imgs, imrs, 256, 
            scale, 0, keep_tracks=1)
    
    faces = [item for sublist in faces for item in sublist]
    
    return random.sample(faces, min(len(faces), max_faces))


def pad_images_to_same_size(images):
    """
    :param images: sequence of images
    :return: list of images padded so that all images have same width and height (max width and height are used)
    """
    width_max = 0
    height_max = 0
    for img in images:
        h, w = img.shape[:2]
        width_max = max(width_max, w)
        height_max = max(height_max, h)

    images_padded = []
    for img in images:
        h, w = img.shape[:2]
        diff_vert = height_max - h
        pad_top = diff_vert//2
        pad_bottom = diff_vert - pad_top
        diff_hori = width_max - w
        pad_left = diff_hori//2
        pad_right = diff_hori - pad_left
        img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        assert img_padded.shape[:2] == (height_max, width_max)
        images_padded.append(img_padded)

    return images_padded


def imwrite_tiled_faces(real_faces, fake_faces):
    print(len(real_faces), len(fake_faces))
    assert len(real_faces) >= len(fake_faces)
    masks = []
    real_faces = real_faces[:len(fake_faces)]
    for i in range(len(fake_faces)):
        im_real = real_faces[i]
        im_fake = fake_faces[i]
        im_diff = cv2.absdiff(im_real, im_fake)
        mask = cv2.cvtColor(im_diff, cv2.COLOR_RGB2GRAY)
        imask =  mask > 5
        canvas = np.zeros_like(im_fake, np.uint8)
        canvas[imask] = im_fake[imask]
        masks.append(canvas)
        # print(im_real, im_fake)
        # print(im_diff, np.sum(im_diff), np.mean(im_diff))
    
    if len(fake_faces) > 0:
        real_imgs = cv2.hconcat(pad_images_to_same_size(real_faces))
        fake_imgs = cv2.hconcat(pad_images_to_same_size(fake_faces))
        mask_imgs = cv2.hconcat(pad_images_to_same_size(masks))
        img = cv2.vconcat([real_imgs, fake_imgs, mask_imgs])
        cv2.imwrite('selected_faces.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def imwrite_faces(output_dir, vid_file, faces, face_size, keep_aspect=KEEP_ASPECT):
    print(len(faces))
    for i, face in enumerate(faces):
        file_name = os.path.join(output_dir, vid_file + '_' + str(i) + '.png')
        if keep_aspect:
            resized_face = process_utils.square_resize(face, face_size)
        else:
            resized_face = cv2.resize(face, (face_size, face_size))
        cv2.imwrite(file_name, cv2.cvtColor(resized_face, cv2.COLOR_RGB2BGR))


def run(detector, input_dir, max_fakes, face_size):
    with open(os.path.join(input_dir, constants.META_DATA)) as json_file:
    # with open(os.path.join('.', 'all_metadata.json')) as json_file:
        label_data = json.load(json_file)

    writing_dir_0 = os.path.join(input_dir, str(face_size), '0')
    os.makedirs(writing_dir_0, exist_ok=True)
    writing_dir_1 = os.path.join(input_dir, str(face_size), '1')
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
                    imwrite_faces(writing_dir_0, real_file, real_faces, face_size)
                imwrite_faces(writing_dir_1, file_name, selected_fake_faces, face_size)


def run_label(detector, input_dir, max_faces, label, face_size):

    writing_dir = os.path.join(input_dir, str(face_size), label)
    os.makedirs(writing_dir, exist_ok=True)
    file_list = glob.glob(input_dir + '/*.mp4')

    for file_path in tqdm.tqdm(file_list):
        file_name = os.path.basename(file_path)
        faces = process_single(detector, file_path, int(label), max_faces)
        imwrite_faces(writing_dir, file_name, faces, face_size)


if __name__ == '__main__':

    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dirs', type=str)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--label', type=str, default='json')
    parser.add_argument('--face_size', type=int, default=constants.TRAIN_FACE_SIZE)
    args = parser.parse_args()

    track_cache = {}

    detector = MTCNN(device=args.device, margin=constants.MARGIN, min_face_size=20, 
        post_process=False, keep_all=False, select_largest=False)

    dirs = glob.glob(args.input_dirs)
    label = args.label
    face_size = args.face_size

    # DEBUG stuff
    # tracks = process_single(detector, '/raid/scratch/tf_train/dset/dfdc_train_part_58/549_531.mp4', 5, 1)
    # print(tracks)
    # DEBUG stuff
    # real_faces, fake_faces, real_detection = process_pair(
    #     detector,
    #     "/raid/scratch/tf_train/dset/dfdc_train_part_0/ldtgofdaqg.mp4",
    #     "/raid/scratch/tf_train/dset/dfdc_train_part_0/kfgdvqjuzu.mp4",
    #     track_cache,
    #     5,
    # )
    # print(real_faces, fake_faces)
    # imwrite_tiled_faces(real_faces, fake_faces)

    for dir in dirs:
        if 'json' in label:
            run(detector, dir, 5, face_size)
        else:
            run_label(detector, dir, 5, label, face_size)

    t1 = time.time()
    print("Execution took: {}".format(t1-t0))

    del detector
    torch.cuda.empty_cache()
