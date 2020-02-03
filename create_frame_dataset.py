import argparse
import constants
import cv2
import numpy as np
import os
import process_utils
import time
import torch

from facenet_pytorch import MTCNN

# TODO: exclude fakes with two faces - there's no good way to identify the fake track

def process_pair(detector, real_vid_path, fake_vid_path, track_cache, max_fakes):
    real_imgs, real_imrs, real_scale = process_utils.parse_vid(real_vid_path, constants.MAX_DETECTION_SIZE,
        constants.TRAIN_FRAME_COUNT, constants.TRAIN_FPS, constants.SKIP_INITIAL_SEC)
    fake_imgs, fake_imrs, fake_scale = process_utils.parse_vid(fake_vid_path, constants.MAX_DETECTION_SIZE, 
        constants.TRAIN_FRAME_COUNT, constants.TRAIN_FPS, constants.SKIP_INITIAL_SEC)
    assert real_scale == fake_scale
    real_vid_name = os.path.basename(real_vid_path)
    if real_vid_name in track_cache:
        tracks = track_cache[real_vid_name]
        real_faces = process_utils.get_faces_from_tracks(real_imgs, tracks, real_scale, constants.TRAIN_FACE_SIZE)
    else:
        real_faces, tracks = process_utils.detect_faces_bbox(detector, 0, real_imgs, real_imrs, 256, 
            real_scale, constants.TRAIN_FACE_SIZE, keep_tracks=2)
    fake_faces = process_utils.get_faces_from_tracks(fake_imgs, tracks, real_scale, constants.TRAIN_FACE_SIZE)

    real_faces = [item for sublist in real_faces for item in sublist]
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
        if face_diff > 70:
            selected_fake_faces.append(fake_face)
            if len(selected_fake_faces) >= max_fakes:
                break

    return real_faces, selected_fake_faces
    

def imwrite_selected_faces(real_faces, fake_faces):
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


def run(input_dir):
    with open(os.path.join(input_dir, constants.META_DATA)) as json_file:
        label_data = json.load(json_file)
    
    for file_name in label_data:
        if label_data[file_name]['label'] == 'FAKE':
            real_file = label_data[file_name]['original']


if __name__ == '__main__':

    t0 = time.time()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)
    detector = MTCNN(device=device, margin=constants.MARGIN, min_face_size=20, post_process=False, keep_all=True, select_largest=False)

    track_cache = {}
    real_faces, fake_faces = process_pair(
        detector,
        "F:/deepfake-data/dfdc_train_part_30/lccunsxtov.mp4",
        "F:/deepfake-data/dfdc_train_part_30/rzjazgejby.mp4",
        track_cache,
        20,
    )
    imwrite_selected_faces(real_faces, fake_faces)

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input_dirs', type=str)
    # args = parser.parse_args()

    # dirs = glob.glob(args.input_dirs)
    # for dir in dirs:
    #     run(dir)

    t1 = time.time()
    print("Execution took: {}".format(t1-t0))

    del detector
    torch.cuda.empty_cache()
