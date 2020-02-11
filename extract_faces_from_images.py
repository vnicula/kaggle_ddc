import argparse
import cv2
import constants
import glob
import os
import process_utils
import time
import torch

from facenet_pytorch import MTCNN


def run_label(detector, input_dir, output_dir, label):

    writing_dir = os.path.join(output_dir, label)
    os.makedirs(writing_dir, exist_ok=True)
    file_list = glob.glob(input_dir + '/*.png')

    bsize = 64
    batch_imgs = [file_list[i * bsize:(i + 1) * bsize] for i in range((len(file_list) + bsize - 1) // bsize)] 
    for batch in batch_imgs:
        imgs = list(map(cv2.imread, batch))
        imgs = list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB), imgs))
        faces = process_utils.detect_faces_no_tracks(detector, imgs, 32, constants.TRAIN_FACE_SIZE)
        out_file_names = []
        for file_name in batch:
            name, _ = os.path.splitext(os.path.basename(file_name))
            out_file_name = name + '.png'
            out_file_name = os.path.join(writing_dir, out_file_name)
            out_file_names.append(out_file_name)
        for out_file, face in zip(out_file_names, faces):
            print('writing {}.'.format(out_file))
            cv2.imwrite(out_file, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':

    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dirs', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--label', type=str)
    args = parser.parse_args()

    detector = MTCNN(device=args.device, margin=constants.MARGIN, min_face_size=20, 
        post_process=False, keep_all=False, select_largest=True)

    dirs = glob.glob(args.input_dirs)
    label = args.label
    # print(dirs)
    for dir in dirs:
        run_label(detector, dir, args.output_dir, label)

    t1 = time.time()
    print("Execution took: {}".format(t1-t0))

    del detector
    torch.cuda.empty_cache()
