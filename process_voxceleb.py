import argparse
import cv2
import glob
from multiprocessing import Pool
import numpy as np
import os
import time


CMDLINE_ARGUMENTS = None

def check_shape_and_save(img_path, output_dir, min_face_size, face_resize):

    image0 = cv2.imread(img_path)
    if image0.shape[0] >= min_face_size or image0.shape[1] >= min_face_size:
        image0 = cv2.resize(image0, (face_resize, face_resize), interpolation = cv2.INTER_LINEAR)
        fname, _ = os.path.splitext(os.path.basename(img_path))
        hash_name = os.path.dirname(img_path).split('/')[-1]
        image0_file = os.path.join(output_dir, hash_name + '_' + fname + '.png')
        cv2.imwrite(image0_file, image0)
        return 1
    return 0


def process_actor(actor_dir, output_dir, min_face_size, face_resize):

    count = 0
    hash_dirs = glob.glob(actor_dir + '/*/*')
    for hash_dir in hash_dirs:
        images = glob.glob(hash_dir + '/*.jpg')
        if len(images) > 0:
            count += check_shape_and_save(images[0], output_dir, min_face_size, face_resize)
            if len(images) > 1:
                count += check_shape_and_save(images[-1], output_dir, min_face_size, face_resize)
    return count


def process_actor_vggface(actor_dir, output_dir, min_face_size, face_resize):

    count = 0
    images = glob.glob(actor_dir + '/*.jpg')
    for image in images:
        count += check_shape_and_save(image, output_dir, min_face_size, face_resize)

    return count


def run(actor_dir):

    # VoxCeleb
    # cnt = process_actor(actor_dir, CMDLINE_ARGUMENTS.output_dir, CMDLINE_ARGUMENTS.min_face_size, 256)
    # VggFace2
    cnt = process_actor_vggface(actor_dir, CMDLINE_ARGUMENTS.output_dir, CMDLINE_ARGUMENTS.min_face_size, 256)
    return cnt


def main():

    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dirs', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--min_face_size', type=int, default=192)
    parser.add_argument('--num_processes', type=int, default=10)
    args = parser.parse_args()
    
    global CMDLINE_ARGUMENTS
    CMDLINE_ARGUMENTS = args

    dirs = glob.glob(args.input_dirs)
    os.makedirs(args.output_dir, exist_ok=True)

    with Pool(args.num_processes) as pool:
        count = pool.map(run, dirs)
    print('Extracted %d images to %s.' % (np.sum(count), args.output_dir))

    t1 = time.time()
    print("Execution took: {}".format(t1-t0))


if __name__ == '__main__':
    main()
