import argparse
import constants
import glob
import imageio
import json
from multiprocessing import Pool
import numpy as np
import os
import random
import time
import tqdm

face_size = 256
with open(os.path.join('.', 'all_metadata.json')) as json_file:
    label_data = json.load(json_file)

def run(input_dir):

    reading_dir_0 = os.path.join(input_dir, str(face_size), '0')
    reading_dir_1 = os.path.join(input_dir, str(face_size), '1')
    writing_dir_0 = os.path.join(input_dir, str(face_size) + '_pair', '0')
    os.makedirs(writing_dir_0, exist_ok=True)
    writing_dir_1 = os.path.join(input_dir, str(face_size) + '_pair', '1')
    os.makedirs(writing_dir_1, exist_ok=True)
    file_list = glob.glob(reading_dir_1 + '/*.png')

    for path_name in tqdm.tqdm(file_list):
        file_name = os.path.basename(path_name)
        frame_name = os.path.splitext(file_name)[0]
        split_frame_name = frame_name.split('_')
        video_name = split_frame_name[0]
        frame_num = split_frame_name[1]
        assert video_name.endswith('.mp4')

        real_file = label_data[video_name]['original']
        real_path = os.path.join(reading_dir_0, real_file)
        real_path += '_' + frame_num + '.png'
        assert os.path.exists(real_path)

        fake_img = imageio.imread(path_name)
        real_img = imageio.imread(real_path)

        label = np.random.randint(2)
        if label == 0:
            sample_name = video_name + '_' + frame_num + '_' + real_file + '.png'
            sample_file = os.path.join(writing_dir_0, sample_name)
            imageio.imwrite(sample_file, np.concatenate([fake_img, real_img], axis=1))
        else:
            sample_name = real_file + '_' + frame_num + '_' + video_name + '.png'
            sample_file = os.path.join(writing_dir_1, sample_name)
            imageio.imwrite(sample_file, np.concatenate([real_img, fake_img], axis=1))


if __name__ == '__main__':

    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dirs', type=str)
    parser.add_argument('--num_processes', type=int, default=10)
    args = parser.parse_args()

    dirs = glob.glob(args.input_dirs)
    with Pool(args.num_processes) as pool:
        pool.map(run, dirs)

    t1 = time.time()
    print("Execution took: {}".format(t1-t0))
