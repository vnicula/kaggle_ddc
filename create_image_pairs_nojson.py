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

def run(fake_dir, real_dir):

    reading_dir_0 = os.path.join(real_dir, str(face_size), '0')
    reading_dir_1 = os.path.join(fake_dir, str(face_size), '1')
    writing_dir_0 = os.path.join(fake_dir, str(face_size) + '_pair', '0')
    os.makedirs(writing_dir_0, exist_ok=True)
    writing_dir_1 = os.path.join(fake_dir, str(face_size) + '_pair', '1')
    os.makedirs(writing_dir_1, exist_ok=True)
    real_file_list = glob.glob(reading_dir_0 + '/*.png')
    fake_file_list = glob.glob(reading_dir_1 + '/*.png')

    for path_name in tqdm.tqdm(fake_file_list):
        file_name = os.path.basename(path_name)
        # fake_name = file_name.split('.')[0]
        source_no = file_name.split('_')[0]
        target_no = file_name.split('_')[1].split('.')[0]
        video_no = file_name.split('.')[1]
        real_path_source = os.path.join(reading_dir_0, source_no + '.' + video_no + '.png')
        real_path_target = os.path.join(reading_dir_0, target_no + '.' + video_no + '.png')
        
        # print(real_path_source, real_path_target)
        # assert os.path.exists(real_path_source) and os.path.exists(real_path_target)

        fake_img = imageio.imread(path_name)
        if os.path.exists(real_path_source):
            real_img_source = imageio.imread(real_path_source)

            label1 = np.random.randint(2)

            if label1 == 0:
                sample_name = source_no + '_' + target_no + '_' + video_no + '.' + source_no + '.png'
                sample_file = os.path.join(writing_dir_0, sample_name)
                imageio.imwrite(sample_file, np.concatenate([fake_img, real_img_source], axis=1))
            else:
                sample_name = source_no + '.' + source_no + '_' + target_no + '_' + video_no + '.png'
                sample_file = os.path.join(writing_dir_1, sample_name)
                imageio.imwrite(sample_file, np.concatenate([real_img_source, fake_img], axis=1))

        if os.path.exists(real_path_target):

            real_img_target = imageio.imread(real_path_target)
            label2 = np.random.randint(2)

            if label2 == 0:
                sample_name = source_no + '_' + target_no + '_' + video_no + '.' + target_no + '.png'
                sample_file = os.path.join(writing_dir_0, sample_name)
                imageio.imwrite(sample_file, np.concatenate([fake_img, real_img_target], axis=1))
            else:
                sample_name = target_no + '.' + source_no + '_' + target_no + '_' + video_no + '.png'
                sample_file = os.path.join(writing_dir_1, sample_name)
                imageio.imwrite(sample_file, np.concatenate([real_img_target, fake_img], axis=1))

        if not os.path.exists(real_path_source) and not os.path.exists(real_path_target):
            print('Could not find a match for %s, both %s and %s do not exist' %(path_name, real_path_source, real_path_target))


if __name__ == '__main__':

    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_dir', type=str)
    parser.add_argument('--real_dir', type=str)
    args = parser.parse_args()

    run(args.fake_dir, args.real_dir)

    t1 = time.time()
    print("Execution took: {}".format(t1-t0))
