#!/usr/bin/env python3
import glob
import os
import sys
import shutil
import tqdm

all_files = glob.glob(sys.argv[1])

for image_file in tqdm.tqdm(all_files):
    dir_name = os.path.dirname(image_file)
    # part_name = dir_name.split('/')[-3].split('_')[-1]
    part_name = dir_name.split('/')[-1]
    file_name = os.path.basename(image_file)
    newname = part_name + "_" + file_name
    # copy the file into the new directory
    newname = os.path.join(sys.argv[2], newname)
    # print('Copying %s to %s.' % (image_file, newname))
    shutil.copyfile(image_file, newname)