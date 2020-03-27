import argparse
import constants
import cv2
import numpy as np
import pandas as pd
import glob
import os
import time
import tqdm


def create_path_dictionary(input_dir, videoDF):
    npz_files = os.path.join(input_dir, 'youtube_faces_*/*.npz')
    npzFilesFullPath = glob.glob(npz_files)
    videoIDs = [x.split('/')[-1].split('.')[0] for x in npzFilesFullPath]
    fullPaths = {}
    for videoID, fullPath in zip(videoIDs, npzFilesFullPath):
        fullPaths[videoID] = fullPath

    # remove from the large csv file all videos that weren't uploaded yet
    videoDF = videoDF.loc[videoDF.loc[:,'videoID'].isin(fullPaths.keys()),:].reset_index(drop=True)
    print('Number of Videos is %d' %(videoDF.shape[0]))
    print('Number of Unique Individuals is %d' %(len(videoDF['personName'].unique())))

    return fullPaths


if __name__ == '__main__':

    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--min_face_size', type=int, default=144)
    parser.add_argument('--face_resize', type=int, default=256)
    args = parser.parse_args()
    face_resize = args.face_resize

    videoDF = pd.read_csv(os.path.join(
        args.input_dir, 'youtube_faces_with_keypoints_large.csv'))
    # print(videoDF.head(10))
    fullPaths = create_path_dictionary(args.input_dir, videoDF)

    framesToShowFromVideo = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    for videoID in tqdm.tqdm(videoDF.videoID):

        videoFile = np.load(fullPaths[videoID])
        colorImages = videoFile['colorImages']
        boundingBox = videoFile['boundingBox']
        # print(boundingBox.shape, boundingBox)

        # select frames and save their content
        selectedFrames = (framesToShowFromVideo*(colorImages.shape[3]-1)).astype(int)
        for frameInd in selectedFrames:
            bbox = boundingBox[:, :, frameInd]
            hbbox_lu = max(int(np.min(bbox[:, 0])) - constants.MARGIN // 2, 0)
            wbbox_lu = max(int(np.min(bbox[:, 1])) - constants.MARGIN // 2, 0)
            hbbox_rd = int(np.max(bbox[:, 0])) + constants.MARGIN
            wbbox_rd = int(np.max(bbox[:, 1])) + constants.MARGIN
            img_file = os.path.join(args.output_dir, '%s_%d.png' % (videoID, frameInd))
            img_crop = colorImages[wbbox_lu:wbbox_rd, hbbox_lu:hbbox_rd, :,frameInd]
            if img_crop.shape[0] >= args.min_face_size or img_crop.shape[1] >= args.min_face_size:
                image0 = cv2.resize(img_crop, (face_resize, face_resize), interpolation = cv2.INTER_LINEAR)
                cv2.imwrite(img_file, cv2.cvtColor(image0, cv2.COLOR_RGB2BGR))

    t1 = time.time()
    print("Execution took: {}".format(t1-t0))
