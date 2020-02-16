import glob
import json
import os
import tqdm

if __name__ == '__main__':

    test_videos = glob.glob('/raid/scratch/tf_train/dset/test_videos/*.mp4')
    json_files = glob.glob('/raid/scratch/tf_train/dset/feat_val/dfdc_train_part_*/*.json')
    print(json_files)

    chunks_target = dict()
    chunks_original = dict()
    video_names = []
    for video_path in tqdm.tqdm(test_videos):
        video_name = os.path.basename(video_path)
        video_names.append(video_name)
        # print(video_name)
        for jfile in json_files:
            chunk_name = os.path.dirname(jfile).split('_')[-1]
            # print(chunk_name)
            with open(jfile) as jf:
                jl = json.load(jf)
                # print(jl)
            for jkey in jl.keys():
                # print(video_name,jkey)
                if video_name == jkey:
                    if chunk_name in chunks_target:
                        chunks_target[chunk_name].append(video_name)
                    else:
                        chunks_target[chunk_name] = [video_name]
                    # print('chunks_target[{}] = {}'.format(chunk_name, video_name))
                if 'original' in jl[jkey].keys() and video_name == jl[jkey]['original']:
                    if chunk_name in chunks_original:
                        chunks_original[chunk_name].append(video_name)
                    else:
                        chunks_original[chunk_name] = [video_name]
            
            
    print('Targets:\n', chunks_target.keys())
    print('Originals\n', chunks_original.keys())

    matched = set()
    for key in chunks_target:
        for name in chunks_target[key]:
            matched.add(name)
    for key in chunks_original:
        for name in chunks_original[key]:
            matched.add(name)

    print('Unmatched: ', set(video_names).difference(matched))