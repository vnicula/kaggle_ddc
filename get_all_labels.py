import glob
import json
from tqdm import tqdm

ALL_JSON = sorted(glob.glob("F:/deepfake-data/dfdc_train_part_*/metadata.json"))
# ALL_JSON = sorted(glob.glob("test_videos/metadata.json"))
print(len(ALL_JSON))

if __name__ == '__main__':

    with open(ALL_JSON[0]) as js:
        all_json = json.load(js)

    ALL_JSON = ALL_JSON[1:]
    for file in tqdm(ALL_JSON):
        with open(file) as js:
            all_json.update(json.load(js))

    with open("all_metadata.json", "w") as fo:
        json.dump(all_json, fo)