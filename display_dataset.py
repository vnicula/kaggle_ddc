import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

IMG_SIZE = 192

def tile_image(keys, values):
    
    line_shape = (IMG_SIZE, 16*IMG_SIZE, 3)
    tile_shape = (10*IMG_SIZE, 16*IMG_SIZE, 3)
    tile_img = np.zeros(tile_shape, dtype=np.uint8)
    for j, val in enumerate(values, 0):
        val_img = np.zeros(line_shape, dtype=np.uint8)
        for i, v in enumerate(val[1], 0):
            # v = cv2.resize(v, (160, 160))
            val_img[:, i*IMG_SIZE:(i+1)*IMG_SIZE, :] = v
        tile_img[j*IMG_SIZE:(j+1)*IMG_SIZE, :, :] = val_img
    
    return tile_img


if __name__ == '__main__':
    
    argp = argparse.ArgumentParser()
    argp.add_argument('--data', type=str)
    args = argp.parse_args()

    with open(args.data, 'rb') as f_p:
        data = pickle.load(f_p)
    keys = list(data.keys())

    plt.figure(figsize=(10,16))
    for start in np.arange(0, len(keys), 10):
        bkeys = keys[start:start+10]
        values = [data[key] for key in bkeys]
        tile_img = tile_image(bkeys, values)
        plt.imshow(tile_img)
        plt.show()
