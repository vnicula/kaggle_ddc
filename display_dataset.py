import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

NUM_ROWS = 16

def tile_image(keys, values):
    max_elems = 0
    for val in values:
        if len(val[1]) > max_elems:
            max_elems = len(val[1])
    print(max_elems)
    IMG_SIZE = values[0][1][0].shape[0]    

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 4
    font_scale = 2

    line_shape = (IMG_SIZE, max_elems*IMG_SIZE, 3)
    tile_shape = (NUM_ROWS*IMG_SIZE, max_elems*IMG_SIZE, 3)
    tile_img = np.zeros(tile_shape, dtype=np.uint8)
    for j, val in enumerate(values, 0):
        val_img = np.zeros(line_shape, dtype=np.uint8)
        for i, v in enumerate(val[1], 0):
            # v = cv2.resize(v, (160, 160))
            val_img[:, i*IMG_SIZE:(i+1)*IMG_SIZE, :] = v
        color = (0, 255, 0) if val[0] == 0 else (255, 0, 0)
        cv2.putText(val_img, str(keys[j])+'=>'+str(val[0]), (10, 50),
                        font_face, font_scale,
                        color, thickness, 2)
        
        tile_img[j*IMG_SIZE:(j+1)*IMG_SIZE, :, :] = val_img
    
    return tile_img


if __name__ == '__main__':
    
    argp = argparse.ArgumentParser()
    argp.add_argument('--data', type=str)
    args = argp.parse_args()

    with open(args.data, 'rb') as f_p:
        data = pickle.load(f_p)
    keys = list(data.keys())

    for start in np.arange(0, len(keys), NUM_ROWS):
        bkeys = keys[start:start+NUM_ROWS]
        values = [data[key] for key in bkeys]
        tile_img = tile_image(bkeys, values)

        plt.figure(figsize=(20, 10))
        plt.imshow(tile_img)
        plt.show()
