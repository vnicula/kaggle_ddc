import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import tensorflow as tf

NUM_ROWS = 16
SEQ_LEN = 30
FEAT_SHAPE = (224, 224, 3)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)


def tile_image(keys, values, masks=None):
    """
    value it's a tuple of (label, sample)
    """
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
            if masks is not None:
                mask_color = (255, 255, 0) if masks[j][i] > 0 else (0, 255, 255)
                cv2.putText(val_img, str(masks[j][i]), (i*IMG_SIZE+180, 60),
                            font_face, font_scale,
                            mask_color, thickness, 2)
        color = (0, 255, 0) if val[0] == 0 else (255, 0, 0)
        cv2.putText(val_img, str(keys[j])+'=>'+str(val[0]), (10, 50),
                        font_face, font_scale,
                        color, thickness, 2)
        
        tile_img[j*IMG_SIZE:(j+1)*IMG_SIZE, :, :] = val_img
    
    return tile_img

def display_pkl(pkl_file):
    with open(pkl_file, 'rb') as f_p:
        data = pickle.load(f_p)
    keys = list(data.keys())

    for start in np.arange(0, len(keys), NUM_ROWS):
        bkeys = keys[start:start+NUM_ROWS]
        values = [data[key] for key in bkeys]
        tile_img = tile_image(bkeys, values)

        plt.figure(figsize=(20, 10))
        plt.imshow(tile_img)
        plt.show()


def display_tfrecord(rec_file):

    # tf.enable_v2_behavior()
    feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'name': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'sample': tf.io.FixedLenFeature([30, 224, 224, 3], tf.float32),
        'mask': tf.io.FixedLenFeature([30], tf.float32),
    }

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, feature_description)

    dataset = tf.data.TFRecordDataset(rec_file)
    # for raw_record in dataset.take(1):
    #     example = tf.train.Example()
    #     example.ParseFromString(raw_record.numpy())
    #     print(example)
        
    parsed_dataset = dataset.map(_parse_function)
    
    samples = []
    keys = []
    labels = []
    masks = []

    for features in parsed_dataset:

        samples.append((features['sample'].numpy() + 1.0) * 127.5)
        keys.append(features['name'].numpy())
        labels.append(features['label'].numpy())
        masks.append(features['mask'].numpy())

    for start in np.arange(0, len(keys), NUM_ROWS):
        bkeys = keys[start:start+NUM_ROWS]
        values = tuple(zip(labels[start:start+NUM_ROWS], samples[start:start+NUM_ROWS]))
        tile_img = tile_image(bkeys, values, masks[start:start+NUM_ROWS])

        plt.figure(figsize=(20, 10))
        plt.imshow(tile_img)
        plt.savefig(os.path.basename(rec_file)+'_%d.png'%(start//NUM_ROWS))
        plt.close()


if __name__ == '__main__':
    
    argp = argparse.ArgumentParser()
    argp.add_argument('--data', type=str)
    args = argp.parse_args()

    filename, file_extension = os.path.splitext(args.data)
    if "pkl" in file_extension:
        display_pkl(args.data)
    elif "tfrec" in file_extension:
        display_tfrecord(args.data)
    else:
        print("Only pkl and tfrecord.")


