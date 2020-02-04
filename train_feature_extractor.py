import constants
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import time


def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return tf.strings.to_number(
        parts[-2],
        out_type=tf.int32,
        name=None
    )


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
#   return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
    return img


def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label


def input_dataset(input_dir, is_training):
    list_ds = tf.data.Dataset.list_files(input_dir)
    labeled_ds = list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(str(label_batch[n]))
      plt.axis('off')


if __name__ == '__main__':

    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--eval_dir', type=str)
    parser.add_argument('--load', type=str, default=None)
    args = parser.parse_args()

    train_dataset = input_dataset(args.train_dir, is_training=True)
    eval_dataset = input_dataset(args.eval_dir, is_training=False)
 
    in_shape = constants.FEAT_SHAPE

    custom_objs = {
        'fraction_positives':fraction_positives,
    }

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model = create_model(in_shape)
        if args.load is not None:
            print('Loading weights from: ', args.load)
            # model = tf.keras.models.load_model(args.load, custom_objects=custom_objs)
            model.load_weights(args.load)
        else:
            print('Training model from scratch.')
        compile_model(model)

    num_epochs = 1000
    # validation_steps = 32
    batch_size = 64

    # Cached for small datasets
    # train_dataset = train_dataset.shuffle(buffer_size=256).cache().batch(batch_size).prefetch(2)
    # eval_dataset = eval_dataset.take(validation_steps * (batch_size + 1)).cache().batch(batch_size).prefetch(1)
    train_dataset = train_dataset.batch(batch_size).prefetch(4)
    eval_dataset = eval_dataset.batch(batch_size).prefetch(4)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='featxw_{epoch}.h5',
            save_best_only=True,
            monitor='val_auc',
            # save_format='tf',
            save_weights_only=True,
            verbose=1),
        tf.keras.callbacks.EarlyStopping(
            # Stop training when `val_loss` is no longer improving
            # monitor='val_loss', # watch out for reg losses
            monitor='val_loss',
            min_delta=1e-4,
            patience=20,
            verbose=1),
        tf.keras.callbacks.CSVLogger('training_featx_log.csv'),
        # tf.keras.callbacks.LearningRateScheduler(step_decay),
        # CosineAnnealingScheduler(T_max=num_epochs, eta_max=0.02, eta_min=1e-5),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
            factor=0.95, patience=3, min_lr=5e-6, verbose=1, mode='min')
    ]
    
    # class_weight={0: 0.65, 1: 0.35}
    # class_weight=[0.99, 0.01]
    history = model.fit(train_dataset, epochs=num_epochs, #class_weight=class_weight, 
        validation_data=eval_dataset, #validation_steps=validation_steps, 
        callbacks=callbacks)
    
    model.save('final_featx_model.h5')
    # new_model = tf.keras.models.load_model('my_model')
    # new_model.summary()

    save_loss(history, 'final_featx_model')
    t1 = time.time()

    print("Execution took: {}".format(t1-t0))

