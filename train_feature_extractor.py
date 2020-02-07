import argparse
import constants
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import time

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Bidirectional, BatchNormalization, Concatenate, Conv2D, Dense, Dropout
from tensorflow.keras.layers import Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Input, GRU, LeakyReLU, LSTM, Masking, MaxPooling2D, multiply, Reshape, TimeDistributed

from keras_utils import binary_focal_loss, save_loss, LRFinder

tfkl = tf.keras.layers

INPUT_WIDTH = 256
INPUT_HEIGHT = 256

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


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
    
    # Xception
    # img = tf.cast(img, tf.float32)
    
    # resize the image to the desired size.
    img = tf.image.resize(img, [INPUT_WIDTH, INPUT_HEIGHT])
    return img


def augment(x: tf.Tensor) -> tf.Tensor:
    """augmentation
    Args:
        x: Image
    Returns:
        Augmented image
    """
    # x = tf.image.random_hue(x, 0.08)
    # x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    x = tf.image.random_flip_left_right(x)
    return x


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    # TODO make sure you take this out for non xception backbones
    # img = tf.keras.applications.xception.preprocess_input(img)
    # img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    # minimal augmentation
    img = augment(img)    

    return img, label


def prepare_dataset(ds, is_training, batch_size, cache, shuffle_buffer_size=60000):
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    if is_training:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    # ds = ds.repeat()

    ds = ds.batch(batch_size)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds


def input_dataset(input_dir, is_training, batch_size, cache):
    list_ds = tf.data.Dataset.list_files(input_dir)
    labeled_ds = list_ds.map(
        process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    prepared_ds = prepare_dataset(labeled_ds, is_training, batch_size, cache)

    return prepared_ds


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5, 5, n+1)
        plt.imshow(image_batch[n])
        plt.title(str(label_batch[n]))
        plt.axis('off')


class MesoInception4():
    def __init__(self, learning_rate=0.001):
        self.model = self.init_model()
        self.model.load_weights('pretrained/Meso/c23/all/weights.h5')
        # optimizer = Adam(lr = learning_rate)
        # self.model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])

    def InceptionLayer(self, a, b, c, d):
        def func(x):
            x1 = Conv2D(a, (1, 1), padding='same', activation='relu')(x)

            x2 = Conv2D(b, (1, 1), padding='same', activation='relu')(x)
            x2 = Conv2D(b, (3, 3), padding='same', activation='relu')(x2)

            x3 = Conv2D(c, (1, 1), padding='same', activation='relu')(x)
            x3 = Conv2D(c, (3, 3), dilation_rate=2, strides=1,
                        padding='same', activation='relu')(x3)

            x4 = Conv2D(d, (1, 1), padding='same', activation='relu')(x)
            x4 = Conv2D(d, (3, 3), dilation_rate=3, strides=1,
                        padding='same', activation='relu')(x4)

            y = Concatenate(axis=-1)([x1, x2, x3, x4])

            return y
        return func

    def init_model(self):
        x = Input(shape=(256, 256, 3))

        x1 = self.InceptionLayer(1, 4, 4, 2)(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

        x2 = self.InceptionLayer(2, 4, 4, 2)(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

        x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

        x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)

        return Model(inputs=x, outputs=y)


def fraction_positives(y_true, y_pred):
    return tf.keras.backend.mean(y_true)


def compile_model(model, mode, lr):

    if mode == 'train' or mode == 'eval':
        optimizer = tf.keras.optimizers.Adam(lr)  # (lr=0.025)
    elif mode == 'tune':
        # optimizer = tf.keras.optimizers.Adam()  # (lr=0.025)
        optimizer = tf.keras.optimizers.SGD(lr, momentum=0.9)

    # learning_rate=CustomSchedule(D_MODEL)
    # optimizer = tf.keras.optimizers.Adam(
    #     learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    thresh = 0.5
    METRICS = [
        tf.keras.metrics.TruePositives(name='tp', thresholds=thresh),
        tf.keras.metrics.FalsePositives(name='fp', thresholds=thresh),
        tf.keras.metrics.TrueNegatives(name='tn', thresholds=thresh),
        tf.keras.metrics.FalseNegatives(name='fn', thresholds=thresh),
        tf.keras.metrics.BinaryAccuracy(name='acc', threshold=thresh),
        tf.keras.metrics.Precision(name='precision', thresholds=thresh),
        tf.keras.metrics.Recall(name='recall', thresholds=thresh),
        tf.keras.metrics.AUC(name='auc'),
        # tf.keras.metrics.BinaryCrossentropy(from_logits=True),
        tf.keras.metrics.BinaryCrossentropy(),
        fraction_positives,
        # lr_metric,
    ]
    # my_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)
    my_loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.025)
    # my_loss = binary_focal_loss(alpha=0.7)
    # my_loss = 'mean_squared_error'
    model.compile(loss=my_loss, optimizer=optimizer, metrics=METRICS)

    return model


def create_meso_model(input_shape, mode):

    classifier = MesoInception4()

    if mode == 'train':
        print('\nFreezing all conv Meso layers!')
        for layer in classifier.model.layers:
            if 'dense' not in layer.name:
                layer.trainable = False
    if mode == 'tune':
        print('\nUnfreezing all Meso layers!')

    for i, layer in enumerate(classifier.model.layers):
        print(i, layer.name, layer.trainable)
    print(classifier.model.summary())

    return classifier.model


def create_xception_model(input_shape, mode):

    input_tensor = Input(shape=input_shape)
    # create the base pre-trained model
    xception_weights = 'pretrained/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
    print('Loading xception weights from: ', xception_weights)
    base_model = Xception(weights=xception_weights, 
        input_tensor=input_tensor, include_top=False, pooling='avg')

    if mode == 'train':
        # print('\nFreezing all Xception layers!')
        # for layer in base_model.layers:
        #     layer.trainable = False
        print('\nUnfreezing last Xception layers!')
        for layer in base_model.layers[:129]:
            layer.trainable = False
        for layer in base_model.layers[129:]:
            layer.trainable = True
    elif mode == 'tune':
        print('\nUnfreezing last k something Xception layers!')
        for layer in base_model.layers[:126]:
            layer.trainable = False
        for layer in base_model.layers[126:]:
            layer.trainable = True

    net = base_model.output
    # net = Dense(1024, activation='relu')(net)
    net = Dropout(0.5)(net)
    out = Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.02))(net)

    model = Model(inputs=base_model.input, outputs=out)
    print(model.summary())

    return model


def create_mobilenet_model(input_shape, mode):

    input_tensor = Input(shape=input_shape)
    # create the base pre-trained model
    mobilenet_weights = 'pretrained/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5'
    print('Loading mobilenet weights from: ', mobilenet_weights)
    base_model = MobileNetV2(weights=mobilenet_weights, alpha = 1.0,
        input_tensor=input_tensor, include_top=False, pooling='avg')

    if mode == 'train':
        # print('\nFreezing all Mobilenet layers!')
        # for layer in base_model.layers:
        #     layer.trainable = False
        print('\nUnfreezing last Mobilenet layers!')
        for layer in base_model.layers[:152]:
            layer.trainable = False
        for layer in base_model.layers[152:]:
            layer.trainable = True
    elif mode == 'tune':
        print('\nUnfreezing last k something mobilenet layers!')
        for layer in base_model.layers[:126]:
            layer.trainable = False
        for layer in base_model.layers[126:]:
            layer.trainable = True

    net = base_model.output
    # net = Dense(1024, activation='relu')(net)
    net = Dropout(0.5)(net)
    out = Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.02))(net)

    model = Model(inputs=base_model.input, outputs=out)
    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.trainable)
    print(model.summary())

    return model


if __name__ == '__main__':

    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--eval_dir', type=str)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--save', type=str, default='true')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=512)
    args = parser.parse_args()

    num_epochs = 1000
    # validation_steps = 32
    batch_size = int(args.batch_size)
    in_shape = constants.FEAT_SHAPE

    custom_objs = {
        'fraction_positives': fraction_positives,
    }

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        # due to bugs need to load weights for mirrored strategy - cannot load full model
        # model = create_mobilenet_model(in_shape, args.mode)
        model = create_meso_model(in_shape, args.mode)
        if args.load is not None:
            print('\nLoading weights from: ', args.load)
            # model = tf.keras.models.load_model(args.load, custom_objects=custom_objs)
            model.load_weights(args.load)
        else:
            print('\nTraining model from scratch.')
        compile_model(model, args.mode, args.lr)

    if args.mode == 'train' or args.mode == 'tune':
        train_dataset = input_dataset(args.train_dir, is_training=True, batch_size=batch_size,
            cache=False
            # cache='/raid/scratch/training_cache.mem'
        )
        eval_dataset = input_dataset(args.eval_dir, is_training=False, batch_size=batch_size, 
            cache=True
        )

        # lr_callback = LRFinder(num_samples=33501, batch_size=batch_size,
        #                minimum_lr=1e-5, maximum_lr=5e-1,
        #                # validation_data=(X_val, Y_val),
        #                lr_scale='exp', save_dir='.')

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath='featx_weights_{epoch}.h5',
                save_best_only=True,
                monitor='val_binary_crossentropy',
                # save_format='tf',
                save_weights_only=True,
                verbose=1),
            tf.keras.callbacks.EarlyStopping(
                # Stop training when `val_loss` is no longer improving
                # monitor='val_loss', # watch out for reg losses
                monitor='val_binary_crossentropy',
                min_delta=1e-4,
                patience=40,
                verbose=1),
            tf.keras.callbacks.CSVLogger('training_featx_log.csv'),
            # tf.keras.callbacks.LearningRateScheduler(step_decay),
            # CosineAnnealingScheduler(T_max=num_epochs, eta_max=0.02, eta_min=1e-5),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.96, patience=4, min_lr=5e-4, verbose=1, mode='min'),
            # lr_callback,
        ]

        class_weight={0: 0.55, 1: 0.45}
        # class_weight=[0.99, 0.01]
        history = model.fit(train_dataset, epochs=num_epochs, class_weight=class_weight,
                            validation_data=eval_dataset,  # validation_steps=validation_steps,
                            callbacks=callbacks)
        
        # lr_callback.plot_schedule()
        save_loss(history, 'final_featx_model')


    elif args.mode == 'eval':
        eval_dataset = input_dataset(args.eval_dir, is_training=False, batch_size=batch_size, cache=True)
        model.evaluate(eval_dataset)

    if args.save is not None:
        model.save(args.load + '_' + args.mode + '_full_model.h5')
        # new_model = tf.keras.models.load_model('my_model')
        # new_model.summary()

    t1 = time.time()

    print("Execution took: {}".format(t1-t0))
