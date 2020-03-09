import argparse
import constants
import feature_extractor_models as featx
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import tf_explain
import tensorflow_addons as tfa
import time

from efficientnet.tfkeras import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from keras_vggface.vggface import VGGFace

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Activation, Bidirectional, BatchNormalization, Concatenate, Conv2D, Dense, Dropout
from tensorflow.keras.layers import Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Input, GRU, LeakyReLU, LSTM, Masking, MaxPooling2D, multiply, Reshape, TimeDistributed

from keras_utils import binary_focal_loss, save_loss, LRFinder, SeqWeightedAttention, balance_dataset, sce_loss, gce_loss

CMDLINE_ARGUMENTS = None

tfkl = tf.keras.layers

tf.random.set_seed(1234)
IMAGES_LOG_DIR = "logs/images"

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


def preprocess_img(img):
    if CMDLINE_ARGUMENTS.model_name == 'efficientnet':
        img = tf.cast(img, tf.float32)
        # img = tf.keras.applications.efficientnet.preprocess_input(img)
    elif CMDLINE_ARGUMENTS.model_name == 'xception':
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.xception.preprocess_input(img)
    elif CMDLINE_ARGUMENTS.model_name == 'mobilenet':
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    elif CMDLINE_ARGUMENTS.model_name == 'facenet' or CMDLINE_ARGUMENTS.model_name == 'vggface' or CMDLINE_ARGUMENTS.model_name == 'resface':
        img = tf.cast(img, tf.float32)
        img = tf.image.per_image_standardization(img)
    else:    
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)

    return img


def random_jitter(image):

    image = tf.image.resize(image, [272, 272]) # method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.image.random_crop(
        image, size=[constants.MESO_INPUT_HEIGHT, constants.MESO_INPUT_WIDTH, 3])

    return image


def random_rotate(image):
    if image.shape.__len__() == 4:
        random_angles = tf.random.uniform(shape = (tf.shape(image)[0], ), minval = -np.pi / 8, maxval = np.pi / 8)
    if image.shape.__len__() == 3:
        random_angles = tf.random.uniform(shape = (), minval = -np.pi / 8, maxval = np.pi / 8)

    # BUG in Tfa ABI undefined symbol: _ZNK10tensorflow15shape_inference16InferenceContext11DebugStringEv
    return tfa.image.rotate(image, random_angles)

@tf.function
def image_augment(x: tf.Tensor, y: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    """augmentation
    Args:
        x: Image
    Returns:
        Augmented image
    """
    # TODO investigate these two
    # x = tf.image.random_hue(x, 0.08)
    # x = tf.image.random_saturation(x, 0.6, 1.6)
    
    x = tf.image.random_brightness(x, 0.1)
    x = tf.image.random_contrast(x, 0.8, 1.2)
    x = tf.image.random_flip_left_right(x)

    jitter_choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    x = tf.cond(jitter_choice < 0.75, lambda: x, lambda: random_jitter(x))

    rotate_choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    x = tf.cond(rotate_choice < 0.75, lambda: x, lambda: random_rotate(x))
    x = tf.reshape(x, [constants.MESO_INPUT_HEIGHT, constants.MESO_INPUT_WIDTH, 3])

    jpeg_choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    x = tf.cond(jpeg_choice < 0.75, lambda: x, lambda: tf.image.random_jpeg_quality(
        x, min_jpeg_quality=40, max_jpeg_quality=90))

    return (x, y)


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=3)
    img = preprocess_img(img)

    return img, label


def split_pair(img, label):
    img_shape = tf.shape(img)
    # assert img_shape[0] == img_shape[1] and img_shape[1] % 2 == 0
    w_offset = img_shape[1] // 2

    left_input = tf.image.crop_to_bounding_box(
        img,
        offset_height=0,
        offset_width=0,
        target_height=img_shape[0],
        target_width=w_offset
    )
    right_input = tf.image.crop_to_bounding_box(
        img,
        offset_height=0,
        offset_width=w_offset,
        target_height=img_shape[0],
        target_width=w_offset
    )

    return tf.data.Dataset.from_tensor_slices(([left_input, right_input], [label, label]))


def prepare_dataset(ds, is_training, batch_size, cache):

    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            print('Caching dataset is_training: %s' % is_training)
            ds = ds.cache()

    if is_training:
        # ds = ds.map(AUGMENTATION, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.map(image_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    ds = ds.batch(batch_size)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds


def build_fnmatch(input_dir, pair):
    subdir = '256' if not pair else '256_pair'
    input_dir = os.path.join(input_dir, subdir)
    input_dir += '/?/*.png'

    return input_dir


def input_dataset(input_dir, is_training, batch_size, cache, pair):
    list_ds = tf.data.Dataset.list_files(build_fnmatch(input_dir, pair))
    if is_training:
        list_ds = list_ds.shuffle(buffer_size=50000)

    labeled_ds = list_ds.map(
        process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if pair:
        labeled_ds = labeled_ds.flat_map(split_pair)
    else:
        labeled_ds = balance_dataset(labeled_ds, is_training)    
    labeled_ds = prepare_dataset(labeled_ds, is_training, batch_size, cache)

    return labeled_ds


def fraction_positives(y_true, y_pred):
    return tf.keras.backend.mean(y_true)


def compile_model(model, mode, lr):

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
        tf.keras.metrics.BinaryCrossentropy(),
    ]
    if mode == 'train':
        METRICS.append(fraction_positives)
    
    optimizer = tf.keras.optimizers.SGD(lr)
    if mode == 'train':
        optimizer = tfa.optimizers.Lookahead(tf.keras.optimizers.SGD(lr, momentum=0.9))

    my_loss = tf.keras.losses.BinaryCrossentropy(
        label_smoothing=0.025
    )
    
    print('Using loss: %s, optimizer: %s' % (my_loss, optimizer))
    model.compile(loss=my_loss, optimizer=optimizer, metrics=METRICS)

    return model


def create_facenet_model(input_shape, mode):

    input_tensor = Input(shape=input_shape)
    base_model = tf.keras.models.load_model('pretrained/facenet_keras.h5')
    for i, layer in enumerate(base_model.layers):
        if layer.name == 'AvgPool':
            output = layer.output
            print('output set to {}.'.format(layer.name))
            break
    backbone_model = Model(inputs=base_model.input, outputs=output)
    
    net = backbone_model(input_tensor)
    net = Dropout(0.25)(net)
    net = Dense(1, activation='sigmoid',
                kernel_regularizer=tf.keras.regularizers.l2(0.02))(net)
    model = Model(inputs=input_tensor, outputs=net)

    return model, backbone_model, [423, 407, 391, 375, 0]


def create_vggface_model(input_shape, mode):

    vggface_weights = None
    if 'train' in CMDLINE_ARGUMENTS.mode and CMDLINE_ARGUMENTS.load is None:
        vggface_weights = 'vggface'
    print('Loading vggface weights from: ', vggface_weights)
    backbone_model = VGGFace(model='vgg16', weights=vggface_weights, input_shape=input_shape,
        include_top=False, pooling='avg')
    
    net = Flatten()(backbone_model.output)
    net = Dropout(0.25)(net)
    net = Dense(1, activation='sigmoid',
                kernel_regularizer=tf.keras.regularizers.l2(0.02))(net)
    model = Model(inputs=backbone_model.input, outputs=net)

    return model, backbone_model, [19, 15, 11, 0]


def create_model(model_name, input_shape, mode):
    # if model_name == 'mobilenet':
    #     return create_mobilenet_model(input_shape, mode)
    # if model_name == 'meso':
    #     return create_meso_model(input_shape, mode)
    # if model_name == 'meso5':
    #     return create_meso5_model(input_shape, mode)
    # if model_name == 'onemil':
    #     return create_onemil_model(input_shape, mode)
    # if model_name == 'xception':
    #     return create_xception_model(input_shape, mode)
    # if model_name == 'resnet':
    #     return create_resnet_model(input_shape, mode)
    # if model_name == 'efficientnet':
    #     return create_efficientnet_model(input_shape, mode)
    if model_name == 'facenet':
        return create_facenet_model(input_shape, mode)
    if model_name == 'vggface':
        return create_vggface_model(input_shape, mode)
    # if model_name == 'resface':
    #     return create_resface_model(input_shape, mode)

    raise ValueError('Unknown model %s' % model_name)


def freeze_first_n(base_model, N):

    print('\nFreezing first %d %s layers!' %(N, CMDLINE_ARGUMENTS.model_name))
    for i, layer in enumerate(base_model.layers):
        if i < N:
            layer.trainable = False
        else:
            layer.trainable = True
        print(i, layer.name, layer.trainable)


def callbacks_list(layer_index):

    model_name = CMDLINE_ARGUMENTS.model_name
    output_dir = CMDLINE_ARGUMENTS.output_dir

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, '%s_weights_li_%s_{epoch}.tf' % (model_name, layer_index)),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            save_format='tf',
            save_weights_only=True,
            verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', # watch out for reg losses
            min_delta=1e-3,
            patience=2,
            mode='min',
            restore_best_weights=True,
            verbose=1),
        tf.keras.callbacks.CSVLogger(
            os.path.join(output_dir, 'training_log_%s_li_%d.csv' % (model_name, layer_index))),
    ]

    return callbacks


def fit_with_schedule(model, backbone_model, layer_index, is_pair):

    train_dataset = input_dataset(
        CMDLINE_ARGUMENTS.train_dir, is_training=True, batch_size=CMDLINE_ARGUMENTS.batch_size, cache=False, pair=is_pair)
    cache_eval = False if CMDLINE_ARGUMENTS.mode == 'eval' else True
    eval_dataset = input_dataset(
        CMDLINE_ARGUMENTS.eval_dir, is_training=False, batch_size=CMDLINE_ARGUMENTS.batch_size, cache=cache_eval, pair=is_pair)

    val_loss = np.Inf
    best_weights = model.weights

    for li in layer_index:
        freeze_first_n(backbone_model, li)
        compile_model(model, CMDLINE_ARGUMENTS.mode, CMDLINE_ARGUMENTS.lr)
        print(model.summary())
        hfit = model.fit(train_dataset, epochs=CMDLINE_ARGUMENTS.epochs,  # class_weight=class_weight,
                                validation_data=eval_dataset,  # validation_steps=validation_steps,
                                callbacks=callbacks_list(li))
        phase_val_loss = min(hfit.history['val_loss'])
        completed_epochs = len(hfit.history['val_loss'])
        if phase_val_loss < val_loss:
            val_loss = phase_val_loss
            print('\nval_loss has improved to %f, backing up best weights.' % val_loss)
            best_weights = model.get_weights()
            if completed_epochs == CMDLINE_ARGUMENTS.epochs:
                print('\nWarning: Not early stopped, possibly not using the best weights.')
        else:
            print('Unfreezing all layers after %d did not improve val_loss, stopping training.' % li)
            break
    
    model.set_weights(best_weights)


def main():

    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str)
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--eval_dir', type=str)
    parser.add_argument('--output_dir', type=str, default='temp_tt')
    parser.add_argument('--model_name', type=str, default='unknown')
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--save', type=str, default='True')
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--frozen', type=int, default=-1)
    
    args = parser.parse_args()
    global CMDLINE_ARGUMENTS
    CMDLINE_ARGUMENTS = args

    num_epochs = args.epochs
    batch_size = int(args.batch_size)
    model_name = args.model_name
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    in_shape = (constants.MESO_INPUT_HEIGHT, constants.MESO_INPUT_WIDTH, 3)

    custom_objs = {
        'fraction_positives': fraction_positives,
        'Lookahead': tfa.optimizers.Lookahead,
        # 'SeqWeightedAttention': SeqWeightedAttention,
        # 'binary_focal_loss_fixed': binary_focal_loss(alpha=0.47),
        # 'sce_loss': sce_loss,
        # 'RectifiedAdam': tfa.optimizers.RectifiedAdam,
    }

    if args.mode == 'train' or args.mode == 'eval':
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        with strategy.scope():
            # due to bugs need to load weights for mirrored strategy - cannot load full model
            model, backbone_model, layer_index = create_model(model_name, in_shape, args.mode)
            if args.load is not None:
                print('\nLoading weights from: ', args.load)
                # model = tf.keras.models.load_model(args.load, custom_objects=custom_objs)
                model.load_weights(args.load)
            else:
                print('\nTraining model from scratch.')

            if args.mode == 'train':
                fit_with_schedule(model, backbone_model, layer_index, True)
                fit_with_schedule(model, backbone_model, layer_index, False)

    # print(next(iter(eval_dataset)))
    # fractions, counts = class_fractions(eval_dataset)
    # print('Eval dataset class counts {} and fractions {}: '.format(counts, fractions))


    elif args.mode == 'eval':
        eval_dataset = input_dataset(
            args.eval_dir, is_training=False, batch_size=batch_size,
            cache=False if args.mode == 'eval' else True,
            pair=False
        )
        model.evaluate(eval_dataset)

    if args.save == 'True':
        model_file_name = args.mode + '_dual_featx_full_%s_model.h5' % model_name
        model_weights_file_name = args.mode + '_dual_featx_full_%s_model_weights.h5' % model_name
        if args.load is not None:
            model_file_name = os.path.basename(args.load) + '_' + model_file_name
            model_weights_file_name = os.path.basename(args.load) + '_' + model_weights_file_name
        model.save_weights(os.path.join(output_dir, model_weights_file_name))
        model.save(os.path.join(output_dir, model_file_name))

        new_model = tf.keras.models.load_model(
            os.path.join(output_dir, model_file_name), custom_objects=custom_objs)
        new_model.summary()

    t1 = time.time()

    print("Execution took: {}".format(t1-t0))


if __name__ == '__main__':
    main()
