import argparse
import augment_image
import constants
import feature_extractor_models as featx
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import tf_explain
import tensorflow_addons as tfa
import time

from PIL import Image

import efficientnet.tfkeras
from efficientnet.tfkeras import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from keras_vggface.vggface import VGGFace
from keras_vggface import utils

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Activation, Bidirectional, BatchNormalization, Concatenate, Conv2D, Dense, Dropout
from tensorflow.keras.layers import Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Input, GRU, LeakyReLU, LSTM, Masking, MaxPooling2D, multiply, Reshape, TimeDistributed

from keras_utils import binary_focal_loss, save_loss, LRFinder, SeqWeightedAttention, balance_dataset, sce_loss, gce_loss
tfkl = tf.keras.layers

CMDLINE_ARGUMENTS = None

np.random.seed(1234)
tf.random.set_seed(1234)

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


def preprocess_img(img, label):
    # Note: at this point image should be 3D-4D tf.float32 0.-255.
    if CMDLINE_ARGUMENTS.model_name == 'efficientnet':
        img = efficientnet.tfkeras.preprocess_input(img)
    elif CMDLINE_ARGUMENTS.model_name == 'xception':
        img = tf.keras.applications.xception.preprocess_input(img)
    elif CMDLINE_ARGUMENTS.model_name == 'mobilenet':
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    elif CMDLINE_ARGUMENTS.model_name == 'vggface':
        # TODO fix these, use channel means
        # img = augment_image.preprocess_symbolic_input_vggface(img, version=1)
        img -= 127.5
    elif CMDLINE_ARGUMENTS.model_name == 'resface':
        # img -= 127.5
        # TODO fix these, channel means are different
        img = augment_image.preprocess_symbolic_input_vggface(img, version=2)
        # img = utils.preprocess_input(img, version=2)
        # img = tf.keras.applications.vgg19.preprocess_input(img)
        # img = tf.keras.applications.imagenet_utils.preprocess_input(img, mode='caffe')
    elif CMDLINE_ARGUMENTS.model_name == 'facenet':
        # TODO: incorrect as it doesn't use their dataset channel means
        img = tf.image.per_image_standardization(img)
    else:
        # [0., 1.)
        img = tf.image.convert_image_dtype(img, tf.float32)

    return img, label


@tf.function
def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.cast(img, tf.float32)
    # img = preprocess_img(img)

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
    # One of the most important 'ones' I ever wrote (was randomizing the labels)
    return tf.data.Dataset.from_tensor_slices(([left_input, right_input], [1-label, label]))


def prepare_dataset(ds, is_training, batch_size, cache):

    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            print('Caching dataset is_training: %s' % is_training)
            ds = ds.cache()

    # TODO move this after batching - deal with batch jitter and jpeg qual
    if is_training:
        # ds = ds.map(AUGMENTATION, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.map(augment_image.image_augment,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.map(preprocess_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
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
        # if CMDLINE_ARGUMENTS.model_name == 'facenet':
        #     # optimizer = tfa.optimizers.Lookahead(
        #     #     optimizer = tf.keras.optimizers.RMSprop(lr, decay=1e-5, momentum=0.9))
        #     optimizer = tf.keras.optimizers.RMSprop(lr, decay=1e-5, momentum=0.9)
        # else:
        # NOTE start simple for pretrained nets
        optimizer = tf.keras.optimizers.SGD(lr, momentum=0.9)

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
    net = Flatten()(net)
    net = Dropout(0.5)(net)
    net = Dense(1, activation='sigmoid',
                kernel_regularizer=tf.keras.regularizers.l2(0.02))(net)
    model = Model(inputs=input_tensor, outputs=net)

    return model, backbone_model, [423, 407, 391, 375, 359, 343, 327, 144, 20, 0]


def create_vggface_model(input_shape, mode):

    weights = None
    if 'train' in CMDLINE_ARGUMENTS.mode and CMDLINE_ARGUMENTS.load is None:
        weights = 'vggface'
    print('Loading vggface weights from: ', weights)
    backbone_model = VGGFace(model='vgg16', weights=weights, input_shape=input_shape,
                             include_top=False, pooling='avg')

    net = Flatten()(backbone_model.output)
    net = Dropout(0.25)(net)
    net = Dense(1, activation='sigmoid',
                kernel_regularizer=tf.keras.regularizers.l2(0.02))(net)
    model = Model(inputs=backbone_model.input, outputs=net)

    return model, backbone_model, [19, 17, 15, 11, 7, 4, 0]


def create_resface_model(input_shape, mode):

    weights = None
    if 'train' in CMDLINE_ARGUMENTS.mode and CMDLINE_ARGUMENTS.load is None:
        weights = 'vggface'
    print('Loading resface weights from: ', weights)
    backbone_model = VGGFace(model='resnet50', weights=weights, input_shape=input_shape,
                             include_top=False, pooling='avg')

    net = Flatten()(backbone_model.output)
    net = Dropout(0.25)(net)
    net = Dense(1, activation='sigmoid',
                kernel_regularizer=tf.keras.regularizers.l2(0.02))(net)
    model = Model(inputs=backbone_model.input, outputs=net)

    return model, backbone_model, [174, 141, 79, 37, 0]


def create_efficientnet_model(input_shape, mode):

    weights = None
    if 'train' in CMDLINE_ARGUMENTS.mode and CMDLINE_ARGUMENTS.load is None:
        weights = 'pretrained/efficientnet-b2_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
    print('Loading efficientnet weights from: ', weights)
    backbone_model = EfficientNetB2(weights=weights, input_shape=input_shape,
                                    include_top=False, pooling='avg')

    net = Flatten()(backbone_model.output)
    net = Dropout(0.25)(net)
    net = Dense(1, activation='sigmoid',
                kernel_regularizer=tf.keras.regularizers.l2(0.02))(net)
    model = Model(inputs=backbone_model.input, outputs=net)

    return model, backbone_model, [332, 329, 301, 228, 170, 112, 0]


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
    if model_name == 'efficientnet':
        return create_efficientnet_model(input_shape, mode)
    if model_name == 'facenet':
        return create_facenet_model(input_shape, mode)
    if model_name == 'vggface':
        return create_vggface_model(input_shape, mode)
    if model_name == 'resface':
        return create_resface_model(input_shape, mode)

    raise ValueError('Unknown model %s' % model_name)


def freeze_first_n(base_model, N):

    print('\nFreezing first %d %s layers!' % (N, CMDLINE_ARGUMENTS.model_name))
    for i, layer in enumerate(base_model.layers):
        if i < N:
            layer.trainable = False
        else:
            layer.trainable = True
        print(i, layer.name, layer.trainable)


def callbacks_list(layer_index, is_pair):

    model_name = CMDLINE_ARGUMENTS.model_name
    output_dir = CMDLINE_ARGUMENTS.output_dir
    phase = 'p' if is_pair else 'u'

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                output_dir, '%s_weights_li%d_ep{epoch}_%s.tf' % (model_name, layer_index, phase)),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            save_format='tf',
            save_weights_only=True,
            verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',  # watch out for reg losses
            min_delta=5e-4,
            patience=2,
            mode='min',
            restore_best_weights=True,
            verbose=1),
        tf.keras.callbacks.CSVLogger(
            os.path.join(output_dir, 'training_log_%s_li_%d_%s.csv' % (model_name, layer_index, phase))),
    ]

    return callbacks


def fit_with_schedule(model, backbone_model, layer_index, is_pair):
    assert CMDLINE_ARGUMENTS.mode == 'train', "Trying to call fit with mode %s" % CMDLINE_ARGUMENTS.mode
    train_dataset = input_dataset(
        CMDLINE_ARGUMENTS.train_dir, is_training=True, batch_size=CMDLINE_ARGUMENTS.batch_size, cache=False, pair=is_pair)
    eval_dataset = input_dataset(
        CMDLINE_ARGUMENTS.eval_dir, is_training=False, batch_size=CMDLINE_ARGUMENTS.batch_size, cache=True, pair=is_pair)

    print(backbone_model.summary())
    val_loss = np.Inf
    best_weights = model.get_weights()

    for i, li in enumerate(layer_index):
        freeze_first_n(backbone_model, li)
        lr = float(CMDLINE_ARGUMENTS.lr) * (0.9 ** i)
        compile_model(model, CMDLINE_ARGUMENTS.mode, lr)
        print('\nStep %d/%d with layer index %d, starting training with lr=%f\n' %(i+1, len(layer_index), li, lr))
        hfit = model.fit(train_dataset, epochs=CMDLINE_ARGUMENTS.epochs,  # class_weight=class_weight,
                         validation_data=eval_dataset,  # validation_steps=validation_steps,
                         callbacks=callbacks_list(li, is_pair))
        phase_val_loss = min(hfit.history['val_loss'])
        completed_epochs = len(hfit.history['val_loss'])
        if phase_val_loss < val_loss:
            val_loss = phase_val_loss
            print('\nOn li %d val_loss has improved to %f, backing up best weights.' % (li, val_loss))
            best_weights = model.get_weights()
            if completed_epochs == CMDLINE_ARGUMENTS.epochs:
                print('\nWarning: Not early stopped, possibly not using the best weights.')
        else:
            print('Unfreezing all layers after %d did not improve val_loss, loading best weights.' % li)
            model.set_weights(best_weights)

    model.set_weights(best_weights)
    
    return val_loss, best_weights


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
    parser.add_argument('--batch_size', type=int, default=256)
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
            model, backbone_model, layer_index = create_model(
                model_name, in_shape, args.mode)
            if args.load is not None:
                print('\nLoading weights from: ', args.load)
                # model = tf.keras.models.load_model(args.load, custom_objects=custom_objs)
                model.load_weights(args.load)
            else:
                print('\nTraining model from scratch.')

            if args.mode == 'train':
                val_loss_p = fit_with_schedule(model, backbone_model, layer_index, True)
                best_model_p_name = args.mode + '_tt_p_%s_model.h5' % model_name
                print('Saving best model on paired dset to %s' % best_model_p_name)
                model.save(os.path.join(output_dir, best_model_p_name))
                val_loss_u = fit_with_schedule(model, backbone_model, layer_index, False)
                print('\nTraining done, val_loss paired: %f, val_loss unpaired: %f\n' % (val_loss_p, val_loss_u))

    # print(next(iter(eval_dataset)))
    # fractions, counts = class_fractions(eval_dataset)
    # print('Eval dataset class counts {} and fractions {}: '.format(counts, fractions))

    if args.mode == 'eval':
        eval_dataset = input_dataset(
            args.eval_dir, is_training=False, batch_size=batch_size,
            cache=False,
            pair=False
        )
        compile_model(model, CMDLINE_ARGUMENTS.mode, CMDLINE_ARGUMENTS.lr)
        model.evaluate(eval_dataset)

    if args.save == 'True':
        model_file_name = args.mode + '_tt_full_%s_model.h5' % model_name
        model_weights_file_name = args.mode + \
            '_tt_full_%s_model_weights.h5' % model_name
        if args.load is not None:
            model_file_name = os.path.basename(
                args.load) + '_' + model_file_name
            model_weights_file_name = os.path.basename(
                args.load) + '_' + model_weights_file_name
        model.save_weights(os.path.join(output_dir, model_weights_file_name))
        model.save(os.path.join(output_dir, model_file_name))

        new_model = tf.keras.models.load_model(
            os.path.join(output_dir, model_file_name), custom_objects=custom_objs)
        new_model.summary()

    t1 = time.time()

    print("Execution took: {}".format(t1-t0))


if __name__ == '__main__':
    main()
