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


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=3)

    if CMDLINE_ARGUMENTS.model_name == 'efficientnet':
        img = tf.cast(img, tf.float32)
        # img = tf.keras.applications.efficientnet.preprocess_input(img)
    elif CMDLINE_ARGUMENTS.model_name == 'xception':
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.xception.preprocess_input(img)
    elif CMDLINE_ARGUMENTS.model_name == 'mobilenet':
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    elif CMDLINE_ARGUMENTS.model_name == 'facenet' or CMDLINE_ARGUMENTS.model_name == 'vggface':
        img = tf.cast(img, tf.float32)
        img = tf.image.per_image_standardization(img)
    else:    
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)

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

    # img = tf.image.pad_to_bounding_box(img, offset_height=0, offset_width=0,
    #     target_height=constants.MESO_INPUT_HEIGHT, target_width=constants.MESO_INPUT_WIDTH)
    # resize the image to the desired size.
    # img = tf.image.resize(img, [constants.MESO_INPUT_HEIGHT, constants.MESO_INPUT_WIDTH])
    
    return (left_input, right_input)


def random_jitter(image):

    image = tf.image.resize(image, [272, 272]) # method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.image.random_crop(
        image, size=[constants.MESO_INPUT_HEIGHT, constants.MESO_INPUT_WIDTH, 3])

    return image


def random_rotate(image):
    # if image.shape.__len__() == 4:
    #     random_angles = tf.random.uniform(shape = (tf.shape(image)[0], ), minval = -np.pi / 4, maxval = np.pi / 4)
    # if image.shape.__len__() == 3:
    #     random_angles = tf.random.uniform(shape = (), minval = -np.pi / 4, maxval = np.pi / 4)

    # # BUG in Tfa ABI undefined symbol: _ZNK10tensorflow15shape_inference16InferenceContext11DebugStringEv
    # return tfa.image.rotate(image, random_angles)

    # NOTE this needs numpy
    image_array = tf.keras.preprocessing.image.random_rotation(
        image.numpy(), 30, row_axis=0, col_axis=1, channel_axis=2, fill_mode='constant', cval=0.0,
        interpolation_order=1
    )
    # image = tf.convert_to_tensor(image_array)
    return image


@tf.function
def image_augment(x: (tf.Tensor, tf.Tensor), y: tf.Tensor) -> ((tf.Tensor, tf.Tensor), tf.Tensor):
    """augmentation
    Args:
        x: Image
    Returns:
        Augmented image
    """
    # TODO investigate these two
    # x = tf.image.random_hue(x, 0.08)
    # x = tf.image.random_saturation(x, 0.6, 1.6)
    
    x0 = tf.image.random_brightness(x[0], 0.1)
    x0 = tf.image.random_contrast(x0, 0.8, 1.2)
    x0 = tf.image.random_flip_left_right(x0)

    x1 = tf.image.random_brightness(x[1], 0.1)
    x1 = tf.image.random_contrast(x1, 0.8, 1.2)
    x1 = tf.image.random_flip_left_right(x1)

    rotate_choice0 = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    x0 = tf.cond(rotate_choice0 < 0.5, lambda: x0, lambda: tf.py_function(random_rotate, [x0], tf.float32))
    x0 = tf.reshape(x0, [constants.MESO_INPUT_HEIGHT, constants.MESO_INPUT_WIDTH, 3])
    
    rotate_choice1 = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    x1 = tf.cond(rotate_choice1 < 0.5, lambda: x1, lambda: tf.py_function(random_rotate, [x1], tf.float32))
    x1 = tf.reshape(x1, [constants.MESO_INPUT_HEIGHT, constants.MESO_INPUT_WIDTH, 3])

    jitter_choice0 = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    x0 = tf.cond(jitter_choice0 < 0.5, lambda: x0, lambda: random_jitter(x0))

    jitter_choice1 = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    x1 = tf.cond(jitter_choice1 < 0.5, lambda: x1, lambda: random_jitter(x1))

    # jpeg_choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    # x = tf.cond(jpeg_choice < 0.75, lambda: x, lambda: tf.image.random_jpeg_quality(
    #     x, min_jpeg_quality=80, max_jpeg_quality=100))

    return ((x0, x1), y)


@tf.function
def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)

    return img, label


def class_func(feat, label):
    return label


class TbAugmentation:
    def __init__(self, logdir: str, max_images: int, name: str):
        self.file_writer = tf.summary.create_file_writer(logdir)
        self.max_images: int = max_images
        self.name: str = name
        self._counter: int = 0

    def __call__(self, image, label):
        augmented_image, _ = image_augment(image, label)
        with self.file_writer.as_default():
            tf.summary.image(
                self.name + str(label),
                [tf.concat(augmented_image, axis=1)],
                step=self._counter,
                max_outputs=self.max_images,
            )

        self._counter += 1
        return augmented_image, label


def prepare_dataset(ds, is_training, batch_size, cache):
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.

    # AUGMENTATION = TbAugmentation(IMAGES_LOG_DIR, max_images=64, name="Images")

    # ds = balance_dataset(ds, is_training)

    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            print('Caching dataset is_training: %s' % is_training)
            ds = ds.cache()

    if is_training:
        # ds = ds.map(AUGMENTATION, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.map(image_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # else:
    #     # TODO: seems rejection_resample doesn't work with keras fit
    #     # resampler = tf.data.experimental.rejection_resample(
    #     #     class_func, target_dist=[0.5, 0.5]) #, initial_dist=[0.345, 0.655])
    #     # ds = ds.apply(resampler)
    #     # ds = ds.map(lambda extra_label, features_and_label: features_and_label)

    #     pass

    # Repeat forever
    # ds = ds.repeat()

    # Switch labels to one hot
    def switch_to_onehot(features, label):
        return features, tf.one_hot(label, depth=2)
    ds = ds.map(switch_to_onehot, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    ds = ds.batch(batch_size)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds


def input_dataset(input_dir, is_training, batch_size, cache):
    list_ds = tf.data.Dataset.list_files(input_dir)
    if is_training:
        list_ds = list_ds.shuffle(buffer_size=50000)

    labeled_ds = list_ds.map(
        process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    prepared_ds = prepare_dataset(labeled_ds, is_training, batch_size, cache)

    return prepared_ds


def tf_explain_dataset(input_dir):
    list_ds = tf.data.Dataset.list_files(input_dir)
    labeled_ds = list_ds.map(process_path, num_parallel_calls=4)
    positive_ds = labeled_ds.filter(lambda features, label: label == 1).take(8)
    features = []
    labels = []
    for feature, label in list(positive_ds.as_numpy_iterator()):
        features.append(feature)
        labels.append(label)

    return (np.array(features), np.array(labels))


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5, 5, n+1)
        plt.imshow(image_batch[n])
        plt.title(str(label_batch[n]))
        plt.axis('off')


def fraction_positives(y_true, y_pred):
    return tf.keras.backend.mean(y_true[:, 1])


def compile_model(model, mode, lr):

    if mode == 'train':
        # optimizer = tfa.optimizers.Lookahead(tfa.optimizers.RectifiedAdam(lr))
        optimizer = tfa.optimizers.RectifiedAdam(lr)
        # optimizer = tf.keras.optimizers.Adam(lr)  # (lr=0.025)
        # optimizer = tf.keras.optimizers.RMSprop(lr, decay=1e-5, momentum=0.9)
    elif mode == 'tune':
        # optimizer = tf.keras.optimizers.Adam()  # (lr=0.025)
        optimizer = tf.keras.optimizers.RMSprop(lr, decay=1e-6)
        # optimizer = tf.keras.optimizers.SGD(lr, momentum=0.9)
    else:
        optimizer = tf.keras.optimizers.SGD(lr)

    # learning_rate=CustomSchedule(D_MODEL)
    # optimizer = tf.keras.optimizers.Adam(
    #     learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    thresh = 0.5
    METRICS = [
        # tf.keras.metrics.TruePositives(name='tp', thresholds=thresh),
        # tf.keras.metrics.FalsePositives(name='fp', thresholds=thresh),
        # tf.keras.metrics.TrueNegatives(name='tn', thresholds=thresh),
        # tf.keras.metrics.FalseNegatives(name='fn', thresholds=thresh),
        # tf.keras.metrics.BinaryAccuracy(name='acc', threshold=thresh),
        # tf.keras.metrics.Precision(name='precision', thresholds=thresh),
        # tf.keras.metrics.Recall(name='recall', thresholds=thresh),
        tf.keras.metrics.AUC(name='auc'),
        # tf.keras.metrics.BinaryCrossentropy(from_logits=True),
        # tf.keras.metrics.BinaryCrossentropy(),
        # lr_metric,

        tf.keras.metrics.CategoricalAccuracy(name='acc'),
        tf.keras.losses.categorical_crossentropy,
    ]

    my_loss = tf.keras.losses.BinaryCrossentropy(
        label_smoothing=0.025
    )
    # my_loss = tf.keras.losses.CategoricalCrossentropy(
    #     label_smoothing=0.025
    # )
    if mode == 'train' or mode == 'tune':
        METRICS.append(fraction_positives)
        # my_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)
        # my_loss = binary_focal_loss(alpha=0.5)
        # my_loss = sce_loss,
        # my_loss = 'mean_squared_error'
        # my_loss = tf.keras.losses.MeanSquaredError()
    
    print('Using loss: %s, optimizer: %s' % (my_loss, optimizer))
    model.compile(loss=my_loss, optimizer=optimizer, metrics=METRICS)

    return model


def create_meso_model(input_shape, mode):

    classifier = featx.MesoInception4(input_shape)

    if mode == 'train':
        meso4_weights = 'pretrained/Meso/raw/all/weights.h5'
        classifier.model.load_weights(meso4_weights)
        print('\nUnfreezing all conv Meso layers!')
        # for layer in classifier.model.layers:
        #     if 'dense' not in layer.name:
        #         layer.trainable = False
    if mode == 'tune':
        print('\nUnfreezing all Meso layers!')

    for i, layer in enumerate(classifier.model.layers):
        print(i, layer.name, layer.trainable)

    return create_dual_model_with_backbone(input_shape, classifier.model)


def create_meso5_model(input_shape, mode):

    classifier = featx.MesoInception5(width=1, input_shape=input_shape)

    for i, layer in enumerate(classifier.model.layers):
        print(i, layer.name, layer.trainable)

    return create_dual_model_with_backbone(input_shape, classifier.model)


def create_onemil_model(input_shape, mode):

    classifier = featx.OneMIL(input_shape)

    for i, layer in enumerate(classifier.model.layers):
        print(i, layer.name, layer.trainable)

    return create_dual_model_with_backbone(input_shape, classifier.model)


def create_xception_model(input_shape, mode):

    xception_weights = 'pretrained/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'

    if mode == 'train':
        print('Loading xception weights from: ', xception_weights)
        base_model = Xception(weights=xception_weights,
                            input_shape=input_shape, include_top=False, pooling='avg')
        # print('\nFreezing all Xception layers!')
        # for layer in base_model.layers:
        #     layer.trainable = False
        print('\nUnfreezing last Xception layers!')
        for layer in base_model.layers[:129]:
            layer.trainable = False
        for layer in base_model.layers[129:]:
            layer.trainable = True
    elif mode == 'tune':
        base_model = Xception(weights=None,
                            input_tensor=input_tensor, include_top=False, pooling='avg')
        print('\nUnfreezing last k something Xception layers!')
        for layer in base_model.layers[:126]:
            layer.trainable = False
        for layer in base_model.layers[126:]:
            layer.trainable = True
    elif mode == 'eval':
        base_model = Xception(weights=None,
                            input_tensor=input_tensor, include_top=False, pooling='avg')

    net = base_model.output
    # net = Dropout(0.5)(net)
    # net = Dense(256, activation='relu')(net)
    net = Dropout(0.5)(net)
    out = Dense(1, activation=None,
                kernel_regularizer=tf.keras.regularizers.l2(0.02))(net)

    backbone_model = Model(inputs=base_model.input, outputs=out)
    for i, layer in enumerate(backbone_model.layers):
        print(i, layer.name, layer.trainable)


    return create_dual_model_with_backbone(input_shape, backbone_model)


def create_mobilenet_model(input_shape, mode):

    # create the base pre-trained model
    mobilenet_weights = 'pretrained/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224_no_top.h5'
    print('Loading mobilenet weights from: ', mobilenet_weights)
    base_model = MobileNetV2(weights=mobilenet_weights, alpha=0.5,
                             input_shape=input_shape, include_top=False, pooling='avg')

    if mode == 'train':
        # print('\nFreezing all Mobilenet layers!')
        # for layer in base_model.layers:
        #     layer.trainable = False
        print('\nUnfreezing last Mobilenet layers!')
        for layer in base_model.layers[:152]:
            layer.trainable = False
        for layer in base_model.layers[152:]:
            layer.trainable = True
        # for layer in base_model.layers[:135]:
        #     layer.trainable = False
        # for layer in base_model.layers[135:]:
        #     layer.trainable = True
    elif mode == 'tune':
        print('\nUnfreezing last k something mobilenet layers!')
        for layer in base_model.layers[:126]:
            layer.trainable = False
        for layer in base_model.layers[126:]:
            layer.trainable = True

    net = base_model.output
    # net = Dense(1024, activation='relu')(net)
    net = Dropout(0.5)(net)
    out = Dense(1, activation=None,
                kernel_regularizer=tf.keras.regularizers.l2(0.02))(net)

    backbone_model = Model(inputs=base_model.input, outputs=out)
    for i, layer in enumerate(backbone_model.layers):
        print(i, layer.name, layer.trainable)

    return create_dual_model_with_backbone(input_shape, backbone_model)


def create_efficientnet_model(input_shape, mode):

    # input_tensor = Input(shape=input_shape)
    # create the base pre-trained model
    efficientnet_weights = None
    if mode == 'train':
        efficientnet_weights = 'pretrained/efficientnet-b2_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
        # efficientnet_weights = 'imagenet'
    print('Loaded efficientnet weights from: ', efficientnet_weights)
    base_model = EfficientNetB2(weights=efficientnet_weights, input_shape=input_shape,
                                include_top=False, pooling='avg')

    if mode == 'train':
        print('\nFreezing all EfficientNet layers!')
        # for layer in base_model.layers:
        #     layer.trainable = False
        print('\nUnfreezing last efficient net layers!')
        # NOTE EffnetB1 and B2 both have top layer at 329
        for layer in base_model.layers[:329]:
            layer.trainable = False
        for layer in base_model.layers[329:]:
            layer.trainable = True
    elif mode == 'tune':
        print('\nUnfreezing last k something EfficientNet layers!')
        # Use 214 for EffNetB0
        for layer in base_model.layers[:214]:
            layer.trainable = False
        for layer in base_model.layers[214:]:
            layer.trainable = True
        # for layer in base_model.layers[:346]:
        #     layer.trainable = False
        # for layer in base_model.layers[346:]:
        #     layer.trainable = True
        # for layer in base_model.layers[:258]:
        #     layer.trainable = False
        # for layer in base_model.layers[258:]:
        #     layer.trainable = True

    net = base_model.output
    # net = Dense(1024, activation='relu')(net)
    net = Dropout(0.5)(net)
    out = Dense(1, activation=None,
                kernel_regularizer=tf.keras.regularizers.l2(0.02))(net)

    backbone_model = Model(inputs=base_model.input, outputs=out)
    for i, layer in enumerate(backbone_model.layers):
        print(i, layer.name, layer.trainable)

    return create_dual_model_with_backbone(input_shape, backbone_model)


def create_vggface_model(input_shape, mode):

    vggface_weights = None
    if 'train' in mode:
        # vggface_weights = 'pretrained/rcmalli_vggface_tf_notop_vgg16.h5'
        vggface_weights = 'vggface'
    print('Loaded vggface weights from: ', vggface_weights)
    base_model = VGGFace(weights=vggface_weights, input_shape=input_shape,
                                include_top=False, pooling='avg')

    if 'train' in mode:
        print('\nUnfreezing last vggface net layers!')
        for i, layer in enumerate(base_model.layers):
            if i < 11:
                layer.trainable = False
            else:
                layer.trainable = True
            print(i, layer.name, layer.trainable)
            # if layer.name == 'pool5':
            #     output = layer.output
            #     print('output set to {}.'.format(layer.name))

    # net = base_model.get_layer('pool5')
    net = base_model.output
    # net = Dense(1024, activation='relu')(net)
    net = Dropout(0.5)(net)
    out = Dense(1, activation=None,
                kernel_regularizer=tf.keras.regularizers.l2(0.02))(net)

    backbone_model = Model(inputs=base_model.input, outputs=out)
    for i, layer in enumerate(backbone_model.layers):
        print(i, layer.name, layer.trainable)

    return create_dual_model_with_backbone(input_shape, backbone_model)


def create_resnet_model(input_shape, mode):

    backbone_model = featx.resnet_18(input_shape, num_filters=4)

    return create_dual_model_with_backbone(input_shape, backbone_model)


def create_facenet_model(input_shape, mode):

    base_model = tf.keras.models.load_model('pretrained/facenet_keras.h5')
    if 'train' in mode:
        print('\nUnfreezing last facenet net layers!')
        # Note 423 too few weights, all block8 too many.
        for i, layer in enumerate(base_model.layers):
            if i < 420:
                layer.trainable = False
            else:
                layer.trainable = True
            # print(i, layer.name, layer.trainable)
            if layer.name == 'AvgPool':
                output = layer.output
                print('output set to {}.'.format(layer.name))

    net = output
    net = Dropout(0.5)(net)
    out = Dense(1, activation=None,
                kernel_regularizer=tf.keras.regularizers.l2(0.02))(net)

    backbone_model = Model(inputs=base_model.input, outputs=out)
    for i, layer in enumerate(backbone_model.layers):
        print(i, layer.name, layer.trainable)

    return create_dual_model_with_backbone(input_shape, backbone_model)


def create_dual_model_with_backbone(input_shape, backbone_model):
    input_left = tf.keras.layers.Input(shape=input_shape, name='input_left')
    input_right = tf.keras.layers.Input(shape=input_shape, name='input_right')

    net_left = backbone_model(input_left)
    net_right = backbone_model(input_right)

    net = tf.keras.layers.concatenate([net_left, net_right])
    # net = tf.keras.layers.Softmax()(net)
    net = Activation('sigmoid')(net)

    model = Model(inputs=[input_left, input_right], outputs=net)
    print(model.summary())

    return model, backbone_model


def create_model(model_name, input_shape, mode):
    if model_name == 'mobilenet':
        return create_mobilenet_model(input_shape, mode)
    if model_name == 'meso':
        return create_meso_model(input_shape, mode)
    if model_name == 'meso5':
        return create_meso5_model(input_shape, mode)
    if model_name == 'onemil':
        return create_onemil_model(input_shape, mode)
    if model_name == 'xception':
        return create_xception_model(input_shape, mode)
    if model_name == 'resnet':
        return create_resnet_model(input_shape, mode)
    if model_name == 'efficientnet':
        return create_efficientnet_model(input_shape, mode)
    if model_name == 'facenet':
        return create_facenet_model(input_shape, mode)
    if model_name == 'vggface':
        return create_vggface_model(input_shape, mode)

    raise ValueError('Unknown model %s' % model_name)


class SavingBackboneCallback(tf.keras.callbacks.Callback):

    def __init__(self, model, file_name):
        super(SavingBackboneCallback, self).__init__()
        self.backbone_model = model
        self.file_name, self.file_ext = os.path.splitext(file_name)
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get('val_loss')
        if np.less(current, self.best):
            self.best = current
            saved_file_name = self.file_name + '_%s' % str(epoch) + self.file_ext
            print('Saving backbone model weights to %s.' % saved_file_name)
            self.backbone_model.save_weights(saved_file_name)


def main():

    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str)
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--eval_dir', type=str)
    parser.add_argument('--output_dir', type=str, default='temp')
    parser.add_argument('--model_name', type=str, default='unknown')
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--save', type=str, default='True')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=512)
    
    args = parser.parse_args()
    global CMDLINE_ARGUMENTS
    CMDLINE_ARGUMENTS = args

    num_epochs = 1000
    # validation_steps = 32
    batch_size = int(args.batch_size)
    model_name = args.model_name
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    in_shape = (constants.MESO_INPUT_HEIGHT, constants.MESO_INPUT_WIDTH, 3)

    custom_objs = {
        'fraction_positives': fraction_positives,
        'SeqWeightedAttention': SeqWeightedAttention,
        # 'binary_focal_loss_fixed': binary_focal_loss(alpha=0.47),
        'sce_loss': sce_loss,
        'RectifiedAdam': tfa.optimizers.RectifiedAdam,
    }

    if args.mode == 'train' or args.mode == 'tune' or args.mode == 'eval':
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        with strategy.scope():
            # due to bugs need to load weights for mirrored strategy - cannot load full model
            model, backbone_model = create_model(model_name, in_shape, args.mode)
            if args.load is not None:
                print('\nLoading weights from: ', args.load)
                # model = tf.keras.models.load_model(args.load, custom_objects=custom_objs)
                model.load_weights(args.load)
            else:
                print('\nTraining model from scratch.')
            compile_model(model, args.mode, args.lr)
    elif args.mode == 'train_single':
        # due to bugs need to load weights for mirrored strategy - cannot load full model
        model, backbone_model = create_model(model_name, in_shape, args.mode)
        if args.load is not None:
            print('\nLoading weights from: ', args.load)
            # model = tf.keras.models.load_model(args.load, custom_objects=custom_objs)
            model.load_weights(args.load)
        else:
            print('\nTraining model from scratch.')
        compile_model(model, args.mode, args.lr)
    elif args.mode == 'explain':
        model, backbone_model = create_model(model_name, in_shape, args.mode)
        if args.load is not None:
            print('\nLoading weights from: ', args.load)
            # model = tf.keras.models.load_model(args.load, custom_objects=custom_objs)
            model.load_weights(args.load)
        else:
            raise ValueError('Explain mode needs --load argument.')
        compile_model(model, args.mode, args.lr)
        explain_dataset = tf_explain_dataset(args.eval_dir)
        print('Explain dataset shape %s %s %s' % (len(explain_dataset), explain_dataset[0].shape, explain_dataset[1].shape))

    eval_dataset = input_dataset(
        args.eval_dir, is_training=False, batch_size=batch_size,
        cache=False if args.mode == 'eval' else True
    )
    # print(next(iter(eval_dataset)))
    # fractions, counts = class_fractions(eval_dataset)
    # print('Eval dataset class counts {} and fractions {}: '.format(counts, fractions))

    if 'train' in args.mode or args.mode == 'tune':
        train_dataset = input_dataset(args.train_dir, is_training=True, batch_size=batch_size,
                                      cache=False,
                                      # cache='/raid/scratch/training_cache.mem'
                                      )
        # print(next(iter(train_dataset)))
        # lr_callback = LRFinder(num_samples=33501, batch_size=batch_size,
        #                minimum_lr=1e-5, maximum_lr=5e-1,
        #                # validation_data=(X_val, Y_val),
        #                lr_scale='exp', save_dir='.')

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(output_dir, 'dual_featx_weights_%s_{epoch}.h5' % (model_name + '_' + args.mode)),
                save_best_only=True,
                monitor='val_categorical_crossentropy',
                mode='min',
                # save_format='tf',
                save_weights_only=True,
                verbose=1),
            tf.keras.callbacks.EarlyStopping(
                # Stop training when `val_loss` is no longer improving
                # monitor='val_loss', # watch out for reg losses
                monitor='val_loss',
                min_delta=1e-4,
                patience=20,
                mode='min',
                verbose=1),
            tf.keras.callbacks.CSVLogger(
                os.path.join(output_dir, 'training_dual_featx_%s_log.csv' % model_name)),
            # tf.keras.callbacks.LearningRateScheduler(step_decay),
            # CosineAnnealingScheduler(T_max=num_epochs, eta_max=0.02, eta_min=1e-5),
            # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
            #                                      factor=0.96, patience=3, min_lr=5e-5, verbose=1, mode='min'),
            # lr_callback,
            # tf.keras.callbacks.TensorBoard(log_dir='./train_featx_%s_logs' % model_name),
            SavingBackboneCallback(backbone_model,
                os.path.join(output_dir, 'dual_featx_backbone_weights_%s.h5' % (model_name + '_' + args.mode)))
        ]
        if args.mode == 'explain':
            explain_callbacks=[
                tf_explain.callbacks.GradCAMCallback(
                    validation_data=explain_dataset, 
                    layer_name='conv2d_22', 
                    class_index=1,
                ),
                tf_explain.callbacks.OcclusionSensitivityCallback(
                    validation_data=explain_dataset,
                    class_index=1,
                    patch_size=4,
                ),
                tf_explain.callbacks.ActivationsVisualizationCallback(
                    validation_data=explain_dataset, 
                    layers_name=['conv2d_21']
                )
            ]
            callbacks.extend(explain_callbacks)

        # class_weight = {0: 0.45, 1: 0.55}
        history = model.fit(train_dataset, epochs=num_epochs,  # class_weight=class_weight,
                            validation_data=eval_dataset,  # validation_steps=validation_steps,
                            callbacks=callbacks)

        # lr_callback.plot_schedule()
        save_loss(history, os.path.join(output_dir, 'final_dual_featx_%s_model' % model_name))

    elif args.mode == 'eval':
        model.evaluate(eval_dataset)

    if args.save == 'True':
        model_file_name = args.mode + '_dual_featx_full_%s_model.h5' % model_name
        backbone_model_file_name = args.mode + '_backbone_dual_featx_%s_model.h5' % model_name
        backbone_model_weights_file_name = args.mode + '_backbone_dual_featx_%s_model_weights.h5' % model_name
        if args.load is not None:
            model_file_name = os.path.basename(args.load) + '_' + model_file_name
            backbone_model_file_name = os.path.basename(args.load) + '_' + backbone_model_file_name
            backbone_model_weights_file_name = os.path.basename(args.load) + '_' + backbone_model_weights_file_name
        model.save(os.path.join(output_dir, model_file_name))
        backbone_model.save_weights(os.path.join(output_dir, backbone_model_weights_file_name))
        backbone_model.save(os.path.join(output_dir, backbone_model_file_name))

        new_model = tf.keras.models.load_model(
            os.path.join(output_dir, model_file_name), custom_objects=custom_objs)
        new_model.summary()

    t1 = time.time()

    print("Execution took: {}".format(t1-t0))


if __name__ == '__main__':
    main()
