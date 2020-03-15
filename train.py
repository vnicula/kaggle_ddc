from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import augment_image
import constants
import feature_extractor_models as featx
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import tensorflow as tf
import tensorflow_addons as tfa
import time
import threading
import tqdm

from scipy.interpolate import griddata
from sklearn.metrics import log_loss

import efficientnet.tfkeras
from efficientnet.tfkeras import EfficientNetB0, EfficientNetB1, EfficientNetB2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.xception import Xception

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Bidirectional, BatchNormalization, Concatenate, Conv2D, Dense, Dropout
from tensorflow.keras.layers import Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Input, GRU, LeakyReLU, LSTM, Masking, MaxPooling2D, multiply, Reshape, TimeDistributed
# from tensorflow.keras.utils import multi_gpu_model

from keras_utils import ScaledDotProductAttention, SeqSelfAttention, SeqWeightedAttention, binary_focal_loss, save_loss
from keras_utils import balance_dataset
# from multi_head import Encoder, CustomSchedule

# Needed because keras model.fit shape checks are weak
# https://github.com/tensorflow/tensorflow/issues/24520
# tf.enable_eager_execution()

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


D_MODEL = 784


@tf.function
def image_augment(x: tf.Tensor, y: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    """augmentation
    Args:
        x: Image
    Returns:
        Augmented image
    """

    img = x['input_1']
    
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_flip_left_right(img)

    jitter_choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    img = tf.cond(jitter_choice < 0.75, lambda: img, lambda: augment_image.random_jitter(img))

    rotate_choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    img = tf.cond(rotate_choice < 0.75, lambda: img, lambda: augment_image.random_rotate(img))

    return {'input_1':img, 'input_2':x['input_2']}, y


def tfrecords_dataset(input_dir, is_training):
    print('Using tfrecords dataset from: ', input_dir)
    
    file_list = tf.data.Dataset.list_files(input_dir)
    if is_training:
        file_list = file_list.shuffle(buffer_size=512)

    dataset = tf.data.TFRecordDataset(filenames=file_list, 
        buffer_size=None, 
        num_parallel_reads=tf.data.experimental.AUTOTUNE
    )

    feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'name': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'sample': tf.io.FixedLenFeature((constants.SEQ_LEN,) + constants.FEAT_SHAPE, tf.float32),
        'mask': tf.io.FixedLenFeature([constants.SEQ_LEN], tf.float32),
    }

    def _parse_function(example_proto):
        example = tf.io.parse_single_example(example_proto, feature_description)
        # TF records were written float32 [0., 1.]
        sample = example['sample'] * 255.0
        sample = efficientnet.tfkeras.preprocess_input(sample)
        if is_training:
            return {'input_1': sample, 'input_2': example['mask']}, example['label']
        return {'input_1': sample, 'input_2': example['mask'], 'name': example['name']}, example['label']

    dataset = dataset.map(map_func=_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = balance_dataset(dataset, is_training)
    if is_training:
        dataset = dataset.map(image_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=256)
    else:
        dataset = dataset.cache()

    return dataset


def fraction_positives(y_true, y_pred):
    return tf.keras.backend.mean(y_true)


def compile_model(model, mode, lr):

    # optimizer = tf.keras.optimizers.Adam(lr=lr) #(lr=0.025)
    # optimizer = tf.keras.optimizers.RMSprop(lr, decay=1e-5, momentum=0.9)
    optimizer = tfa.optimizers.Lookahead(tf.keras.optimizers.SGD(lr, momentum=0.9))

    # learning_rate=CustomSchedule(D_MODEL)
    # optimizer = tf.keras.optimizers.Adam(
    #     learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # TODO keras needs custom_objects when loading models with custom metrics
    # But this one needs the optimizer.
    # def get_lr_metric(optimizer):
    #     def lr(y_true, y_pred):
    #         return optimizer.lr
    #     return lr
    # lr_metric = get_lr_metric(optimizer)

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
    ]

    # my_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)
    if mode == 'train':
        METRICS.append(fraction_positives)
        my_loss = tf.keras.losses.BinaryCrossentropy(
            label_smoothing=0.025
        )
        # my_loss = binary_focal_loss(alpha=0.7)
    else:
        my_loss = tf.keras.losses.BinaryCrossentropy(
            # label_smoothing=0.025
        )

    model.compile(loss=my_loss, optimizer=optimizer, metrics=METRICS)

    return model


def weighted_ce_logits(y_true, y_pred):
    return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, 0.3, name='weighted_ce_logits')


def load_feature_extractor(input_shape, extractor_file_name):

    custom_objs = {
        'fraction_positives':fraction_positives,
    }

    # TODO rename 'flatten' to something specific
    extractor_model = tf.keras.models.load_model(extractor_file_name,
        custom_objects=custom_objs)
    output = None
    for i, layer in enumerate(extractor_model.layers):
        # print(i, layer.name, layer.trainable)
        if layer.name == 'flatten':
            output = layer.output
            print('output set to {}.'.format(layer.name))
    if output is None:
        raise ValueError('Could not get feature extractor output')

    feature_extractor = Model(inputs=extractor_model.input, outputs=output)

    return feature_extractor


def load_efficientnetb1_model(input_shape, backbone_weights):

    backbone_model = EfficientNetB1(weights=None, input_shape=input_shape,
                                    include_top=False, pooling='avg')
    net = Flatten()(backbone_model.output)
    net = Dropout(0.25)(net)
    net = Dense(1, activation='sigmoid',
                kernel_regularizer=tf.keras.regularizers.l2(0.02))(net)
    model = Model(inputs=backbone_model.input, outputs=net)
    model.load_weights(backbone_weights)

    return backbone_model


def load_efficientnetb2_model(input_shape, backbone_weights):

    backbone_model = EfficientNetB2(weights=None, input_shape=input_shape,
                                    include_top=False, pooling='avg')
    net = Flatten()(backbone_model.output)
    net = Dropout(0.5)(net)
    net = Dense(1, activation='sigmoid',
                kernel_regularizer=tf.keras.regularizers.l2(0.02))(net)
    model = Model(inputs=backbone_model.input, outputs=net)
    model.load_weights(backbone_weights)

    return backbone_model


def load_meso_model(input_shape, weights):

    classifier = featx.MesoInception5(1, input_shape)
    # print(classifier.model.summary())
    # classifier.model.load_weights('pretrained/Meso/raw/all/weights.h5')
    classifier.model.load_weights(weights)

    for i, layer in enumerate(classifier.model.layers):
        # print(i, layer.name, layer.trainable)
        if layer.name == 'flatten':
            output = layer.output
            print('output set to {}.'.format(layer.name))

    feature_extractor = Model(inputs=classifier.model.input, outputs=output)

    return feature_extractor


def load_resnet_model(input_shape, weights):

    backbone_model = featx.resnet_18(input_shape, 4)

    for i, layer in enumerate(backbone_model.layers):
        # print(i, layer.name, layer.trainable)
        if layer.name == 'flatten':
            output = layer.output
            print('output set to {}.'.format(layer.name))

    feature_extractor = Model(inputs=backbone_model.input, outputs=output)

    return feature_extractor


# TODO: use Bidirectional and try narrower MobileNet
# TODO: explore removal of projection at the end of MobileNet to LSTM
# TODO: resolve from logits for all metrics, perhaps do your own weighted BCE
# oh wait its already in TF doc
def create_model(input_shape, model_name, backbone_weights):

    input_layer = Input(shape=input_shape)
    input_mask = Input(shape=(input_shape[0]))
    # reshape = Reshape([224, 224, 3])(input_layer)

    # 'pretrained/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5'
    # 'pretrained/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224_no_top.h5',
    # weights = 'pretrained/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
    weights = backbone_weights

    # feature_extractor = MobileNetV2(include_top=False,
    #     weights='pretrained/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224_no_top.h5',
    #     # weights='imagenet',
    #     alpha=0.5,
    #     input_shape=input_shape[-3:],
    #     pooling='avg'
    #     # pooling=None
    # )

    # feature_extractor = Xception(include_top=False, 
    #     weights='imagenet', 
    #     # input_tensor=None, 
    #     input_shape=None, 
    #     pooling='avg'
    # )
    
    if model_name == 'meso':
        feature_extractor = load_meso_model(input_shape[-3:], weights)
    elif model_name == 'efficientnetb2':
        feature_extractor = load_efficientnetb2_model(input_shape[-3:], weights)
    elif model_name == 'efficientnetb1':
        feature_extractor = load_efficientnetb1_model(input_shape[-3:], weights)
    elif model_name == 'resnet':
        feature_extractor = load_resnet_model(input_shape[-3:], weights)
    elif model_name == 'extractor':
        feature_extractor = load_feature_extractor(input_shape[-3:], weights)
    else:
        raise ValueError('Unknown feature extractor.')

    feature_extractor.trainable = False
    for i, layer in enumerate(feature_extractor.layers):
        layer.trainable = False
        print(i, layer.name, layer.trainable)
    print(feature_extractor.summary())

    # features = Conv2D(512, (1, 1), strides=(1, 1), padding='valid', activation='relu')(features)
    # features = Conv2D(256, (3, 3), strides=(2, 2), padding='valid', activation='relu')(features)
    # # # features = MaxPooling2D(pool_size=(2, 2))(features)
    # features = GlobalMaxPooling2D()(features)
    # features = Flatten()(features)
    # features = Dense(128, activation='elu')(features)
    # features = Dropout(0.25)(features)

    # feature_extractor = Model(inputs=mobilenet.input, outputs=features)
    # # print(feature_extractor.summary())
    
    # net = input_layer
    net = TimeDistributed(feature_extractor)(input_layer)
    # net = TimeDistributed(Flatten())(net)
    # net = TimeDistributed(Conv2D(256, (1, 1), strides=(1, 1), padding='valid', activation='relu'))(net)
    # net = TimeDistributed(Conv2D(D_MODEL, (3, 3), strides=(2, 2), padding='valid', activation='relu'))(net)
    # net = TimeDistributed(BatchNormalization())(net)
    # net = TimeDistributed(GlobalMaxPooling2D())(net)
    # net = Encoder(num_layers=1, d_model=D_MODEL, num_heads=2, dff=256,
    #     maximum_position_encoding=1000)(net, mask=input_mask)
    # net = multiply([net, input_mask])
    # net = Masking(mask_value = 0.0)(net)
    # net = Bidirectional(GRU(128, return_sequences=True))(net, mask=input_mask)
    # net = SeqSelfAttention(attention_type='additive', attention_activation='sigmoid')(net, mask=input_mask)
    # net = Bidirectional(GRU(64, dropout=0.25, return_sequences=True))(net, mask=input_mask)
    # net = ScaledDotProductAttention()(net, mask=input_mask)

    net = TimeDistributed(Dropout(0.25))(net)
    # net = TimeDistributed(Dense(32, activation='elu'))(net)
    # net = Bidirectional(GRU(32, return_sequences=False))(net, mask=input_mask)
    
    # net = Dense(256, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(net)
    net = SeqWeightedAttention()(net, mask=input_mask)
    net = Dropout(0.25)(net)
    out = Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.02),
        # bias_initializer=tf.keras.initializers.Constant(np.log([1.5]))
    )(net)
    # out = Dense(1, activation='sigmoid')(net)

    model = Model(inputs=[input_layer, input_mask], outputs=out)

    # This saturates GPU:0 for large models
    # parallel_model = multi_gpu_model(model, cpu_merge=False, gpus=4)
    # parallel_model = multi_gpu_model(model, gpus=2)
    # print("Training using multiple GPUs..")
    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.trainable)
   
    print(model.summary())
    return model


def step_decay(epoch):
    initial_lrate = 0.045
    drop = 0.9
    epochs_drop = 2.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    print('learning rate ', lrate)
    return lrate


if __name__ == '__main__':

    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--eval_dir', type=str)
    parser.add_argument('--output_dir', type=str, default='temp_toptrain')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--backbone_weights', type=str, default=None)
    parser.add_argument('--save', type=str, default='True')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
 
    in_shape = (constants.SEQ_LEN,) + constants.FEAT_SHAPE

    custom_objs = {
        'fraction_positives':fraction_positives,
        'SeqWeightedAttention':SeqWeightedAttention,
        # 'SeqSelfAttention':SeqSelfAttention,
        # 'weighted_ce_logits':weighted_ce_logits,
    }

    batch_size = args.batch_size
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # NOTE have to load extractor outside mirrored strategy
    # see https://www.gitmemory.com/issue/tensorflow/tensorflow/30850/513363504
    # extractor_model = None
    # if args.backbone_weights is not None:
    #     extractor_model = tf.keras.models.load_model(args.backbone_weights, custom_objects=custom_objs)
    #     print(extractor_model.summary())

    if args.mode == 'train':

        num_epochs = 1000
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        with strategy.scope():
            model = create_model(in_shape, args.model_name, args.backbone_weights)
            if args.load is not None:
                print('Loading model and weights from: ', args.load)
                # model = tf.keras.models.load_model(args.load, custom_objects=custom_objs)
                model.load_weights(args.load)
            else:
                print('Training model from scratch.')
            compile_model(model, args.mode, args.lr)

        train_dataset = tfrecords_dataset(args.train_dir, is_training=True)
        train_dataset = train_dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        eval_dataset = tfrecords_dataset(args.eval_dir, is_training=False)
        eval_dataset = eval_dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(output_dir, 'seq_%s_{epoch}.tf' % args.model_name),
                save_best_only=True,
                monitor='val_binary_crossentropy',
                mode='min',
                save_format='tf',
                save_weights_only=True,
                verbose=1),
            tf.keras.callbacks.EarlyStopping(
                # Stop training when `val_loss` is no longer improving
                # monitor='val_loss', # watch out for reg losses
                monitor='val_binary_crossentropy',
                min_delta=1e-4,
                patience=10,
                verbose=1),
            tf.keras.callbacks.CSVLogger(os.path.join(output_dir, 'training_seq_%s_log.csv' % args.model_name)),
            # tf.keras.callbacks.LearningRateScheduler(step_decay),
            # CosineAnnealingScheduler(T_max=num_epochs, eta_max=0.02, eta_min=1e-5),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                factor=0.96, patience=2, min_lr=5e-6, verbose=1, mode='min')
        ]
        
        # class_weight={0: 0.82, 1: 0.18}
        # class_weight=[0.99, 0.01]
        history = model.fit(train_dataset, epochs=num_epochs, # class_weight=class_weight, 
            validation_data=eval_dataset, #validation_steps=validation_steps, 
            callbacks=callbacks)
        save_loss(history, os.path.join(output_dir, 'final_%s_model' % args.model_name))

    elif args.mode == 'eval':
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        with strategy.scope():
            model = create_model(in_shape, args.model_name, args.backbone_weights)
            if args.load is not None:
                print('Loading model and weights from: ', args.load)
                # model = tf.keras.models.load_model(args.load, custom_objects=custom_objs)
                model.load_weights(args.load)
            else:
                raise ValueError('Eval mode needs --load argument.')
            compile_model(model, args.mode, args.lr)

        eval_dataset = tfrecords_dataset(args.eval_dir, is_training=False)
        eval_dataset = eval_dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        model.evaluate(eval_dataset)

    elif args.mode == 'predict':
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        with strategy.scope():
            model = create_model(in_shape, args.model_name, args.backbone_weights)
            if args.load is not None:
                print('Loading model and weights from: ', args.load)
                # model = tf.keras.models.load_model(args.load, custom_objects=custom_objs)
                model.load_weights(args.load)
            else:
                raise ValueError('Predict mode needs --weights argument.')
            compile_model(model, args.mode, args.lr)

        predict_dataset = tfrecords_dataset(args.eval_dir, is_training=False)
        predict_dataset = predict_dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        # predict_dataset = predict_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE).take(1)
        
        # for elem in tqdm.tqdm(predict_dataset):
        #     vid = elem[0]['input_1']
        #     mask = elem[0]['input_2'].numpy()
        #     name = str(elem[1].numpy(), 'utf-8')
        #     preds = model.predict(vid).flatten()
        #     preds *= mask
        #     pred = preds.mean()
        #     predictions.append(pred)
        #     truths.append(elem[2].numpy())
        #     saved.append([name, pred])
        predictions = []
        truths = []
        saved = []
        for batch in tqdm.tqdm(predict_dataset):
            # print(batch[0]['name'].numpy())
            names = [str(x, 'utf-8') for x in batch[0]['name'].numpy()]
            truths.extend(batch[1].numpy())
            preds = model.predict(batch[0])
            predictions.extend(preds)
            saved.extend(zip(names, preds))
        
        # print(saved)
        if len(predictions) > 0:
            print('Log loss on predictions: {}'.format(log_loss(truths, predictions, labels=[0, 1])))
            constants.save_predictions(saved)
        else:
            print('No predictions, check input.')
    
    if args.save == 'True':
        model_file_name = args.mode + '_final_full_seq_%s_model.h5' % args.model_name
        if args.load is not None:
            prefix, _ = os.path.splitext(os.path.basename(args.load))
            model_file_name = prefix + '_' + model_file_name
        model.save(os.path.join(output_dir, model_file_name))

        new_model = tf.keras.models.load_model(os.path.join(output_dir, model_file_name), custom_objects=custom_objs)
        new_model.summary()

    t1 = time.time()

    print("Execution took: {}".format(t1-t0))

