from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import constants
import cv2
import feature_extractor_models as featx
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import tensorflow as tf
import time
import threading
import tqdm

from scipy.interpolate import griddata
from sklearn.metrics import log_loss

from efficientnet.tfkeras import EfficientNetB0
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.xception import Xception

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Bidirectional, BatchNormalization, Concatenate, Conv2D, Dense, Dropout
from tensorflow.keras.layers import Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Input, GRU, LeakyReLU, LSTM, Masking, MaxPooling2D, multiply, Reshape, TimeDistributed
# from tensorflow.keras.utils import multi_gpu_model

from keras_utils import ScaledDotProductAttention, SeqSelfAttention, SeqWeightedAttention, binary_focal_loss, save_loss
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

# TODO there's an input warning that input doesnt come from input layer
# TODO use parallel_interleave

D_MODEL = 784

def save_sample_img(name, label, values):
    IMG_SIZE = values[0].shape[0]    

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 4
    font_scale = 2

    # line_shape = (IMG_SIZE, max_elems*IMG_SIZE, 3)
    tile_shape = (IMG_SIZE, constants.SEQ_LEN*IMG_SIZE, 3)
    tile_img = np.zeros(tile_shape, dtype=np.float32)
    for j in range(len(values)):
        color = (0, 255, 0) if label == 0 else (255, 0, 0)
        cv2.putText(tile_img, name, (10, 50),
                        font_face, font_scale,
                        color, thickness, 2)
        
        tile_img[:, j*IMG_SIZE:(j+1)*IMG_SIZE, :] = values[j]

    plt.imsave(name+'.jpg', tile_img)

    return tile_img


def tfrecords_dataset(input_dir, is_training):
    print('Using tfrecords dataset from: ', input_dir)
    
    file_list = tf.data.Dataset.list_files(input_dir).shuffle(512)
    dataset = tf.data.TFRecordDataset(filenames=file_list, 
        buffer_size=None, 
        num_parallel_reads=tf.data.experimental.AUTOTUNE
    )
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)

    feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'name': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'sample': tf.io.FixedLenFeature((constants.SEQ_LEN,) + constants.FEAT_SHAPE, tf.float32),
        'mask': tf.io.FixedLenFeature([constants.SEQ_LEN], tf.float32),
    }

    def _parse_function(example_proto):
        example = tf.io.parse_single_example(example_proto, feature_description)
        sample = (example['sample'] + 1.0) / 2
        if is_training:
            return {'input_1': sample, 'input_2': example['mask']}, example['label']
        return {'input_1': sample, 'input_2': example['mask'], 'name': example['name']}, example['label']

    dataset = dataset.map(map_func=_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)
    return dataset


def fraction_positives(y_true, y_pred):
    return tf.keras.backend.mean(y_true)


def compile_model(model, mode, lr):

    optimizer = tf.keras.optimizers.Adam(lr=lr) #(lr=0.025)
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
            # label_smoothing=0.025
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


def create_efficientnet_model(input_shape):

    # create the base pre-trained model
    # efficientnet_weights = 'pretrained/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
    efficientnet_weights = None
    print('Loading efficientnet weights from: ', efficientnet_weights)
    base_model = EfficientNetB0(weights=efficientnet_weights, # input_tensor=input_layer, 
        input_shape=input_shape, 
        include_top=False, pooling='avg')

    net = base_model.output
    net = Dropout(0.5)(net)
    out = Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.02))(net)

    model = Model(inputs=base_model.input, outputs=out)

    return model, base_model


# TODO: use Bidirectional and try narrower MobileNet
# TODO: explore removal of projection at the end of MobileNet to LSTM
# TODO: resolve from logits for all metrics, perhaps do your own weighted BCE
# oh wait its already in TF doc
def create_model(input_shape):

    input_layer = Input(shape=input_shape)
    input_mask = Input(shape=(input_shape[0]))
    # reshape = Reshape([224, 224, 3])(input_layer)

    # classifier = featx.MesoInception5(width=1)
    # # print(classifier.model.summary())
    # # classifier.model.load_weights('pretrained/Meso/raw/all/weights.h5')
    # classifier.model.load_weights('one_model_weights.h5')

    # for i, layer in enumerate(classifier.model.layers):
    #     # print(i, layer.name, layer.trainable)
    #     if layer.name == 'flatten':
    #         output = layer.output
    #         print('output set to {}.'.format(layer.name))

    # feature_extractor = Model(inputs=classifier.model.input, outputs=output)

    # 'pretrained/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5'
    # 'pretrained/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224_no_top.h5',
    # weights = 'pretrained/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
    weights = 'one_model_weights.h5'

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
    effnet, feature_extractor = create_efficientnet_model(input_shape[-3:])
    print('Loading feature extractor weights from: ', weights)
    effnet.load_weights(weights)

    # feature_extractor = EfficientNetB0(weights=weights, input_shape=input_shape[-3:], 
    #     include_top=False, pooling='avg')
    
    feature_extractor.trainable = False
    for i, layer in enumerate(feature_extractor.layers):
        layer.trainable = False
        print(i, layer.name, layer.trainable)
    print(feature_extractor.summary())

    # for layer in feature_extractor.layers:
    #     if (('block_16' not in layer.name) # and ('block_15' not in layer.name)
    #         # and ('block_14' not in layer.name) and ('block_13' not in layer.name)
    #         # and ('block_12' not in layer.name) and ('block_11' not in layer.name)
    #         # and ('block_10' not in layer.name) and ('block_9' not in layer.name)
    #     ):
    #         layer.trainable = False
    #     else:
    #         print('Layer {} trainable {}'.format(layer.name, layer.trainable))
    
    # for layer in feature_extractor.layers:
    #     if (('block_1_' in layer.name) or ('block_2_' in layer.name) 
    #         or ('block_3_' in layer.name) or ('block_4_' in layer.name)
    #         or ('block_5_' in layer.name) or ('block_6_' in layer.name)
    #         or ('block_7_' in layer.name) or ('block_8_' in layer.name)
    #     ):
    #         layer.trainable = False
    #         print('Layer {} trainable {}'.format(layer.name, layer.trainable))

    # features = mobilenet.output
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

    net = SeqWeightedAttention()(net, mask=input_mask)
    # net = Bidirectional(GRU(128, return_sequences=False))(net, mask=input_mask)
    
    # net = Dense(256, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(net)
    # net = Dropout(0.25)(net)
    out = Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01),
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
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--save', type=str, default='true')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
 
    in_shape = (constants.SEQ_LEN,) + constants.FEAT_SHAPE

    custom_objs = {
        # 'fraction_positives':fraction_positives,
        'SeqWeightedAttention':SeqWeightedAttention,
        # 'SeqSelfAttention':SeqSelfAttention,
        # 'weighted_ce_logits':weighted_ce_logits,
    }

    batch_size = args.batch_size

    if args.mode == 'train':

        num_epochs = 1000
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        with strategy.scope():
            model = create_model(in_shape)
            if args.weights is not None:
                print('Loading model and weights from: ', args.weights)
                # model = tf.keras.models.load_model(args.load, custom_objects=custom_objs)
                model.load_weights(args.weights)
            else:
                print('Training model from scratch.')
            compile_model(model, args.mode, args.lr)

        train_dataset = tfrecords_dataset(args.train_dir, is_training=True)
        train_dataset = train_dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        eval_dataset = tfrecords_dataset(args.eval_dir, is_training=False)
        eval_dataset = eval_dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath='fattw_{epoch}.h5',
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
                patience=30,
                verbose=1),
            tf.keras.callbacks.CSVLogger('training_log.csv'),
            # tf.keras.callbacks.LearningRateScheduler(step_decay),
            # CosineAnnealingScheduler(T_max=num_epochs, eta_max=0.02, eta_min=1e-5),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                factor=0.96, patience=2, min_lr=5e-6, verbose=1, mode='min')
        ]
        
        class_weight={0: 0.7, 1: 0.3}
        # class_weight=[0.99, 0.01]
        history = model.fit(train_dataset, epochs=num_epochs, class_weight=class_weight, 
            validation_data=eval_dataset, #validation_steps=validation_steps, 
            callbacks=callbacks)
        save_loss(history, 'final_model')

    elif args.mode == 'eval':
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        with strategy.scope():
            model = create_model(in_shape)
            if args.weights is not None:
                print('Loading model and weights from: ', args.weights)
                # model = tf.keras.models.load_model(args.weights, custom_objects=custom_objs)
                model.load_weights(args.weights)
            else:
                raise ValueError('Predict mode needs --weights argument.')
            compile_model(model, args.mode, args.lr)

        eval_dataset = tfrecords_dataset(args.eval_dir, is_training=False)
        eval_dataset = eval_dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        model.evaluate(eval_dataset)

    elif args.mode == 'predict':
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        with strategy.scope():
            model = create_model(in_shape)
            if args.weights is not None:
                print('Loading model and weights from: ', args.weights)
                # model = tf.keras.models.load_model(args.weights, custom_objects=custom_objs)
                model.load_weights(args.weights)
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
    
    if args.save == 'true':
        model_file_name = args.mode + '_final_full_model.h5'
        if args.weights is not None:
            prefix, _ = os.path.splitext(os.path.basename(args.weights))
            model_file_name = prefix + '_' + model_file_name
        model.save(model_file_name)

        new_model = tf.keras.models.load_model(model_file_name, custom_objects=custom_objs)
        new_model.summary()

    t1 = time.time()

    print("Execution took: {}".format(t1-t0))

