import argparse
import constants
import feature_extractor_models as featx
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import tensorflow_addons as tfa
import time

from efficientnet.tfkeras import EfficientNetB0, EfficientNetB3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Bidirectional, BatchNormalization, Concatenate, Conv2D, Dense, Dropout
from tensorflow.keras.layers import Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Input, GRU, LeakyReLU, LSTM, Masking, MaxPooling2D, multiply, Reshape, TimeDistributed
from tensorflow.keras.preprocessing import image as tfk_image
from keras_utils import binary_focal_loss, save_loss, LRFinder, SeqWeightedAttention, balance_dataset

tfkl = tf.keras.layers

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


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5, 5, n+1)
        plt.imshow(image_batch[n])
        plt.title(str(label_batch[n]))
        plt.axis('off')


def fraction_positives(y_true, y_pred):
    return tf.keras.backend.mean(y_true)


def compile_model(model, mode, lr):

    if mode == 'train':
        # optimizer = tfa.optimizers.Lookahead(tfa.optimizers.RectifiedAdam(lr))
        # optimizer = tf.keras.optimizers.Adam(lr)  # (lr=0.025)
        optimizer = tf.keras.optimizers.RMSprop(lr, decay=1e-5, momentum=0.9)
    elif mode == 'tune':
        # optimizer = tf.keras.optimizers.Adam()  # (lr=0.025)
        optimizer = tf.keras.optimizers.RMSprop(lr, decay=1e-6)
        # optimizer = tf.keras.optimizers.SGD(lr, momentum=0.9)
    elif mode == 'eval':
        optimizer = tf.keras.optimizers.SGD(lr)

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
        # lr_metric,
        # Write TensorBoard logs to `./logs` directory
    ]
    if mode == 'train' or mode == 'tune':
        METRICS.append(fraction_positives)
    # my_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)
    my_loss = tf.keras.losses.BinaryCrossentropy(
        label_smoothing=0.025
    )
    # my_loss = binary_focal_loss(alpha=0.5)
    # my_loss = 'mean_squared_error'
    print('Using loss: %s, optimizer: %s' % (my_loss, optimizer))
    model.compile(loss=my_loss, optimizer=optimizer, metrics=METRICS)

    return model


def create_meso_model(input_shape, mode):

    classifier = featx.MesoInception5(width=1, input_shape=input_shape)

    # classifier = featx.MesoInception4(input_shape)

    if mode == 'train':
        # meso4_weights = 'pretrained/Meso/c23/all/weights.h5'
        # classifier.model.load_weights(meso4_weights)
        print('\nUnfreezing all conv Meso layers!')
        # for layer in classifier.model.layers:
        #     if 'dense' not in layer.name:
        #         layer.trainable = False
    if mode == 'tune':
        print('\nUnfreezing all Meso layers!')

    for i, layer in enumerate(classifier.model.layers):
        print(i, layer.name, layer.trainable)
    print(classifier.model.summary())

    return classifier.model


def create_onemil_model(input_shape, mode):

    classifier = featx.OneMIL(input_shape)

    for i, layer in enumerate(classifier.model.layers):
        print(i, layer.name, layer.trainable)

    print(classifier.model.summary())

    return classifier.model


def create_xception_model(input_shape, mode):

    input_tensor = Input(shape=input_shape)
    xception_weights = 'pretrained/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'

    if mode == 'train':
        print('Loading xception weights from: ', xception_weights)
        base_model = Xception(weights=xception_weights,
                            input_tensor=input_tensor, include_top=False, pooling='avg')
        # print('\nFreezing all Xception layers!')
        # for layer in base_model.layers:
        #     layer.trainable = False
        print('\nUnfreezing last Xception layers!')
        for layer in base_model.layers[:126]:
            layer.trainable = False
        for layer in base_model.layers[126:]:
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
    net = Dropout(0.25)(net)
    out = Dense(1, activation='sigmoid',
                kernel_regularizer=tf.keras.regularizers.l2(0.02))(net)

    model = Model(inputs=base_model.input, outputs=out)
    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.trainable)
    print(model.summary())

    return model


def create_mobilenet_model(input_shape, mode):

    input_tensor = Input(shape=input_shape)
    # create the base pre-trained model
    mobilenet_weights = 'pretrained/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224_no_top.h5'
    print('Loading mobilenet weights from: ', mobilenet_weights)
    base_model = MobileNetV2(weights=mobilenet_weights, alpha=0.5,
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
    out = Dense(1, activation='sigmoid',
                kernel_regularizer=tf.keras.regularizers.l2(0.02))(net)

    model = Model(inputs=base_model.input, outputs=out)
    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.trainable)
    print(model.summary())

    return model


def create_efficientnet_model(input_shape, mode):

    input_tensor = Input(shape=input_shape)
    # create the base pre-trained model
    efficientnet_weights = None
    if mode == 'train':
        efficientnet_weights = 'pretrained/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
    print('Loaded efficientnet weights from: ', efficientnet_weights)
    base_model = EfficientNetB0(weights=efficientnet_weights, input_tensor=input_tensor,
                                include_top=False, pooling='avg')

    if mode == 'train':
        print('\nFreezing all EfficientNet layers!')
        # for layer in base_model.layers:
        #     layer.trainable = False
        print('\nUnfreezing last efficient net layers!')
        for layer in base_model.layers[:227]:
            layer.trainable = False
        for layer in base_model.layers[227:]:
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
    out = Dense(1, activation='sigmoid',
                kernel_regularizer=tf.keras.regularizers.l2(0.02))(net)

    model = Model(inputs=base_model.input, outputs=out)
    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.trainable)
    print(model.summary())

    return model


def create_resnet_model(input_shape, mode):

    model = featx.resnet_18(input_shape, num_filters=4)
    print(model.summary())

    return model


def create_model(model_name, input_shape, mode):
    if model_name == 'mobilenet':
        return create_mobilenet_model(input_shape, mode)
    if model_name == 'meso':
        return create_meso_model(input_shape, mode)
    if model_name == 'onemil':
        return create_onemil_model(input_shape, mode)
    if model_name == 'xception':
        return create_xception_model(input_shape, mode)
    if model_name == 'resnet':
        return create_xception_model(input_shape, mode)
    if model_name == 'efficientnet':
        return create_efficientnet_model(input_shape, mode)

    raise ValueError('Unknown model %s' % model_name)


def train_val_data_gen(train_dir, eval_dir, image_size, batch_size):
    train_datagen = tfk_image.ImageDataGenerator(
        rescale=1./255,
        # shear_range=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    eval_datagen = tfk_image.ImageDataGenerator(
        rescale=1./255
    )
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
    )
    eval_generator = eval_datagen.flow_from_directory(
        eval_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        save_to_dir='eval_images'
    )

    return train_generator, eval_generator


if __name__ == '__main__':

    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str)
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--eval_dir', type=str)
    parser.add_argument('--model_name', type=str, default='unknown')
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--save', type=str, default='True')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=512)
    args = parser.parse_args()

    num_epochs = 1000
    # validation_steps = 32
    batch_size = int(args.batch_size)
    model_name = args.model_name
    in_shape = (constants.MESO_INPUT_HEIGHT, constants.MESO_INPUT_WIDTH, 3)

    custom_objs = {
        'fraction_positives': fraction_positives,
        'SeqWeightedAttention': SeqWeightedAttention,
        'binary_focal_loss_fixed': binary_focal_loss(alpha=0.47),
    }

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        # due to bugs need to load weights for mirrored strategy - cannot load full model
        model = create_model(model_name, in_shape, args.mode)
        if args.load is not None:
            print('\nLoading weights from: ', args.load)
            # model = tf.keras.models.load_model(args.load, custom_objects=custom_objs)
            model.load_weights(args.load)
        else:
            print('\nTraining model from scratch.')
        compile_model(model, args.mode, args.lr)

    train_generator, eval_generator = train_val_data_gen(args.train_dir, args.eval_dir, in_shape[:2], batch_size)

    if args.mode == 'train' or args.mode == 'tune':

        # lr_callback = LRFinder(num_samples=33501, batch_size=batch_size,
        #                minimum_lr=1e-5, maximum_lr=5e-1,
        #                # validation_data=(X_val, Y_val),
        #                lr_scale='exp', save_dir='.')

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath='krs_featx_weights_%s_{epoch}.h5' % (model_name + '_' + args.mode),
                save_best_only=True,
                monitor='val_binary_crossentropy',
                # save_format='tf',
                save_weights_only=True,
                verbose=1),
            tf.keras.callbacks.EarlyStopping(
                # Stop training when `val_loss` is no longer improving
                # monitor='val_loss', # watch out for reg losses
                monitor='val_auc',
                min_delta=1e-4,
                patience=10,
                mode='max',
                verbose=1),
            tf.keras.callbacks.CSVLogger(
                'krs_training_featx_%s_log.csv' % model_name),
            # tf.keras.callbacks.LearningRateScheduler(step_decay),
            # CosineAnnealingScheduler(T_max=num_epochs, eta_max=0.02, eta_min=1e-5),
            # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
            #                                      factor=0.96, patience=3, min_lr=5e-5, verbose=1, mode='min'),
            # lr_callback,
            # tf.keras.callbacks.TensorBoard(log_dir='./train_featx_%s_logs' % model_name),
        ]

        # class_weight = {0: 0.45, 1: 0.55}
        history = model.fit(train_generator, epochs=num_epochs,  # class_weight=class_weight,
                            validation_data=eval_generator,  # validation_steps=validation_steps,
                            callbacks=callbacks)

        # lr_callback.plot_schedule()
        save_loss(history, 'krs_final_featx_%s_model' % model_name)

    elif args.mode == 'eval':
        model.evaluate(eval_generator)

    if args.save == 'True':
        model_file_name = args.mode + '_krs_featx_full_%s_model.h5' % model_name
        if args.load is not None:
            model_file_name = args.load + '_' + model_file_name
        model.save(model_file_name)

        new_model = tf.keras.models.load_model(
            model_file_name, custom_objects=custom_objs)
        new_model.summary()

    t1 = time.time()

    print("Execution took: {}".format(t1-t0))
