import argparse
import constants
import feature_extractor_models as featx
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import time

# from efficientnet.tfkeras import EfficientNetB0, EfficientNetB3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Bidirectional, BatchNormalization, Concatenate, Conv2D, Dense, Dropout
from tensorflow.keras.layers import Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Input, GRU, LeakyReLU, LSTM, Masking, MaxPooling2D, multiply, Reshape, TimeDistributed

from keras_utils import binary_focal_loss, save_loss, LRFinder, SeqWeightedAttention

tfkl = tf.keras.layers

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
    # img = tf.image.pad_to_bounding_box(img, offset_height=0, offset_width=0, 
    #     target_height=constants.MESO_INPUT_HEIGHT, target_width=constants.MESO_INPUT_WIDTH)
    
    # Xception
    # img = tf.cast(img, tf.float32)
    
    # resize the image to the desired size.
    # img = tf.image.resize(img, [constants.MESO_INPUT_HEIGHT, constants.MESO_INPUT_WIDTH])
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
    x = tf.image.random_brightness(x, 0.1)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_jpeg_quality(x, min_jpeg_quality=50, max_jpeg_quality=100)
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


def class_func(feat, label):
    return label


def balance_dataset(dset):
    negative_ds = dset.filter(lambda features, label: label==0).take(36267) # eval 0-9
    # num_neg_elements = tf.data.experimental.cardinality(negative_ds).numpy()
    # positive_ds = dset.filter(lambda features, label: label==1).take(37436)
    # positive_ds = dset.filter(lambda features, label: label==1).take(6239) # eval 0,1,2
    # positive_ds = dset.filter(lambda features, label: label==1).take(12378)  # eval 0, 1, 2, 3, 4
    positive_ds = dset.filter(lambda features, label: label==1)
    # print('Negative dataset class fractions: ', class_fractions(negative_ds))
    # print('Positive dataset class fractions: ', class_fractions(positive_ds))
    
    balanced_ds = tf.data.experimental.sample_from_datasets(
        [negative_ds, positive_ds], [0.5, 0.5]
    )
    return balanced_ds


def prepare_dataset(ds, is_training, batch_size, cache, shuffle_buffer_size=60000):
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.

    if is_training:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    else:
        # TODO: seems rejection_resample doesn't work with keras fit
        # resampler = tf.data.experimental.rejection_resample(
        #     class_func, target_dist=[0.5, 0.5]) #, initial_dist=[0.345, 0.655])
        # ds = ds.apply(resampler)
        # ds = ds.map(lambda extra_label, features_and_label: features_and_label)

        ds = balance_dataset(ds)

    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

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


def fraction_positives(y_true, y_pred):
    return tf.keras.backend.mean(y_true)


def compile_model(model, mode, lr):

    if mode == 'train' or mode == 'eval':
        optimizer = tf.keras.optimizers.Adam(lr)  # (lr=0.025)
        # optimizer = tf.keras.optimizers.RMSProp(lr)  # (lr=0.025)
    elif mode == 'tune':
        # optimizer = tf.keras.optimizers.Adam()  # (lr=0.025)
        optimizer = tf.keras.optimizers.RMSprop(lr, decay=1e-6)
        # optimizer = tf.keras.optimizers.SGD(lr, momentum=0.9)

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
        # label_smoothing=0.025
    )
    # my_loss = binary_focal_loss(alpha=0.57)
    # my_loss = 'mean_squared_error'
    model.compile(loss=my_loss, optimizer=optimizer, metrics=METRICS)

    return model


def create_meso_model(input_shape, mode):

    classifier = featx.MesoInception5(width=1, input_shape=input_shape)

    # classifier = featx.MesoInception4(input_shape)
    # meso4_weights = 'pretrained/Meso/c23/all/weights.h5'
    # classifier.model.load_weights(meso4_weights)

    if mode == 'train':
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
    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.trainable)
    print(model.summary())

    return model


def create_mobilenet_model(input_shape, mode):

    input_tensor = Input(shape=input_shape)
    # create the base pre-trained model
    mobilenet_weights = 'pretrained/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224_no_top.h5'
    print('Loading mobilenet weights from: ', mobilenet_weights)
    base_model = MobileNetV2(weights=mobilenet_weights, alpha = 0.5,
        input_tensor=input_tensor, include_top=False, pooling='avg')

    if mode == 'train':
        print('\nFreezing all Mobilenet layers!')
        # for layer in base_model.layers:
        #     layer.trainable = False
        print('\nUnfreezing last Mobilenet layers!')
        # for layer in base_model.layers[:152]:
        #     layer.trainable = False
        # for layer in base_model.layers[152:]:
        #     layer.trainable = True
        for layer in base_model.layers[:135]:
            layer.trainable = False
        for layer in base_model.layers[135:]:
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


def create_efficientnet_model(input_shape, mode):

    input_tensor = Input(shape=input_shape)
    # create the base pre-trained model
    efficientnet_weights = 'pretrained/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
    print('Loading efficientnet weights from: ', efficientnet_weights)
    base_model = EfficientNetB0(weights=efficientnet_weights, input_tensor=input_tensor, 
        include_top=False, pooling='avg')

    if mode == 'train':
        print('\nFreezing all EfficientNet layers!')
        # for layer in base_model.layers:
        #     layer.trainable = False
        print('\nUnfreezing last efficient net layers!')
        for layer in base_model.layers[:214]:
            layer.trainable = False
        for layer in base_model.layers[214:]:
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
    out = Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.02))(net)

    model = Model(inputs=base_model.input, outputs=out)
    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.trainable)
    print(model.summary())

    return model


def create_resnet_model(input_shape, mode):

    model = featx.resnet_18(input_shape, num_filters=8)
    print(model.summary())

    return model


def count(counts, batch):
  features, labels = batch
  class_1 = labels == 1
  class_1 = tf.cast(class_1, tf.int32)

  class_0 = labels == 0
  class_0 = tf.cast(class_0, tf.int32)

  counts['class_0'] += tf.reduce_sum(class_0)
  counts['class_1'] += tf.reduce_sum(class_1)

  return counts


def class_fractions(dset):
    counts = dset.reduce(initial_state={'class_0': 0, 'class_1': 0}, reduce_func = count)
    counts = np.array([counts['class_0'].numpy(), counts['class_1'].numpy()]).astype(np.float32)
    fractions = counts/counts.sum()

    return fractions, counts


if __name__ == '__main__':

    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str)
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--eval_dir', type=str)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--save', type=str, default='True')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=512)
    args = parser.parse_args()

    num_epochs = 1000
    # validation_steps = 32
    batch_size = int(args.batch_size)
    in_shape = (constants.MESO_INPUT_HEIGHT, constants.MESO_INPUT_WIDTH, 3)

    custom_objs = {
        'fraction_positives': fraction_positives,
        'SeqWeightedAttention': SeqWeightedAttention,
    }

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        # due to bugs need to load weights for mirrored strategy - cannot load full model
        # model = create_meso_model(in_shape, args.mode)
        model = create_onemil_model(in_shape, args.mode)
        if args.load is not None:
            print('\nLoading weights from: ', args.load)
            # model = tf.keras.models.load_model(args.load, custom_objects=custom_objs)
            model.load_weights(args.load)
        else:
            print('\nTraining model from scratch.')
        compile_model(model, args.mode, args.lr)

    eval_dataset = input_dataset(args.eval_dir, is_training=False, batch_size=batch_size, cache=True)
    fractions, counts = class_fractions(eval_dataset)
    print('Eval dataset class counts {} and fractions {}: '.format(counts, fractions))

    if args.mode == 'train' or args.mode == 'tune':
        train_dataset = input_dataset(args.train_dir, is_training=True, batch_size=batch_size,
            cache=False
            # cache='/raid/scratch/training_cache.mem'
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
                patience=25,
                verbose=1),
            tf.keras.callbacks.CSVLogger('training_featx_log.csv'),
            # tf.keras.callbacks.LearningRateScheduler(step_decay),
            # CosineAnnealingScheduler(T_max=num_epochs, eta_max=0.02, eta_min=1e-5),
            # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
            #                                      factor=0.96, patience=3, min_lr=5e-5, verbose=1, mode='min'),
            # lr_callback,
            # tf.keras.callbacks.TensorBoard(log_dir='./train_featx_logs'),
        ]

        class_weight={0: 0.45, 1: 0.55}
        history = model.fit(train_dataset, epochs=num_epochs, class_weight=class_weight,
                            validation_data=eval_dataset,  # validation_steps=validation_steps,
                            callbacks=callbacks)
        
        # lr_callback.plot_schedule()
        save_loss(history, 'final_featx_model')

    elif args.mode == 'eval':
        model.evaluate(eval_dataset)

    if args.save == 'True':
        model_file_name = args.mode + '_featx_full_model.h5'
        if args.load is not None:
            model_file_name = args.load + '_' + model_file_name
        model.save(model_file_name)

        new_model = tf.keras.models.load_model(model_file_name, custom_objects=custom_objs)
        new_model.summary()

    t1 = time.time()

    print("Execution took: {}".format(t1-t0))
