import argparse
import constants
import gc
import glob

import numpy as np
import os
import pickle
from sklearn.metrics import log_loss
import time
import tensorflow as tf
import tqdm
from keras_utils import SeqWeightedAttention

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Bidirectional, BatchNormalization, Concatenate, Conv2D, Dense, Dropout
from tensorflow.keras.layers import Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Input, GRU, LeakyReLU, LSTM, Masking, MaxPooling2D, multiply, Reshape, TimeDistributed


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the last GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[-1], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[-1], True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)


def fraction_positives(y_true, y_pred):
    return tf.keras.backend.mean(y_true)


def predict_dataset(file_pattern):

    file_list = tf.data.Dataset.list_files(file_pattern)
    dataset = tf.data.TFRecordDataset(filenames=file_list, buffer_size=None, num_parallel_reads=4)

    feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'name': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'sample': tf.io.FixedLenFeature((constants.SEQ_LEN,) + constants.FEAT_SHAPE, tf.float32),
        'mask': tf.io.FixedLenFeature([constants.SEQ_LEN], tf.float32),
    }

    def _parse_function(example_proto):
        example = tf.io.parse_single_example(example_proto, feature_description)
        sample = (example['sample'] + 1.0) / 2
        # sample = tf.image.resize(sample, (constants.MESO_INPUT_HEIGHT, constants.MESO_INPUT_WIDTH))

        return {'input_1': sample, 'input_2': example['mask']}, example['name'], example['label']

    dataset = dataset.map(map_func=_parse_function, num_parallel_calls=4)

    return dataset


# TODO break these out
def compile_model(model):

    optimizer = tf.keras.optimizers.Adam()

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
    my_loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05)
    model.compile(loss=my_loss, optimizer=optimizer, metrics=METRICS)

    return model


class MesoInception4():
    def __init__(self, learning_rate = 0.001):
        self.model = self.init_model()
        # optimizer = Adam(lr = learning_rate)
        # self.model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])
    
    def InceptionLayer(self, a, b, c, d):
        def func(x):
            x1 = Conv2D(a, (1, 1), padding='same', activation='relu')(x)
            
            x2 = Conv2D(b, (1, 1), padding='same', activation='relu')(x)
            x2 = Conv2D(b, (3, 3), padding='same', activation='relu')(x2)
            
            x3 = Conv2D(c, (1, 1), padding='same', activation='relu')(x)
            x3 = Conv2D(c, (3, 3), dilation_rate = 2, strides = 1, padding='same', activation='relu')(x3)
            
            x4 = Conv2D(d, (1, 1), padding='same', activation='relu')(x)
            x4 = Conv2D(d, (3, 3), dilation_rate = 3, strides = 1, padding='same', activation='relu')(x4)

            y = Concatenate(axis = -1)([x1, x2, x3, x4])
            
            return y
        return func
    
    def init_model(self):
        x = Input(shape = (constants.MESO_INPUT_HEIGHT, constants.MESO_INPUT_WIDTH, 3))
        
        x1 = self.InceptionLayer(1, 4, 4, 2)(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
        x2 = self.InceptionLayer(2, 4, 4, 2)(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)        
        
        x3 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
        
        x4 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
        
        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation = 'sigmoid')(y)

        return Model(inputs = x, outputs = y)


class MesoInception5():
    def __init__(self, width):
        self.width = width
        self.model = self.init_model()
    
    def InceptionLayer(self, a, b, c, d):
        def func(x):
            x1 = Conv2D(a, (1, 1), padding='same', activation='relu')(x)
            
            x2 = Conv2D(b, (1, 1), padding='same', activation='relu')(x)
            x2 = Conv2D(b, (3, 3), padding='same', activation='relu')(x2)
            
            x3 = Conv2D(c, (1, 1), padding='same', activation='relu')(x)
            x3 = Conv2D(c, (3, 3), dilation_rate = 2, strides = 1, padding='same', activation='relu')(x3)
            
            x4 = Conv2D(d, (1, 1), padding='same', activation='relu')(x)
            x4 = Conv2D(d, (3, 3), dilation_rate = 3, strides = 1, padding='same', activation='relu')(x4)

            y = Concatenate(axis = -1)([x1, x2, x3, x4])
            
            return y
        return func
    
    def init_model(self):
        x = Input(shape = (224, 224, 3))
        
        x1 = self.InceptionLayer(1*self.width, 2*self.width, 2*self.width, 1*self.width)(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
        x2 = self.InceptionLayer(2*self.width, 4*self.width, 4*self.width, 2*self.width)(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)        
        
        x3 = self.InceptionLayer(2*self.width, 4*self.width, 4*self.width, 2*self.width)(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)        

        x4 = self.InceptionLayer(4*self.width, 8*self.width, 8*self.width, 4*self.width)(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(2, 2), padding='same')(x4)        

        x5 = self.InceptionLayer(8*self.width, 16*self.width, 16*self.width, 8*self.width)(x4)
        x5 = BatchNormalization()(x5)
        x5 = MaxPooling2D(pool_size=(2, 2), padding='same')(x5)
        
        x6 = Conv2D(64*self.width, (3, 3), padding='same', activation = 'relu')(x5)
        x6 = BatchNormalization()(x6)
        x6 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x6)
        
        y = Flatten()(x6)
        y = Dropout(0.5)(y)
        #TODO investigate num units for this dense layer.
        y = Dense(32*self.width)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation = 'sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01))(y)

        return Model(inputs = x, outputs = y)


def create_meso_model(input_shape):

    classifier = MesoInception5(width=1)

    for i, layer in enumerate(classifier.model.layers):
        print(i, layer.name, layer.trainable)

    return classifier.model

if __name__ == '__main__':

    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrec', type=str, default=None)
    parser.add_argument('--imgs', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--save', type=str, default=None)
    args = parser.parse_args()

    assert args.model is not None or args.weights is not None
    assert args.tfrec is not None or args.imgs is not None 

    custom_objs = {
        'fraction_positives':fraction_positives,
        # 'SeqWeightedAttention':SeqWeightedAttention,
    }

    if args.model is not None:
        model = tf.keras.models.load_model(args.model, custom_objects=custom_objs)
    elif args.weights is not None:
        # TODO TF complains about function tracing when distributing predictions on mirrored.
        # strategy = tf.distribute.MirroredStrategy()
        # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        # with strategy.scope():
        model = create_meso_model((constants.SEQ_LEN,) + constants.FEAT_SHAPE)
        model.load_weights(args.weights)
        compile_model(model)
        if args.save is not None:
            model_file, _ = os.path.splitext(args.weights)
            model.save(model_file + '_saved_model.h5')
    
    print(model.summary())

    # TODO use tf model maybe as it can be retrained
    # model.save('tf_model')
    # print("Loading tf SavedModel")
    # loaded_model = load_model('tf_model', custom_objects=custom_objs, compile=False)
    # loaded_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.025), 
    #     loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05))

    predictions = []
    truths = []
    saved = []

    if args.tfrec is not None:
        dataset = predict_dataset(args.tfrec)
        for elem in tqdm.tqdm(dataset):
            vid = elem[0]['input_1']
            mask = elem[0]['input_2'].numpy()
            name = str(elem[1].numpy(), 'utf-8')
            preds = model.predict(vid).flatten()
            preds *= mask
            pred = preds.mean()
            predictions.append(pred)
            truths.append(elem[2].numpy())
            saved.append([name, pred])

    # print(saved)
    if len(predictions) > 0:
        print('Log loss on predictions: {}'.format(log_loss(truths, predictions, labels=[0, 1])))
        constants.save_predictions(saved)
    else:
        print('No predictions, check input.')

    t1 = time.time()
    print("Execution took: {}".format(t1-t0))
