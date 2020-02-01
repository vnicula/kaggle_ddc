import argparse
import gc
import glob

import numpy as np
import os
import pickle
from sklearn.metrics import log_loss
import time
import tensorflow as tf
from keras_utils import SeqWeightedAttention

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Bidirectional, BatchNormalization, Concatenate, Conv2D, Dense, Dropout
from tensorflow.keras.layers import Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Input, GRU, LeakyReLU, LSTM, Masking, MaxPooling2D, multiply, Reshape, TimeDistributed


SEQ_LEN = 30
FEAT_SHAPE = (224, 224, 3)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[3], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[3], True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)


def fraction_positives(y_true, y_pred):
    return tf.keras.backend.mean(y_true)


def read_file(file_path):
    
    names = []
    labels = []
    samples = []
    masks = []
    
    with open(file_path, 'rb') as f_p:
        data = pickle.load(f_p)

        for key in data.keys():
            label = data[key][0]

            feat_shape = data[key][1][0].shape
            my_seq_len = len(data[key][1])
            data_seq_len = min(my_seq_len, SEQ_LEN)
            sample = np.zeros((SEQ_LEN,) + feat_shape, dtype=np.float32)
            # mask = np.zeros(SEQ_LEN, dtype=np.float32)
            mask = np.zeros(SEQ_LEN, dtype=np.float32)
            for indx in range(data_seq_len):
                # NOTE mesonet seems to work with [0, 1]
                sample[indx] = (data[key][1][indx].astype(np.float32) / 127.5) - 1.0
                # sample[indx] = data[key][1][indx].astype(np.float32) / 255.0
                mask[indx] = 1.0
            
            # print(file_path, len(samples))
            names.append(key)
            samples.append(sample)
            masks.append(mask)
            labels.append(label)

            # save_sample_img(key+'_o', 0, sample)
            # save_sample_img(key+'_f', 0, sample_f)
        
        del data
    # NOTE if one sample doesn't have enough frames Keras will error out here with 'assign a sequence'
    npsamples = np.array(samples, dtype=np.float32)
    npmasks = np.array(masks, dtype=np.float32)
    nplabels = np.array(labels, dtype=np.int32)

    print('file {} Shape samples {}, labels {}'.format(file_path, npsamples.shape, nplabels.shape))
    return names, npsamples, npmasks, nplabels


def save_predictions(predictions):
    with open('sample_submission.csv', 'w') as sf:
        sf.write('filename,label\n')
        for name, score in predictions:
            sf.write('%s,%1.6f\n' % (name, score))

# TODO break these out
def compile_model(model):

    optimizer = tf.keras.optimizers.Adam(lr=0.025)
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
        x = Input(shape = (256, 256, 3))
        
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


def create_model(input_shape):

    input_layer = Input(shape=input_shape)
    input_mask = Input(shape=(input_shape[0]))

    classifier = MesoInception4()
    classifier.model.load_weights('pretrained/Meso/raw/all/weights.h5')
    for layer in classifier.model.layers:
        if layer.name == 'max_pooling2d_3':
            output = layer.output
    feature_extractor = Model(inputs=classifier.model.input, outputs=output)
    for layer in feature_extractor.layers:
        layer.trainable = False

    net = TimeDistributed(feature_extractor)(input_layer)
    net = TimeDistributed(Flatten())(net)
    net = Bidirectional(GRU(256, return_sequences=True))(net, mask=input_mask)
    net = SeqWeightedAttention()(net, mask=input_mask)
    out = Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.001),
        bias_initializer=tf.keras.initializers.Constant(np.log([1.5])))(net)

    model = Model(inputs=[input_layer, input_mask], outputs=out)

    return model


if __name__ == '__main__':

    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', type=str)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--weights', type=str, default=None)
    args = parser.parse_args()

    custom_objs = {
        'fraction_positives':fraction_positives,
        'SeqWeightedAttention':SeqWeightedAttention,
    }

    if args.model is not None:
        model = tf.keras.models.load_model(args.model, custom_objects=custom_objs)
    elif args.weights is not None:
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            model = create_model((SEQ_LEN,) + FEAT_SHAPE)
            model.load_weights(args.weights)
            compile_model(model)
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
    pkl_files = glob.glob(args.pkl)
    for pkl_file in pkl_files:
        print('Predicting on samples from {}'.format(pkl_file))
        names, npsamples, npmasks, nplabels = read_file(pkl_file)
        preds = model.predict([npsamples, npmasks], verbose=1, batch_size=8)
        print(preds)
        predictions.extend(preds)
        saved.extend(zip(names, preds))
        truths.extend(nplabels)
        del npsamples, npmasks
        gc.collect()
    if len(predictions) > 0:
        print('Log loss on predictions: {}'.format(log_loss(truths, predictions, labels=[0, 1])))
        save_predictions(saved)
    else:
        print('No predictions, check input.')
    t1 = time.time()
    print("Execution took: {}".format(t1-t0))
