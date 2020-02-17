import tensorflow

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Bidirectional, BatchNormalization, Concatenate, Conv2D, Dense, Dropout
from tensorflow.keras.layers import Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Input, GRU, LeakyReLU, LSTM, Masking, MaxPooling2D, multiply, Reshape, TimeDistributed


def InceptionLayer(a, b, c, d):
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


class MesoInception4():
    def __init__(self, learning_rate = 0.001):
        self.model = self.init_model()
    
    def init_model(self):
        x = Input(shape = (256, 256, 3))
        
        x1 = InceptionLayer(1, 4, 4, 2)(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
        x2 = InceptionLayer(2, 4, 4, 2)(x1)
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
        
    def init_model(self):
        x = Input(shape = (224, 224, 3))
        
        x1 = InceptionLayer(2*self.width, 2*self.width, 2*self.width, 2*self.width)(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
        x2 = InceptionLayer(4*self.width, 4*self.width, 4*self.width, 4*self.width)(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)        
        
        x3 = InceptionLayer(4*self.width, 4*self.width, 4*self.width, 4*self.width)(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)        

        x4 = InceptionLayer(8*self.width, 8*self.width, 8*self.width, 8*self.width)(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(2, 2), padding='same')(x4)        

        x5 = InceptionLayer(8*self.width, 8*self.width, 8*self.width, 8*self.width)(x4)
        x5 = BatchNormalization()(x5)
        x5 = MaxPooling2D(pool_size=(2, 2), padding='same')(x5)
        
        # x6 = Conv2D(64*self.width, (1, 1), padding='same', activation = 'relu')(x5)
        # x6 = BatchNormalization()(x6)
        # x6 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x6)
        # x6 = GlobalAveragePooling2D()(x5)
        
        y = Flatten()(x5)
        y = Dropout(0.5)(y)
        #TODO investigate num units for this dense layer.
        # y = Dense(32*self.width)(y)
        # y = LeakyReLU(alpha=0.1)(y)
        # y = Dropout(0.5)(y)
        y = Dense(1, activation = 'sigmoid', kernel_regularizer=tensorflow.keras.regularizers.l2(0.005))(y)

        return Model(inputs = x, outputs = y)


def residual_unit(X, filter_num, stride_num):
    X_shortcut = X

    # Main path.
    # First component.
    X = tf.keras.layers.Conv2D(filters=filter_num,
                               kernel_size=(3, 3),
                               strides=stride_num,
                               padding='same')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.ELU()(X)
    # Second component.
    X = tf.keras.layers.Conv2D(filters=filter_num,
                               kernel_size=(3, 3),
                               strides=1,
                               padding='same')(X)
    X = tf.keras.layers.BatchNormalization()(X)

    # Shortcut Path.
    X_shortcut = tf.keras.layers.Conv2D(filters=filter_num,
                                        kernel_size=(1, 1),
                                        strides=stride_num,
                                        padding='same')(X_shortcut)
    X_shortcut = tf.keras.layers.BatchNormalization()(X_shortcut)

    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.ELU()(X)

    return X


def resnet_18(input_shape, label_num):
    X_input = tf.keras.layers.Input(shape=input_shape)
    X = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=(7, 7),
                               strides=2,
                               padding='same')(X_input)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.ELU()(X)
    X = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2,
                                  padding='same')(X)

    X = residual_unit(X, filter_num=64, stride_num=1)
    X = residual_unit(X, filter_num=64, stride_num=1)

    X = residual_unit(X, filter_num=128, stride_num=2)
    X = residual_unit(X, filter_num=128, stride_num=1)

    X = residual_unit(X, filter_num=256, stride_num=2)
    X = residual_unit(X, filter_num=256, stride_num=1)

    X = residual_unit(X, filter_num=512, stride_num=2)
    X = residual_unit(X, filter_num=512, stride_num=1)

    X = tf.keras.layers.GlobalAveragePooling2D()(X)
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(units=1000, activation=tf.nn.elu)(X)
    X = tf.keras.layers.Dense(units=label_num, activation=tf.nn.softmax)(X)

    model = tf.keras.models.Model(inputs=X_input, outputs=X)
    return model

