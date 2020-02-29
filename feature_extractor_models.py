import tensorflow as tf
import keras_utils

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
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self.init_model()
    
    def init_model(self):
        x = Input(shape=self.input_shape)
        
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
    def __init__(self, width, input_shape):
        self.width = width
        self.input_shape = input_shape
        self.model = self.init_model()
        
    def init_model(self):
        x = Input(shape=self.input_shape)
        
        x1 = InceptionLayer(2*self.width, 2*self.width, 2*self.width, 2*self.width)(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
        x2 = InceptionLayer(4*self.width, 4*self.width, 4*self.width, 4*self.width)(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)        
        
        x3 = InceptionLayer(4*self.width, 4*self.width, 4*self.width, 4*self.width)(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)        

        x4 = Conv2D(32*self.width, (5, 5), padding='same', activation = 'relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(2, 2), padding='same')(x4)        

        x5 = Conv2D(32*self.width, (5, 5), padding='same', activation = 'relu')(x4)
        x5 = BatchNormalization()(x5)
        x5 = MaxPooling2D(pool_size=(4, 4), padding='same')(x5)
        
        # x6 = InceptionLayer(16*self.width, 16*self.width, 16*self.width, 16*self.width)(x5)
        # x6 = BatchNormalization()(x6)
        # x6 = MaxPooling2D(pool_size=(2, 2), padding='same')(x6)

        # x7 = InceptionLayer(64*self.width, 64*self.width, 64*self.width, 64*self.width)(x6)
        # x7 = BatchNormalization()(x7)
        # x7 = GlobalAveragePooling2D()(x6)
        # x7 = MaxPooling2D(pool_size=(2, 2), padding='same')(x7)

        
        y = Flatten()(x5)
        y = Dropout(0.5)(y)
        #TODO investigate num units for this dense layer.
        # y = Dense(32*self.width)(y)
        # y = LeakyReLU(alpha=0.1)(y)
        # y = Dropout(0.5)(y)
        y = Dense(2, activation = 'softmax', kernel_regularizer=tf.keras.regularizers.l2(0.02))(y)

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


def resnet_18(input_shape, num_filters=64):
    X_input = tf.keras.layers.Input(shape=input_shape)
    X = tf.keras.layers.Conv2D(filters=num_filters,
                               kernel_size=(7, 7),
                               strides=2,
                               padding='same')(X_input)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.ELU()(X)
    X = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2,
                                  padding='same')(X)

    X = residual_unit(X, filter_num=num_filters, stride_num=1)
    X = residual_unit(X, filter_num=num_filters, stride_num=1)

    X = residual_unit(X, filter_num=2*num_filters, stride_num=2)
    X = residual_unit(X, filter_num=2*num_filters, stride_num=1)

    X = residual_unit(X, filter_num=4*num_filters, stride_num=2)
    X = residual_unit(X, filter_num=4*num_filters, stride_num=1)

    X = residual_unit(X, filter_num=8*num_filters, stride_num=2)
    X = residual_unit(X, filter_num=8*num_filters, stride_num=1)

    X = tf.keras.layers.GlobalAveragePooling2D()(X)
    X = tf.keras.layers.Flatten()(X)
    # X = tf.keras.layers.Dense(units=16*num_filters, activation=tf.nn.elu)(X)
    X = tf.keras.layers.Dropout(0.5)(X)
    X = tf.keras.layers.Dense(units=1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01))(X)

    model = tf.keras.models.Model(inputs=X_input, outputs=X)
    return model


class OneMIL():

    def __init__(self, input_shape, width=1):
        self.input_shape = input_shape
        self.width = width
        self.mil_input_shape = (input_shape[0] // 2, input_shape[1] // 2, input_shape[2])
        self.mil_input_height = self.mil_input_shape[0]
        self.mil_input_width = self.mil_input_shape[1]
        # self.full_model = self.create_full_model(input_shape)
        self.create_mil_models()
        self.model = self.create_model()

    def mil_backbone(self, input_shape):
        # num_filters = 4
        # X_input = tf.keras.layers.Input(shape=input_shape)
        # X = tf.keras.layers.Conv2D(filters=num_filters,
        #                         kernel_size=(7, 7),
        #                         strides=2,
        #                         padding='same')(X_input)
        # X = tf.keras.layers.BatchNormalization()(X)
        # X = tf.keras.layers.ELU()(X)
        # X = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2,
        #                             padding='same')(X)

        # X = residual_unit(X, filter_num=num_filters, stride_num=1)
        # X = residual_unit(X, filter_num=num_filters, stride_num=1)

        # X = residual_unit(X, filter_num=2*num_filters, stride_num=2)
        # X = residual_unit(X, filter_num=2*num_filters, stride_num=1)

        # X = residual_unit(X, filter_num=4*num_filters, stride_num=2)
        # X = residual_unit(X, filter_num=4*num_filters, stride_num=1)

        # X = residual_unit(X, filter_num=8*num_filters, stride_num=2)
        # X = residual_unit(X, filter_num=8*num_filters, stride_num=1)

        # X = tf.keras.layers.GlobalAveragePooling2D()(X)
        # y = tf.keras.layers.Flatten()(X)

        x = Input(shape=input_shape)
        
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

        return Model(inputs=x, outputs=y)

    def create_full_model(self, input_shape):
        num_filters = 4
        X_input = tf.keras.layers.Input(shape=input_shape)
        X = tf.keras.layers.Conv2D(filters=num_filters,
                                kernel_size=(7, 7),
                                strides=2,
                                padding='same')(X_input)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.ELU()(X)
        X = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2,
                                    padding='same')(X)

        X = residual_unit(X, filter_num=num_filters, stride_num=1)
        X = residual_unit(X, filter_num=num_filters, stride_num=1)

        X = residual_unit(X, filter_num=2*num_filters, stride_num=2)
        X = residual_unit(X, filter_num=2*num_filters, stride_num=1)

        X = residual_unit(X, filter_num=4*num_filters, stride_num=2)
        X = residual_unit(X, filter_num=4*num_filters, stride_num=1)

        X = residual_unit(X, filter_num=8*num_filters, stride_num=2)
        X = residual_unit(X, filter_num=8*num_filters, stride_num=1)

        X = tf.keras.layers.GlobalAveragePooling2D()(X)
        y = tf.keras.layers.Flatten()(X)

        # x = Input(shape=input_shape)
        
        # x1 = InceptionLayer(2*self.width, 2*self.width, 2*self.width, 2*self.width)(x)
        # x1 = BatchNormalization()(x1)
        # x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
        # x2 = InceptionLayer(4*self.width, 4*self.width, 4*self.width, 4*self.width)(x1)
        # x2 = BatchNormalization()(x2)
        # x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)        
        
        # x3 = InceptionLayer(4*self.width, 4*self.width, 4*self.width, 4*self.width)(x2)
        # x3 = BatchNormalization()(x3)
        # x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)        

        # x4 = InceptionLayer(8*self.width, 8*self.width, 8*self.width, 8*self.width)(x3)
        # x4 = BatchNormalization()(x4)
        # x4 = MaxPooling2D(pool_size=(2, 2), padding='same')(x4)        

        # x5 = InceptionLayer(8*self.width, 8*self.width, 8*self.width, 8*self.width)(x4)
        # x5 = BatchNormalization()(x5)
        # x5 = MaxPooling2D(pool_size=(2, 2), padding='same')(x5)
        
        # x6 = InceptionLayer(8*self.width, 8*self.width, 8*self.width, 8*self.width)(x5)
        # x6 = BatchNormalization()(x6)
        # x6 = MaxPooling2D(pool_size=(2, 2), padding='same')(x6)
        # # x6 = Conv2D(64*self.width, (1, 1), padding='same', activation = 'relu')(x5)
        # # x6 = BatchNormalization()(x6)
        # # x6 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x6)
        # # x6 = GlobalAveragePooling2D()(x5)
        
        # y = Flatten()(x6)
        # # y = Dropout(0.5)(y)
        # #TODO investigate num units for this dense layer.
        # # y = Dense(32*self.width)(y)
        # # y = LeakyReLU(alpha=0.1)(y)
        # # y = Dropout(0.5)(y)
        # # y = Dense(1, activation = 'sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.002))(y)

        return Model(inputs=X_input, outputs=y)

    def create_mil_models(self):
        self.left_up_model = self.mil_backbone(self.mil_input_shape)
        self.right_up_model = self.left_up_model
        self.left_down_model = self.mil_backbone(self.mil_input_shape)
        self.right_down_model = self.left_down_model
        self.center_model = self.mil_backbone(self.mil_input_shape)

    def create_model(self):

        x_input = Input(shape=self.input_shape)

        left_up_input = tf.image.crop_to_bounding_box(
            x_input,
            offset_height=0,
            offset_width=0,
            target_height=self.mil_input_height,
            target_width=self.mil_input_width
        )
        right_up_input = tf.image.crop_to_bounding_box(
            x_input,
            offset_height=0,
            offset_width=self.mil_input_width,
            target_height=self.mil_input_height,
            target_width=self.mil_input_width
        )
        left_down_input = tf.image.crop_to_bounding_box(
            x_input,
            offset_height=self.mil_input_height,
            offset_width=0,
            target_height=self.mil_input_height,
            target_width=self.mil_input_width
        )
        right_down_input = tf.image.crop_to_bounding_box(
            x_input,
            offset_height=self.mil_input_height,
            offset_width=self.mil_input_width,
            target_height=self.mil_input_height,
            target_width=self.mil_input_width
        )
        center_input = tf.image.crop_to_bounding_box(
            x_input,
            offset_height=self.mil_input_height // 2,
            offset_width=self.mil_input_width // 2,
            target_height=self.mil_input_height,
            target_width=self.mil_input_width
        )

        left_up_out = self.left_up_model(left_up_input)
        right_up_out = self.right_up_model(right_up_input)
        left_down_out = self.left_down_model(left_down_input)
        right_down_out = self.right_down_model(right_down_input)
        center_out = self.center_model(center_input)
        # full_out = self.full_model(x_input)

        all_mil_outs = tf.stack([left_up_out, right_up_out, left_down_out, right_down_out, center_out], axis=1)
        # out = keras_utils.SeqSelfAttention(attention_type='multiplicative', attention_activation='sigmoid')(all_outs)
        mil_out = keras_utils.SeqWeightedAttention()(all_mil_outs)
        mil_out = Dropout(0.5)(mil_out)
        mil_out = tf.keras.layers.Dense(units=1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.05))(mil_out)

        # full_out = Dropout(0.5)(full_out)
        # full_out = tf.keras.layers.Dense(units=1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01))(full_out)
        
        # out = 0.5 * mil_out + 0.5 * full_out

        return Model(inputs=x_input, outputs=mil_out)