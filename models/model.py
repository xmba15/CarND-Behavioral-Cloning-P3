#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Reshape,
    Dense,
    Dropout,
    SpatialDropout2D,
    Flatten,
    Activation,
    LeakyReLU,
    ReLU,
    ELU,
    Conv2D,
    UpSampling2D,
    Conv2DTranspose,
    BatchNormalization,
    ZeroPadding2D,
    Cropping2D,
    Lambda,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def resize_image(x):
    return tf.image.resize(x, (66, 200), method=tf.image.ResizeMethod.BICUBIC)


def hsv_conversion(x):
    return tf.image.rgb_to_hsv(x)


def BehavioralModel(input_shape=(160, 320, 3), opt=Adam(0.0001)):
    model_input = Input(shape=input_shape)
    x = Cropping2D(cropping=((70, 25), (0, 0)))(model_input)
    x = Lambda(lambda x: (x / 255 - 0.5))(x)

    x = Conv2D(24, (5, 5), strides=(2, 2), kernel_initializer="he_normal")(x)
    x = ELU()(x)

    x = Conv2D(36, (5, 5), strides=(2, 2), kernel_initializer="he_normal")(x)
    x = ELU()(x)

    x = Conv2D(48, (5, 5), strides=(2, 2), kernel_initializer="he_normal")(x)
    x = ELU()(x)

    x = Conv2D(64, (3, 3), kernel_initializer="he_normal")(x)
    x = ELU()(x)

    x = Conv2D(64, (3, 3), kernel_initializer="he_normal")(x)
    x = ELU()(x)

    x = Dropout(0.5)(x)
    x = Flatten()(x)

    x = Dense(100, kernel_initializer="he_normal")(x)
    x = ELU()(x)

    x = Dense(50, kernel_initializer="he_normal")(x)
    x = ELU()(x)

    x = Dense(10, kernel_initializer="he_normal")(x)
    x = ELU()(x)

    model_output = Dense(1, kernel_initializer="he_normal")(x)

    model = Model(model_input, model_output)
    model.compile(loss="mse", optimizer=opt)
    model.summary()

    return model
