#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
from tensorflow.keras.regularizers import l1


def BehavioralModel(input_shape=(160, 320, 3), opt=Adam(0.0001, 0.5)):
    model_input = Input(shape=input_shape)
    x = Cropping2D(cropping=((50, 20), (0, 0)))(model_input)

    x = Lambda(lambda x: (x / 255.0))(model_input)

    x = Conv2D(24, (5, 5), strides=(2, 2), kernel_initializer="he_normal")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(36, (5, 5), strides=(2, 2), kernel_initializer="he_normal")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(48, (5, 5), strides=(2, 2), kernel_initializer="he_normal")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = SpatialDropout2D(0.5)(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer="he_normal")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer="he_normal")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Flatten()(x)

    x = Dense(100, kernel_initializer="he_normal", activity_regularizer=l1(0.001))(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.5)(x)

    x = Dense(50, kernel_initializer="he_normal", activity_regularizer=l1(0.001))(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)

    x = Dense(10, kernel_initializer="he_normal", activity_regularizer=l1(0.001))(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)

    model_output = Dense(1, kernel_initializer="he_normal", activity_regularizer=l1(0.001))(x)

    model = Model(model_input, model_output)
    model.compile(loss="mse", optimizer=opt)
    model.summary()

    return model
