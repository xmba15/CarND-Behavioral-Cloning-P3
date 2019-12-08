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
    Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def BehavioralModel(input_shape, opt=Adam(0.001, 0.5)):
    model_input = Input(shape=input_shape)
    x = Cropping2D(cropping=((50, 20), (0, 0)))(model_input)
    x = Lambda(lambda x: (x / 255.0) - 0.5)(x)

    x = Conv2D(36, (5, 5), activation="relu", name="conv_2", strides=(2, 2))(x)
    x = Conv2D(48, (5, 5), activation="relu", name="conv_3", strides=(2, 2))(x)
    # x = SpatialDropout2D(0.5)(x)

    x = Conv2D(64, (3, 3), activation="relu", name="conv_4", strides=(1, 1))(x)
    x = Conv2D(64, (3, 3), activation="relu", name="conv_5", strides=(1, 1))(x)

    x = Flatten()(x)

    x = Dense(1164)(x)
    x = Dropout(0.5)(x)
    x = Dense(100, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation="relu")(x)
    x = Dropout(0.5)(x)
    model_output = Dense(1)(x)
    model = Model(model_input, model_output)

    model.compile(loss="mse", optimizer=opt, metrics=["acc"])
    model.summary()

    return model
