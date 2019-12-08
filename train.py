#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from config import Config
from data_loader import BehavioralDataset
from models import BehavioralModel
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def load_data(dataset):
    # x_train, y_train = dataset.load_data_with_bias()
    x_train, _, _, y_train = dataset.load_data()
    y_train.reshape(-1, 1)
    # x_train, y_train, x_val, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    return x_train, y_train


def main():
    dt_config = Config()
    dataset = BehavioralDataset(path_to_data=dt_config.DATA_PATH)
    x_train, y_train = load_data(dataset)

    model = BehavioralModel(input_shape=(160, 320, 3))
    callbacks = [
        ModelCheckpoint(dt_config.SAVED_MODELS, monitor='val_loss',
                        verbose=1, save_best_only=True, mode='auto', period=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0,
                          mode='auto', min_lr=0.00001),
        EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=0, mode='auto')
    ]

    history = model.fit(x_train, y_train, epochs=dt_config.EPOCHS, batch_size=dt_config.BATCH_SIZE,
                        validation_split=0.1, shuffle=True, callbacks=callbacks)


if __name__ == "__main__":
    main()
