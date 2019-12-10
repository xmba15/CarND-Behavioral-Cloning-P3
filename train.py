#!/usr/bin/env python
# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from config import Config
from data_loader import BehavioralDataset
from models import BehavioralModel
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def load_data(dataset):
    x_train, y_train = dataset.load_data_with_bias()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    return x_train, x_val, y_train, y_val


def plot_history(history):
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    if "acc" in history.history:
        train_acc = history.history["acc"]
        val_acc = history.history["val_acc"]
    epoch_nums = len(train_loss)

    if "acc" in history.history:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
        loss_ax = ax[0]
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        loss_ax = ax

    loss_ax.set_xlabel("epochs")
    loss_ax.set_ylabel("loss")
    loss_ax.set_title("loss")
    loss_ax.plot(range(0, epoch_nums), train_loss, label="train loss")
    loss_ax.plot(range(0, epoch_nums), val_loss, label="val loss")
    loss_ax.legend()

    if "acc" in history.history:
        ax[1].set_xlabel("epochs")
        ax[1].set_ylabel("val")
        ax[1].set_title("accuracy")
        ax[1].plot(range(0, epoch_nums), train_acc, label="train acc")
        ax[1].plot(range(0, epoch_nums), val_acc, label="val acc")
        ax[1].legend()

    plt.suptitle("train-val logs")
    # plt.show()
    plt.savefig("train_val_logs.png")


def main():
    dt_config = Config()
    dataset = BehavioralDataset(path_to_data=dt_config.DATA_PATH)
    x_train, x_val, y_train, y_val = load_data(dataset)

    model = BehavioralModel(input_shape=(160, 320, 3))
    callbacks = [
        ModelCheckpoint(
            dt_config.SAVED_MODELS, monitor="val_loss", verbose=1, save_best_only=True, mode="auto", period=1
        ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10, verbose=0, mode="auto", min_lr=0.00001),
        EarlyStopping(monitor="val_loss", min_delta=0, patience=15, verbose=0, mode="auto"),
    ]

    history = model.fit_generator(
        batch_generator(x_train, y_train, dt_config.BATCH_SIZE),
        epochs=dt_config.EPOCHS,
        steps_per_epoch=(20000 - 1) // dt_config.BATCH_SIZE + 1,
        validation_data=batch_generator(x_val, y_val, dt_config.BATCH_SIZE, False),
        validation_steps=(len(x_val) - 1) // dt_config.BATCH_SIZE + 1,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    main()
