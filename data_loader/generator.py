#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
from albumentations import (
    Compose,
    GaussianBlur,
    CLAHE,
    RandomBrightness,
    RandomContrast,
    RandomSnow,
    RandomRain,
    RandomFog,
    RandomSunFlare,
    RandomShadow,
    CoarseDropout,
    Equalize,
)
from sklearn.utils import shuffle


def random_flip(image, steering_angle):
    if np.random.rand() < 0.5:
        return cv2.flip(image, 1), -steering_angle

    return image, steering_angle


def random_translate(image, steering_angle, range_x=100, range_y=100):
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    result_steering_angle = steering_angle + trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    h, w = image.shape[:2]
    result_image = cv2.warpAffine(image, trans_m, (w, h))

    return result_image, result_steering_angle


def augment_data(image, steering_angle):
    h, w = image.shape[:2]

    result_image = np.copy(image)
    result_steering_angle = steering_angle

    result_image, result_steering_angle = random_flip(result_image, result_steering_angle)
    # result_image, result_steering_angle = random_translate(result_image, result_steering_angle)
    aug = Compose(
        [
            # GaussianBlur(blur_limit=5, p=0.5),
            RandomBrightness(limit=0.1, p=0.5),
            RandomContrast(limit=0.1, p=0.5),
            # CLAHE(p=0.5),
            # Equalize(p=0.5),
            # RandomSnow(p=0.5),
            # RandomRain(p=0.5),
            # RandomFog(p=0.5),
            # RandomSunFlare(p=0.5),
            RandomShadow(p=0.5),
            CoarseDropout(
                max_holes=4,
                max_height=h // 30,
                max_width=w // 30,
                min_holes=1,
                min_height=h // 40,
                min_width=w // 40,
                p=0.5,
            ),
        ],
        p=1,
    )
    result_image = aug(image=result_image)["image"]

    return result_image, result_steering_angle


STEERING_OFFSET = [0, 0.4, -0.4]


def choose_image(triple, steering_angle):
    no = np.random.randint(3)
    return triple[no], steering_angle + STEERING_OFFSET[no]


def batch_generator(triples, steering_angles, batch_size, is_training=True):
    total = len(triples)
    h, w, c = triples.shape[2:]

    result_images = np.empty([batch_size, h, w, c])
    result_steer_angles = np.empty(batch_size)
    while True:
        i = 0
        for idx in np.random.permutation(total):
            triple = triples[idx]
            steering_angle = steering_angles[idx]

            if is_training and np.random.rand() < 0.6:
                image, steering_angle = choose_image(triple, steering_angle)
                image, steering_angle = augment_data(image, steering_angle)
            else:
                image = triple[0]

            result_images[i] = image
            result_steer_angles[i] = steering_angle

            i += 1
            if i == batch_size:
                break

        yield result_images, result_steer_angles
