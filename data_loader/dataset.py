#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import csv
import cv2
import numpy as np
from tqdm import tqdm

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
try:
    sys.path.append(os.path.join(_CURRENT_DIR, ".."))
    from utility import remove_near_duplicate_images
except:
    print("Cannot load utility module")
    exit


class BehavioralDataset(object):
    def __init__(self, path_to_data, left_offset=0.2, right_offset=-0.2):
        if not os.path.isdir(path_to_data):
            raise RuntimeError("Path to data does not exist!")
        if not os.path.isfile(os.path.join(path_to_data, "driving_log.csv")):
            raise RuntimeError("Please provide driving log file")
        if not os.path.isdir(os.path.join(path_to_data, "IMG")):
            raise RuntimeError("Please provide frame for training")

        with open(os.path.join(path_to_data, "driving_log.csv")) as csvfile:
            reader = csv.reader(csvfile)
            self.__lines = [line for line in reader]

        self.__center_image_paths = [os.path.join(path_to_data, line[0]) for line in self.__lines]
        self.__left_image_paths = [os.path.join(path_to_data, line[1]) for line in self.__lines]
        self.__right_image_paths = [os.path.join(path_to_data, line[2]) for line in self.__lines]
        self.__measurements = [float(line[3]) for line in self.__lines]

        self.left_offset = left_offset
        self.right_offset = right_offset

    def __len__(self):
        return len(self.__lines)

    def __getitem__(self, idx):
        center_image_path = self.__center_image_paths[idx]
        left_image_path = self.__left_image_paths[idx]
        right_image_path = self.__right_image_paths[idx]
        measurement = self.__measurements[idx]

        return cv2.imread(center_image_path), cv2.imread(left_image_path), cv2.imread(right_image_path), measurement

    def load_data(self):
        print("loading data ...")
        center_images = []
        left_images = []
        right_images = []
        measurements = []

        for idx in tqdm(range(self.__len__())):
            center_image, left_image, right_image, measurement = self.__getitem__(idx)
            center_images.append(center_image)
            left_images.append(left_image)
            right_images.append(right_image)
            measurements.append(measurement)

        return np.asarray(center_images), np.asarray(left_images), np.asarray(right_images), np.asarray(measurements)

    def load_data_with_bias(self):
        center_images, left_images, right_images, measurements = self.load_data()

        add_idx = np.where((measurements > self.left_offset) | (measurements < self.right_offset))
        x_train = center_images[add_idx]
        y_train = measurements[add_idx]

        process_idx = np.where((self.right_offset <= measurements) & (measurements <= self.left_offset))
        keep_indices, remove_indices = remove_near_duplicate_images(center_images[process_idx])
        remove_indices_length = len(remove_indices)

        x_train = np.concatenate((x_train, center_images[process_idx][keep_indices]))
        y_train = np.append(y_train, measurements[process_idx][keep_indices])

        np.random.seed(15)
        left_idx = np.random.choice(range(remove_indices_length), remove_indices_length // 2, replace=False)
        right_idx = np.array([i for i in range(remove_indices_length) if i not in left_idx])

        x_train = np.concatenate((x_train, left_images[process_idx][remove_indices][left_idx]))
        y_train = np.append(y_train, measurements[process_idx][remove_indices][left_idx] + self.left_offset)

        x_train = np.concatenate((x_train, right_images[process_idx][remove_indices][right_idx]))
        y_train = np.append(y_train, measurements[process_idx][remove_indices][right_idx] + self.right_offset)

        return x_train, y_train
