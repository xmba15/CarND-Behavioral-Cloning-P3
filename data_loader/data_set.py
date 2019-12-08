#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import csv
import cv2


class BehavioralDataset(object):
    def __init__(self, path_to_data):
        if not os.path.isdir(path_to_data):
            raise RuntimeError("Path to data does not exist!")
        if not os.path.isfile(os.path.join(path_to_data, "driving_log.csv")):
            raise RuntimeError("Please provide driving log file")
        if not os.path.isdir(os.path.join(path_to_data, "IMG")):
            raise RuntimeError("Please provide frame for training")

        with open(os.path.join(path_to_data, "driving_log.csv")) as csvfile:
            reader = csv.reader(csvfile)
            self.lines = [line for line in reader]

        print("loading data ...")
        self.images = [cv2.imread(os.path.join(path_to_data, line[0])) for line in self.lines]
        self.measurements = [float(line[3]) for line in self.lines]
