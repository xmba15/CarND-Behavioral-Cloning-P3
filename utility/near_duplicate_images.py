#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import imagehash
from tqdm import tqdm
from PIL import Image
from time import time


def hamming_distance(str1, str2):
    diffs = 0
    for ch1, ch2 in zip(str1, str2):
        if ch1 != ch2:
            diffs += 1
    return diffs


def remove_near_duplicate_images(images, hash_func=imagehash.dhash, threshold=7):
    """
    Parameters
    ----------
    images: list of images

    Returns
    -------
    tuple
        indices to keep, indices to remove
    """
    image_dict = {}
    for i, image in tqdm(enumerate(images)):
        try:
            hash = str(hash_func(Image.fromarray(image)))
            if hash not in image_dict:
                remove_flag = False
                for ihash in image_dict.keys():
                    if hamming_distance(ihash, hash) < threshold:
                        remove_flag = True
                        break
                if not remove_flag:
                    image_dict[hash] = i
        except Exception as e:
            print("Problem:", e, "with image num ", i)

    keep_indices = np.array(list(image_dict.values()))
    remove_indices = np.array([i for i in range(len(images)) if i not in keep_indices])

    return keep_indices, remove_indices
