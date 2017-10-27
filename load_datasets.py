# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:06:51 2016

@author: yamane
"""

import os
import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue

import fuel
from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.datasets import IterableDataset

import utils


def load_voc2012_stream(batch_size, train_size=16500, validation_size=500,
                        test_size=100, shuffle=False):
    fuel_root = fuel.config.data_path[0]
    # データセットファイル保存場所
    hdf5_filepath = os.path.join(
        fuel_root, 'voc2012\hdf5_dataset\hdf5_dataset.hdf5')
    valid_size = train_size + validation_size
    test_size = valid_size + test_size
    indices_train = range(0, train_size)
    indices_valid = range(train_size, valid_size)
    indices_test = range(valid_size, test_size)

    h5py_file = h5py.File(hdf5_filepath)
    dataset = H5PYDataset(h5py_file, ['train'])

    scheme_class = ShuffledScheme if shuffle else SequentialScheme
    scheme_train = scheme_class(indices_train, batch_size=batch_size)
    scheme_valid = scheme_class(indices_valid, batch_size=batch_size)
    scheme_test = scheme_class(indices_test, batch_size=batch_size)

    stream_train = DataStream(dataset, iteration_scheme=scheme_train)
    stream_valid = DataStream(dataset, iteration_scheme=scheme_valid)
    stream_test = DataStream(dataset, iteration_scheme=scheme_test)
    next(stream_train.get_epoch_iterator())
    next(stream_valid.get_epoch_iterator())
    next(stream_test.get_epoch_iterator())

    return stream_train, stream_valid, stream_test


def data_crop(X_batch, aspect_ratio_max=3.0, output_size=256, crop_size=224,
              random=True, t=0):
    images = []
    ts = []

    for b in range(X_batch.shape[0]):
        # 補間方法を乱数で設定
        u = np.random.randint(5)
        image = X_batch[b]
        if random is False:
            t = t
        else:
            t = utils.sample_random_aspect_ratio(np.log(aspect_ratio_max),
                                                   -np.log(aspect_ratio_max))
        r = np.exp(t)
        # 歪み画像生成
        image = utils.change_aspect_ratio(image, r, u)
        # 中心切り抜き
        square_image = utils.crop_center(image)
        if u == 0:
            resize_image = cv2.resize(square_image, (output_size, output_size),
                                      interpolation=cv2.INTER_NEAREST)
        elif u == 1:
            resize_image = cv2.resize(square_image, (output_size, output_size),
                                      interpolation=cv2.INTER_LINEAR)
        elif u == 2:
            resize_image = cv2.resize(square_image, (output_size, output_size),
                                      interpolation=cv2.INTER_AREA)
        elif u == 3:
            resize_image = cv2.resize(square_image, (output_size, output_size),
                                      interpolation=cv2.INTER_CUBIC)
        elif u == 4:
            resize_image = cv2.resize(square_image, (output_size, output_size),
                                      interpolation=cv2.INTER_LANCZOS4)
        if random is False:
            crop_image = utils.crop_224(resize_image)
        else:
            crop_image = utils.random_crop_and_flip(resize_image, crop_size)
        images.append(crop_image)
        ts.append(t)
    X = np.stack(images, axis=0)
    X = np.transpose(X, (0, 3, 1, 2))
    X = X.astype(np.float32)
    T = np.array(ts, dtype=np.float32).reshape(-1, 1)
    return X, T


def data_padding(X_batch, aspect_ratio_max=3.0, output_size=256, crop_size=224,
                 random=True, t=0):
    images = []
    ts = []
    for b in range(X_batch.shape[0]):
        # 補間方法を乱数で設定
        u = np.random.randint(5)
        image = X_batch[b]
        if random is False:
            t = t
        else:
            t = utils.sample_random_aspect_ratio(np.log(aspect_ratio_max),
                                                   -np.log(aspect_ratio_max))
        r = np.exp(t)
        image = utils.change_aspect_ratio(image, r, u)
        square_image = utils.crop_center(image)
        if u == 0:
            resize_image = cv2.resize(square_image, (output_size, output_size),
                                      interpolation=cv2.INTER_NEAREST)
        elif u == 1:
            resize_image = cv2.resize(square_image, (output_size, output_size),
                                      interpolation=cv2.INTER_LINEAR)
        elif u == 2:
            resize_image = cv2.resize(square_image, (output_size, output_size),
                                      interpolation=cv2.INTER_AREA)
        elif u == 3:
            resize_image = cv2.resize(square_image, (output_size, output_size),
                                      interpolation=cv2.INTER_CUBIC)
        elif u == 4:
            resize_image = cv2.resize(square_image, (output_size, output_size),
                                      interpolation=cv2.INTER_LANCZOS4)
        resize_image = resize_image[..., None]
        if random is False:
            crop_image = utils.crop_224(resize_image)
        else:
            crop_image = utils.random_crop_and_flip(resize_image, crop_size)
        images.append(crop_image)
        ts.append(t)
    X = np.stack(images, axis=0)
    X = np.transpose(X, (0, 3, 1, 2))
    X = X.astype(np.float32)
    T = np.array(ts, dtype=np.float32).reshape(-1, 1)
    return X, T


def load_data(queue, stream, crop, aspect_ratio_max=3.0, output_size=256,
              crop_size=224, random=True, t=0):
    while True:
        for X in stream.get_epoch_iterator():
            if crop is True:
                X, T = data_crop(X[0], aspect_ratio_max, output_size,
                                 crop_size, random, t)
            else:
                X, T = data_padding(X[0], aspect_ratio_max, crop_size,
                                    random, t)
            queue.put((X, T))
