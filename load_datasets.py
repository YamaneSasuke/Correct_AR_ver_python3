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

import chainer
from chainer.iterators import MultiprocessIterator

import fuel
from fuel.datasets.hdf5 import H5PYDataset

import utils

class Dataset(chainer.dataset.DatasetMixin):
    def __init__(self, batch_size, start=0, end=17100, aspectratio_max=4.0, train=True):
        self.start=start
        self.end=end
        self.aspect_ratio_max=aspectratio_max
        self.output_size=256
        self.crop_size=224
        self.num_data=end-start
        self.num_batch=self.num_data/batch_size
        permu=np.random.permutation(self.num_data)
        self.indexes=np.array_split(permu, self.num_batch)
        self.i=0
        self.finish=False
        self.train=train

        fuel_root = fuel.config.data_path[0]
        # データセットファイル保存場所
        hdf5_filepath=os.path.join(
                fuel_root, 'voc2012\hdf5_dataset\hdf5_dataset.hdf5')
        h5py_file=h5py.File(hdf5_filepath)
        self.dataset=H5PYDataset(h5py_file, ['train'])

    def __len__(self):
        return (self.end - self.start)

    def get_example(self, i):
        self.finish = False
        handle = self.dataset.open()
        data = self.dataset.get_data(handle, list(self.indexes[self.i]))
        self.dataset.close(handle)
        X, T = self.create_distorted_img(data[0], self.aspect_ratio_max,
                                         self.output_size, self.crop_size)
        self.i = self.i + 1
        if self.i == self.num_batch:
            self.i = 0
            permu=np.random.permutation(self.num_data)
            self.indexes=np.array_split(permu, self.num_batch)
            self.finish = True
        return X, T, self.finish

    def create_distorted_img(self, X_batch, aspect_ratio_max=3.0, output_size=256,
                             crop_size=224):
        images = []
        ts = []

        for b in range(X_batch.shape[0]):
            # 補間方法を乱数で設定
            u = np.random.randint(5)
            image = X_batch[b]
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
            if self.train is True:
                crop_image = utils.random_crop_and_flip(resize_image, crop_size)
            else:
                crop_image = utils.crop_224(resize_image)
            th = np.random.rand()
            if th < 0.33333:
                crop_image = utils.draw_horizontal_line(crop_image)
            elif th > 0.66666:
                crop_image = utils.draw_vertical_line(crop_image)
            else:
                crop_image = crop_image
            images.append(crop_image)
            ts.append(t)
        X = np.stack(images, axis=0)
        X = np.transpose(X, (0, 3, 1, 2))
        X = X.astype(np.float32)
        T = np.array(ts, dtype=np.float32).reshape(-1, 1)
        return X, T

class TestDataset(chainer.dataset.DatasetMixin):
    def __init__(self, batch_size, start=0, end=17200):
        self.start=start
        self.end=end
        self.output_size=224
        self.num_data=end-start
        self.num_batch=self.num_data/batch_size
        permu=np.random.permutation(self.num_data)
        self.indexes=np.array_split(permu, self.num_batch)
        self.i=0
        self.finish=False

        fuel_root = fuel.config.data_path[0]
        # データセットファイル保存場所
        hdf5_filepath=os.path.join(
                fuel_root, 'voc2012\hdf5_dataset\hdf5_dataset.hdf5')
        h5py_file=h5py.File(hdf5_filepath)
        self.dataset=H5PYDataset(h5py_file, ['train'])

    def __len__(self):
        return (self.end - self.start)

    def get_example(self, t):
        self.finish = False
        handle = self.dataset.open()
        data = self.dataset.get_data(handle, list(self.indexes[self.i]))
        self.dataset.close(handle)
        X, T = self.create_distorted_img(data[0], t, self.output_size)
        self.i = self.i + 1
        if self.i == self.num_batch:
            self.i = 0
            permu=np.random.permutation(self.num_data)
            self.indexes=np.array_split(permu, self.num_batch)
            self.finish = True
        return X, T, self.finish

    def create_distorted_img(self, X_batch, t, output_size=224):
        images = []
        ts = []

        for b in range(X_batch.shape[0]):
            # 補間方法を乱数で設定
            u = np.random.randint(5)
            image = X_batch[b]
            t = t
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
            images.append(resize_image)
            ts.append(t)
        X = np.stack(images, axis=0)
        X = np.transpose(X, (0, 3, 1, 2))
        X = X.astype(np.float32)
        T = np.array(ts, dtype=np.float32).reshape(-1, 1)
        return X, T

if __name__ == '__main__':
    __spec__ = None
    batch_size = 100
    start = 0
    end = 1000

    train = Dataset(batch_size, start, end)

    ite = MultiprocessIterator(train, 1, n_processes=1)
    n=0
    i = 0
    for a in range(10):
        print('a', a)
        while True:
            batch = next(ite)
            x = batch[0][0]
            t = batch[0][1]
            finish = batch[0][2]
            print(n)
            print(finish)
            print('---------------------------------------------------------------')
            print(x.shape)
            print(t.shape)
            plt.imshow(np.transpose(x[0], (1,2,0))/255.0)
            plt.show()
            print()
            n = n+1
            if finish is True:
                break
    ite.finalize()