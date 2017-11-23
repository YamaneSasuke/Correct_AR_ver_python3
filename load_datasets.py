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
    def __init__(self, batch_size, start=0, end=17100, aspectratio_max=4.0,
                 output_size=256, crop_size=224, train=True):
        self.start=start
        self.end=end
        self.ar_max=aspectratio_max
        self.output_size=output_size
        self.crop_size=crop_size
        self.num_data=end-start
        self.num_batch=self.num_data/batch_size
        if train is True:
            self.permu=np.random.permutation(self.num_data) + start
        elif train is False:
            self.permu=np.array(range(self.num_data)) + start
        self.indexes=np.array_split(self.permu, self.num_batch)
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
        x, t = self.create_distorted_img(data[0])
        self.i = self.i + 1
        if self.i == self.num_batch:
            self.i = 0
            if self.train is True:
                self.permu=np.random.permutation(self.num_data)
            elif self.train is False:
                self.permu=np.array(range(self.num_data))
            self.indexes=np.array_split(self.permu, self.num_batch)
            self.finish = True
        return x, t, self.finish

    def create_distorted_img(self, x_batch):
        img_list = []
        t_list = []

        for b in range(x_batch.shape[0]):
            # 補間方法を乱数で設定
            u = np.random.randint(5)
            img = x_batch[b]
            t = utils.sample_random_aspect_ratio(np.log(self.ar_max),
                                                 -np.log(self.ar_max))
            r = np.exp(t)
            # 歪み画像生成
            dis_img = utils.change_aspect_ratio(img, r, u)
            # 中心切り抜き
            square_img = utils.crop_center(dis_img)
            if u == 0:
                resize_img = cv2.resize(square_img,
                                        (self.output_size, self.output_size),
                                        interpolation=cv2.INTER_NEAREST)
            elif u == 1:
                resize_img = cv2.resize(square_img,
                                        (self.output_size, self.output_size),
                                        interpolation=cv2.INTER_LINEAR)
            elif u == 2:
                resize_img = cv2.resize(square_img,
                                        (self.output_size, self.output_size),
                                        interpolation=cv2.INTER_AREA)
            elif u == 3:
                resize_img = cv2.resize(square_img,
                                        (self.output_size, self.output_size),
                                        interpolation=cv2.INTER_CUBIC)
            elif u == 4:
                resize_img = cv2.resize(square_img,
                                        (self.output_size, self.output_size),
                                        interpolation=cv2.INTER_LANCZOS4)
            if self.train is True:
                crop_img = utils.random_crop_and_flip(resize_img, self.crop_size)
            else:
                crop_img = utils.crop_224(resize_img)
#            th = np.random.rand()
#            if th < (1/3):
#                crop_img = utils.draw_horizontal_line(crop_img)
#            elif th > (2/3):
#                crop_img = utils.draw_vertical_line(crop_img)
#            else:
#                crop_img = crop_img
            img_list.append(crop_img)
            t_list.append(t)
        x = np.stack(img_list, axis=0)
        x = np.transpose(x, (0, 3, 1, 2))
        x = x.astype(np.float32)
        t = np.array(t_list, dtype=np.float32).reshape(-1, 1)
        return x, t

class TestDataset(chainer.dataset.DatasetMixin):
    def __init__(self, batch_size, start=0, end=17100, output_size=224):
        self.start=start
        self.end=end
        self.output_size=output_size
        self.num_data=end-start
        self.num_batch=self.num_data/batch_size
        permu=np.array(range(self.num_data)) + start
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
        x, t = self.create_distorted_img(data[0], t)
        self.i = self.i + 1
        if self.i == self.num_batch:
            self.i = 0
            self.finish = True
        return x, t, self.finish

    def create_distorted_img(self, x_batch, t):
        img_list = []
        t_list = []

        for b in range(x_batch.shape[0]):
            # 補間方法を乱数で設定
            u = 1
            img = x_batch[b]
            t = t
            r = np.exp(t)
            # 歪み画像生成
            img = utils.change_aspect_ratio(img, r, u)
            # 中心切り抜き
            square_img = utils.crop_center(img)
            if u == 0:
                resize_img = cv2.resize(square_img,
                                        (self.output_size, self.output_size),
                                        interpolation=cv2.INTER_NEAREST)
            elif u == 1:
                resize_img = cv2.resize(square_img,
                                        (self.output_size, self.output_size),
                                        interpolation=cv2.INTER_LINEAR)
            elif u == 2:
                resize_img = cv2.resize(square_img,
                                        (self.output_size, self.output_size),
                                        interpolation=cv2.INTER_AREA)
            elif u == 3:
                resize_img = cv2.resize(square_img,
                                        (self.output_size, self.output_size),
                                        interpolation=cv2.INTER_CUBIC)
            elif u == 4:
                resize_img = cv2.resize(square_img,
                                        (self.output_size, self.output_size),
                                        interpolation=cv2.INTER_LANCZOS4)
#            th = np.random.rand()
#            if th < (1/3):
#                resize_img = utils.draw_horizontal_line(resize_img)
#            elif th > (2/3):
#                resize_img = utils.draw_vertical_line(resize_img)
#            else:
#                resize_img = resize_img
            img_list.append(resize_img)
            t_list.append(t)
        x = np.stack(img_list, axis=0)
        x = np.transpose(x, (0, 3, 1, 2))
        x = x.astype(np.float32)
        t = np.array(t_list, dtype=np.float32).reshape(-1, 1)
        return x, t

if __name__ == '__main__':
    __spec__ = None
    batch_size = 1
    start = 17000
    end = 17100
    t = 0

    train = Dataset(batch_size, start, end, train=False)

    ite = MultiprocessIterator(train, 1, n_processes=1)

    while True:
        batch = next(ite)
        x = batch[0][0]
        t = batch[0][1]
        finish = batch[0][2]
        print(finish)
        print('---------------------------------------------------------------')
        print(x.shape)
        print(t.shape)
        img = np.transpose(x[0], (1,2,0))/255.0
        plt.imshow(img)
        plt.show()
        print()
        if finish is True:
            break
    ite.finalize()

#    train = TestDataset(batch_size, start, end)
#
#    while True:
#        batch = train.get_example(t)
#        x = batch[0]
#        t = batch[1]
#        finish = batch[2]
#        print(finish)
#        print('---------------------------------------------------------------')
#        print(x.shape)
#        print(t.shape)
#        img = np.transpose(x[0], (1,2,0))/255.0
#        plt.imshow(img)
#        plt.show()
#        print()
#        if finish is True:
#            break
