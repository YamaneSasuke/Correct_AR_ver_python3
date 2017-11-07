# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 15:33:32 2017

@author: yamane
"""

import os
import numpy as np
import time
import tqdm
import copy
import matplotlib.pyplot as plt

import cv2

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, serializers
from chainer.iterators import MultiprocessIterator

import bias_sum_pooling, conv_pooling, ave_pooling, max_pooling
from load_datasets import TestDataset
import utils

def visualize(model, model_name, t=0.0):
    test_data = TestDataset(1, 1700, 17200)

    batch = test_data.get_example(t)
    X = batch[0]
    T = batch[1]
    X = cuda.to_gpu(X)
    T = cuda.to_gpu(T)
    # 勾配を初期化
    model.cleargrads()
    with chainer.using_config('train', False):
        # 順伝播を計算し、誤差と精度を取得
        # 逆伝搬を計算
        h1 = model.conv(X)
        if model_name == 'bias_sum_pooling':
            h2 = model.bias_sum_pooling(h1)
        elif model_name == 'conv_pooling':
            h2 = model.conv_pooling(h1)
        elif model_name == 'ave_pooling':
            h2 = F.average_pooling_2d(h1, 7)
        elif model_name == 'max_pooling':
            h2 = F.max_pooling_2d(h1)
        h3 = model.l1(h2)
        h3.backward(retain_grad=True)
        w = F.average_pooling_2d(h1.grad, 7)
        w = F.broadcast_to(w, h1.shape)
        grad_cam = F.relu(F.sum(w * h1.data, axis=1))
        img = cuda.to_cpu(X)
        grad_cam = cuda.to_cpu(grad_cam.data)
        grad_cam = cv2.resize(grad_cam[0], (224, 224))
        img = np.transpose(img, (0, 2, 3, 1))
        plt.imshow(img[0]/ 255.0)
        plt.imshow(grad_cam, cmap=plt.cm.jet, alpha=0.4)
        plt.show()
        print('target', T)
        print('predict', h3.data)

if __name__ == '__main__':
    model_file = r'C:\Users\yamane\OneDrive\M1\correct_aspect_ratio\bias_sum_pooling\1509613391.5179036\bias_sum_pooling.npz'
    model_name = model_file.split('\\')[-3]
    # モデル読み込み
    model = bias_sum_pooling.Bias_sum_pooling().to_gpu()
    # Optimizerの設定
    serializers.load_npz(model_file, model)
    t = 1/1.5
    visualize(model, model_name, t=t)
