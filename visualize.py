# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 15:33:32 2017

@author: yamane
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import cv2

import chainer
import chainer.functions as F
from chainer import cuda, serializers
from chainer.links import VGG16Layers

from chainercv.datasets import voc_detection_label_names
from chainercv.links import SSD300, FasterRCNNVGG16
from chainercv.visualizations import vis_bbox

import bias_sum_pooling, conv_pooling, ave_pooling, max_pooling
from load_datasets import TestDataset

def grad_cam_asp(model, model_name, output_root_dir, t=0.0):
    test_data = TestDataset(1, 17000, 17100)

    for i in range(200):
        batch = test_data.get_example(t)
        X = batch[0]
        T = batch[1]
        img_name = os.path.join(output_root_dir, str(i))
        X = cuda.to_gpu(X)
        T = cuda.to_gpu(T)
        grad = np.ndarray((1,1))
        grad[0] = 1.0
        grad = cuda.to_gpu(grad.astype('f'))
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
            y = model.l1(h2)
            y.grad = grad
            y.backward(retain_grad=True)
            w = F.average_pooling_2d(h1.grad, 7)
            w = F.broadcast_to(w, h1.shape)
            grad_cam = F.relu(F.sum(w * h1.data, axis=1))
            img = cuda.to_cpu(X)
            grad_cam = cuda.to_cpu(grad_cam.data)
            grad_cam = cv2.resize(grad_cam[0], (224, 224))
            img = np.transpose(img, (0, 2, 3, 1))
            plt.imshow(img[0]/ 255.0)
            plt.imshow(grad_cam, cmap=plt.cm.jet, alpha=0.4)
            plt.tick_params(labelbottom='off', labeltop='off', labelleft='off',
                            labelright='off')
            plt.tick_params(bottom='off', top='off', left='off', right='off')
            plt.savefig(img_name+'.jpg', format='jpg', bbox_inches='tight')
            plt.show()
            print('target', T)
            print('predict', y.data)

def grad_cam_vgg(img_root, label_num):
    img = plt.imread(img_root)
    img = cv2.resize(img, (224, 224))
    img = np.transpose(img,(2,0,1))
    img = img.astype('f')

    model_name = 'VGG'
    file_name = 'test'
    # 結果保存フォルダ作成
    output_root_dir = os.path.join(output_location, model_name)
    output_root_dir = os.path.join(output_root_dir, file_name)
    img_name = os.path.join(output_root_dir, str(label_num))
    if os.path.exists(output_root_dir):
        pass
    else:
        os.makedirs(output_root_dir)

    X = img.reshape(-1, img.shape[0], img.shape[1], img.shape[2])
    T = np.ndarray((1,))
    T[0] = label_num
    grad = onehot(T)
    grad = grad.astype('f')
    grad = cuda.to_gpu(grad)
    X = cuda.to_gpu(X)

    model = VGG16Layers().to_gpu()
    # 勾配を初期化
    model.cleargrads()
    with chainer.using_config('train', False):
        # 順伝播を計算し、誤差と精度を取得
        # 逆伝搬を計算
        feature = model.extract([img], layers=["conv5_3", "prob"])
        h = feature["conv5_3"]
        y = feature["prob"]
        y.grad = grad
        y.backward(retain_grad=True)
    w = F.average_pooling_2d(h.grad, h.data.shape[-1])
    w = F.broadcast_to(w, h.shape)
    grad_cam = F.relu(F.sum(w * h.data, axis=1))
    img = cuda.to_cpu(X)
    grad_cam = cuda.to_cpu(grad_cam.data)
    grad_cam = cv2.resize(grad_cam[0], (img.shape[-1], img.shape[-1]))
    img = np.transpose(img, (0, 2, 3, 1))
    plt.imshow(img[0]/ 255.0)
    plt.imshow(grad_cam, cmap=plt.cm.jet, alpha=0.4)
    plt.tick_params(labelbottom='off', labeltop='off', labelleft='off',
                    labelright='off')
    plt.tick_params(bottom='off', top='off', left='off', right='off')
    plt.savefig(img_name+'.jpg', format='jpg', bbox_inches='tight')
    plt.show()
    list_ = list(y.data[0])
    print('target', T)
    print('predict', list_.index(max(y.data[0])))

def onehot(k, num_classes=1000):
    t_onehot = np.zeros((len(k), num_classes))
    for i, k_i in enumerate(k):
        t_onehot[i][int(k_i)] = 1
    return t_onehot

if __name__ == '__main__':
    output_location = r'C:\Users\yamane\OneDrive\M1\correct_aspect_ratio\visualize'

    # ARestimatorの場合
    t = 1
    model_file = r'C:\Users\yamane\OneDrive\M1\correct_aspect_ratio\bias_sum_pooling\1510231098.3499577\bias_sum_pooling.npz'
    model_name = model_file.split('\\')[-3]
    file_name = model_file.split('\\')[-2]
    # 結果保存フォルダ作成
    output_root_dir = os.path.join(output_location, model_name)
    output_root_dir = os.path.join(output_root_dir, file_name)
    output_root_dir = os.path.join(output_root_dir, str(t))
    if os.path.exists(output_root_dir):
        pass
    else:
        os.makedirs(output_root_dir)    # モデル読み込み
    model = bias_sum_pooling.BiasSumPooling().to_gpu()
    # Optimizerの設定
    serializers.load_npz(model_file, model)
    grad_cam_asp(model, model_name, output_location, t=t)

    # VGG16netの場合
#    img_root = r'C:\Users\yamane\OneDrive\M1\correct_aspect_ratio\images\dog_cat.jpg'
#    label_num = 283
#    grad_cam_vgg(img_root, label_num)
