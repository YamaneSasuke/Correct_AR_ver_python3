# -*- coding: utf-8 -*-
"""
Created on Fri Dec 09 12:02:10 2016

@author: yamane
"""

import os

import chainer.functions as F
import chainer.links as L
from chainer import optimizers, Chain

from links import ARConvnet, CBR
from train import trainer

# ネットワークの定義
class ConvPooling(Chain):
    def __init__(self):
        super(ConvPooling, self).__init__(
            conv=ARConvnet(),
            create_w_1=CBR(512, 300, 1),
            create_w_2=CBR(300, 100, 1),
            create_w_3=L.Convolution2D(100, 1, 1),
            bn=L.BatchNormalization(1),
            l1=L.Linear(512, 1)
        )

    def __call__(self, X):
        h = self.conv(X)
        h = self.conv_pooling(h)
        y = self.l1(h)
        return y

    def conv_pooling(self, x):
        w = self.create_w_1(x)
        w = self.create_w_2(w)
        w = F.tanh(self.bn(self.create_w_3(w)))
        w = F.broadcast_to(w, x.shape)
        weighted_x = x * w
        pooled_x = F.sum(weighted_x, axis=(2, 3))
        return F.relu(pooled_x)

    def lossfun(self, X, t):
        y = self(X)
        loss = F.mean_squared_error(y, t)
        return loss

if __name__ == '__main__':
    __spec__ = None
    file_name = os.path.splitext(os.path.basename(__file__))[0]

    # 超パラメータ
    max_iteration = 100000  # 繰り返し回数
    batch_size = 100  # ミニバッチサイズ
    num_train = 16500  # 学習データ数
    num_valid = 500  # 検証データ数
    learning_rate = 0.001  # 学習率 test loss順位1
    aspect_ratio_max = 4.0  # 最大アスペクト比の誤り
    # 学習結果保存場所
    output_location = r'C:\Users\yamane\OneDrive\M1\correct_aspect_ratio'
    # モデル読み込み
    model = ConvPooling().to_gpu()

    # Optimizerの設定
    optimizer = optimizers.Adam(learning_rate)
    optimizer.setup(model)

    params = [max_iteration, batch_size, num_train, num_valid, learning_rate,
              aspect_ratio_max, output_location]
    # モデルの学習
    train_losses, valid_losses, best_model = trainer(
            file_name, model, optimizer, params)
