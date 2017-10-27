# -*- coding: utf-8 -*-
"""
Created on Fri Dec 09 12:02:10 2016

@author: yamane
"""

import os
import numpy as np
import time
import tqdm
import copy
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue

import chainer
from chainer import cuda, optimizers, Chain, serializers
import chainer.functions as F
import chainer.links as L

from links import CBR
import voc2012_regression
import load_datasets


# ネットワークの定義
class Convnet(Chain):
    def __init__(self):
        super(Convnet, self).__init__(
            cbr1_1=CBR(3, 64, 3, 2, 1),
            cbr2_1=CBR(64, 128, 3, 2, 1),
            cbr3_1=CBR(128, 128, 3, 2, 1),
            cbr4_1=CBR(128, 256, 3, 1, 1),
            cbr4_2=CBR(256, 256, 3, 2, 1),
            cbr5_1=CBR(256, 512, 3, 1, 1),
            cbr5_2=CBR(512, 512, 3, 2, 1),

            l1=L.Linear(512, 1)
        )

    def __call__(self, X, test):
        h = self.cbr1_1(X)

        h = self.cbr2_1(h)

        h = self.cbr3_1(h)

        h = self.cbr4_1(h)
        h = self.cbr4_2(h)

        h = self.cbr5_1(h)
        h = self.cbr5_2(h)

        h = F.max_pooling_2d(h, 7)
        y = self.l1(h)
        return y

    def lossfun(self, X, t):
        y = self.forward(X)
        loss = F.mean_squared_error(y, t)
        return loss

    def loss_ave(self, queue, num_batches):
        losses = []
        for i in range(num_batches):
            X_batch, T_batch = queue.get()
            X_batch = cuda.to_gpu(X_batch)
            T_batch = cuda.to_gpu(T_batch)
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                loss = self.lossfun(X_batch, T_batch)
            losses.append(cuda.to_cpu(loss.data))
        return np.mean(losses)

    def predict(self, X):
        X = cuda.to_gpu(X)
        y = self.forward(X)
        y = cuda.to_cpu(y.data)
        return y


if __name__ == '__main__':
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    time_start = time.time()
    image_list = []
    epoch_loss = []
    epoch_valid_loss = []
    loss_valid_best = np.inf
    t_loss = []

    # 超パラメータ
    max_iteration = 1000  # 繰り返し回数
    batch_size = 100  # ミニバッチサイズ
    num_train = 16500  # 学習データ数
    num_valid = 500  # 検証データ数
    learning_rate = 0.01  # 学習率
    output_size = 256  # 生成画像サイズ
    crop_size = 224  # ネットワーク入力画像サイズ
    aspect_ratio_min = 1.0  # 最小アスペクト比の誤り
    aspect_ratio_max = 4.0  # 最大アスペクト比の誤り
    crop = True
    # 学習結果保存場所
    output_location = r'C:\Users\yamane\OneDrive\M1\correct_aspect_ratio'
    # 学習結果保存フォルダ作成
    output_root_dir = os.path.join(output_location, file_name)
    folder_name = str(time_start) + '_asp_max_' + str(aspect_ratio_max)
    output_root_dir = os.path.join(output_root_dir, folder_name)
    if os.path.exists(output_root_dir):
        pass
    else:
        os.makedirs(output_root_dir)
    # ファイル名を作成
    model_filename = str(file_name) + '.npz'
    loss_filename = 'epoch_loss' + str(time_start) + '.png'
    t_dis_filename = 't_distance' + str(time_start) + '.png'
    model_filename = os.path.join(output_root_dir, model_filename)
    loss_filename = os.path.join(output_root_dir, loss_filename)
    t_dis_filename = os.path.join(output_root_dir, t_dis_filename)
    # バッチサイズ計算
    num_batches_train = int(num_train / batch_size)
    num_batches_valid = int(num_valid / batch_size)
    # stream作成
    streams = load_datasets.load_voc2012_stream(
        batch_size, num_train, num_batches_valid)
    train_stream, valid_stream, test_stream = streams
    # キューを作成、プロセススタート
    queue_train = Queue(10)
    process_train = Process(target=load_datasets.load_data,
                            args=(queue_train, train_stream, crop,
                                  aspect_ratio_max, output_size, crop_size))
    process_train.start()
    queue_valid = Queue(10)
    process_valid = Process(target=load_datasets.load_data,
                            args=(queue_valid, valid_stream, crop,
                                  aspect_ratio_max, output_size, crop_size))
    process_valid.start()
    # モデル読み込み
    model = Convnet().to_gpu()
    # Optimizerの設定
    optimizer = optimizers.Adam(learning_rate)
    optimizer.setup(model)

    time_origin = time.time()
    try:
        for epoch in range(max_iteration):
            time_begin = time.time()
            losses = []
            accuracies = []
            for i in tqdm.tqdm(range(num_batches_train)):
                X_batch, T_batch = queue_train.get()
                X_batch = cuda.to_gpu(X_batch)
                T_batch = cuda.to_gpu(T_batch)
                # 勾配を初期化
                model.cleargrads()
                with chainer.using_config('train', True), chainer.no_backprop_mode():
                    # 順伝播を計算し、誤差と精度を取得
                    loss = model.lossfun(X_batch, T_batch)
                    # 逆伝搬を計算
                    loss.backward()
                optimizer.update()
                losses.append(cuda.to_cpu(loss.data))

            time_end = time.time()
            epoch_time = time_end - time_begin
            total_time = time_end - time_origin
            epoch_loss.append(np.mean(losses))

            loss_valid = model.loss_ave(queue_valid, num_batches_valid)
            epoch_valid_loss.append(loss_valid)
            if loss_valid < loss_valid_best:
                loss_valid_best = loss_valid
                epoch__loss_best = epoch
                model_best = copy.deepcopy(model)

            # 訓練データでの結果を表示
            print()
            print("dog_data_regression_ave_pooling.py")
            print("epoch:", epoch)
            print("time", epoch_time, "(", total_time, ")")
            print("loss[train]:", epoch_loss[epoch])
            print("loss[valid]:", loss_valid)
            print("loss[valid_best]:", loss_valid_best)
            print("epoch[valid_best]:", epoch__loss_best)

            if (epoch % 10) == 0:
                plt.figure(figsize=(16, 12))
                plt.plot(epoch_loss)
                plt.plot(epoch_valid_loss)
                plt.ylim(0, 0.5)
                plt.title("loss")
                plt.legend(["train", "valid"], loc="upper right")
                plt.grid()
                plt.show()

            # 検証用のデータを取得
            X_valid, T_valid = queue_valid.get()
            t_loss = voc2012_regression.test_output(model_best, X_valid,
                                                    T_valid, t_loss)

    except KeyboardInterrupt:
        print("割り込み停止が実行されました")

    plt.figure(figsize=(16, 12))
    plt.plot(epoch_loss)
    plt.plot(epoch_valid_loss)
    plt.ylim(0, 0.5)
    plt.title("loss")
    plt.legend(["train", "valid"], loc="upper right")
    plt.grid()
    plt.savefig(loss_filename)
    plt.show()

    model_filename = os.path.join(output_root_dir, model_filename)
    serializers.save_npz(model_filename, model_best)

    process_train.terminate()
    process_valid.terminate()
    print('max_iteration:', max_iteration)
    print('learning_rate:', learning_rate)
    print('batch_size:', batch_size)
    print('train_size', num_train)
    print('valid_size', num_valid)
    print('output_size', output_size)
    print('crop_size', crop_size)
    print('aspect_ratio_min', aspect_ratio_min)
    print('aspect_ratio_max', aspect_ratio_max)
