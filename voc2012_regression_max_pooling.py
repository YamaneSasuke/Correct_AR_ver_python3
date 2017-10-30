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

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, Chain, serializers
from chainer.iterators import MultiprocessIterator

from links import CBR
import voc2012_regression
from load_datasets import Dataset


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

            l1=L.Linear(512, 1),
            norm1=L.BatchNormalization(512),
            l2=L.Linear(512, 1)
        )

    def __call__(self, X):
        h = self.cbr1_1(X)

        h = self.cbr2_1(h)

        h = self.cbr3_1(h)

        h = self.cbr4_1(h)
        h = self.cbr4_2(h)

        h = self.cbr5_1(h)
        h = self.cbr5_2(h)

        h = F.relu(self.norm1(self.l1(h)))
        y = self.l2(h)
        return y

    def lossfun(self, X, t):
        y = self(X)
        loss = F.mean_squared_error(y, t)
        return loss

    def loss_ave(self, iterator):
        losses = []
        while True:
            batch = next(iterator)
            X_batch = batch[0][0]
            T_batch = batch[0][1]
            finish = batch[0][2]
            X_batch = cuda.to_gpu(X_batch)
            T_batch = cuda.to_gpu(T_batch)
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                loss = self.lossfun(X_batch, T_batch)
            losses.append(cuda.to_cpu(loss.data))
            if finish is True:
                break
        return np.mean(losses)

    def predict(self, X):
        X = cuda.to_gpu(X)
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            y = self(X)
        y = cuda.to_cpu(y.data)
        return y

# ネットワークの定義
class Convnet_max(Chain):
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

    def __call__(self, X):
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
        y = self(X)
        loss = F.mean_squared_error(y, t)
        return loss

    def loss_ave(self, iterator):
        losses = []
        while True:
            batch = next(iterator)
            X_batch = batch[0][0]
            T_batch = batch[0][1]
            finish = batch[0][2]
            X_batch = cuda.to_gpu(X_batch)
            T_batch = cuda.to_gpu(T_batch)
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                loss = self.lossfun(X_batch, T_batch)
            losses.append(cuda.to_cpu(loss.data))
            if finish is True:
                break
        return np.mean(losses)

# ネットワークの定義
class Convnet_ave(Chain):
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

    def __call__(self, X):
        h = self.cbr1_1(X)

        h = self.cbr2_1(h)

        h = self.cbr3_1(h)

        h = self.cbr4_1(h)
        h = self.cbr4_2(h)

        h = self.cbr5_1(h)
        h = self.cbr5_2(h)

        h = F.average_pooling_2d(h, 7)
        y = self.l1(h)
        return y

    def lossfun(self, X, t):
        y = self(X)
        loss = F.mean_squared_error(y, t)
        return loss

    def loss_ave(self, iterator):
        losses = []
        while True:
            batch = next(iterator)
            X_batch = batch[0][0]
            T_batch = batch[0][1]
            finish = batch[0][2]
            X_batch = cuda.to_gpu(X_batch)
            T_batch = cuda.to_gpu(T_batch)
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                loss = self.lossfun(X_batch, T_batch)
            losses.append(cuda.to_cpu(loss.data))
            if finish is True:
                break
        return np.mean(losses)


if __name__ == '__main__':
    __spec__ = None
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    time_start = time.time()
    epoch_loss1 = []
    epoch_loss2 = []
    epoch_loss3 = []
    epoch_loss4 = []
    epoch_loss5 = []
    epoch_valid_loss1 = []
    epoch_valid_loss2 = []
    epoch_valid_loss3 = []
    epoch_valid_loss4 = []
    epoch_valid_loss5 = []
    loss_valid_best1 = np.inf
    loss_valid_best2 = np.inf
    loss_valid_best3 = np.inf
    loss_valid_best4 = np.inf
    loss_valid_best5 = np.inf
#    t_loss = []

    # 超パラメータ
    max_iteration = 100000  # 繰り返し回数
    batch_size = 100  # ミニバッチサイズ
    num_train = 16500  # 学習データ数
    num_valid = 500  # 検証データ数
    learning_rate = 0.001  # 学習率 test loss順位1
    aspect_ratio_max = 4.0  # 最大アスペクト比の誤り
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
    model_filename1 = str(file_name) + '_1' + '.npz'
    model_filename2 = str(file_name) + '_2' + '.npz'
    model_filename3 = str(file_name) + '_3' + '.npz'
    loss_filename = 'epoch_loss' + str(time_start) + '.png'
#    t_dis_filename = 't_distance' + str(time_start) + '.png'
    model_filename1 = os.path.join(output_root_dir, model_filename1)
    model_filename2 = os.path.join(output_root_dir, model_filename2)
    model_filename3 = os.path.join(output_root_dir, model_filename3)
    loss_filename = os.path.join(output_root_dir, loss_filename)
#    t_dis_filename = os.path.join(output_root_dir, t_dis_filename)
    # バッチサイズ計算
    num_batches_train = int(num_train / batch_size)
    num_batches_valid = int(num_valid / batch_size)
    # stream作成
    train_data = Dataset(batch_size, 0, 16500, train=True)
    valid_data = Dataset(batch_size, 16500, 17000, train=False)
    test_data = Dataset(batch_size, 1700, 17200, train=False)
    train_ite = MultiprocessIterator(train_data, 1, n_processes=1)
    valid_ite = MultiprocessIterator(valid_data, 1, n_processes=1)
    test_ite = MultiprocessIterator(test_data, 1, n_processes=1)
    # モデル読み込み
    model1 = Convnet().to_gpu()
    model2 = Convnet_max().to_gpu()
    model3 = Convnet_ave().to_gpu()
    # Optimizerの設定
    optimizer1 = optimizers.Adam(learning_rate)
    optimizer1.setup(model1)
    optimizer2 = optimizers.Adam(learning_rate)
    optimizer2.setup(model2)
    optimizer3 = optimizers.Adam(learning_rate)
    optimizer3.setup(model3)

    time_origin = time.time()
    try:
        for epoch in range(max_iteration):
            time_begin = time.time()
            losses1 = []
            losses2 = []
            losses3 = []
            losses4 = []
            losses5 = []
            for i in tqdm.tqdm(range(num_batches_train)):
                batch = next(train_ite)
                X_batch = batch[0][0]
                T_batch = batch[0][1]
                finish = batch[0][2]
                X_batch = cuda.to_gpu(X_batch)
                T_batch = cuda.to_gpu(T_batch)
                # 勾配を初期化
                model1.cleargrads()
                model2.cleargrads()
                model3.cleargrads()
                with chainer.using_config('train', True):
                    # 順伝播を計算し、誤差と精度を取得
                    loss1 = model1.lossfun(X_batch, T_batch)
                    loss2 = model2.lossfun(X_batch, T_batch)
                    loss3 = model3.lossfun(X_batch, T_batch)
                    # 逆伝搬を計算
                    loss1.backward()
                    loss2.backward()
                    loss3.backward()
                optimizer1.update()
                optimizer2.update()
                optimizer3.update()
                losses1.append(cuda.to_cpu(loss1.data))
                losses2.append(cuda.to_cpu(loss2.data))
                losses3.append(cuda.to_cpu(loss3.data))
                if finish is True:
                    break

            time_end = time.time()
            epoch_time = time_end - time_begin
            total_time = time_end - time_origin
            epoch_loss1.append(np.mean(losses1))
            epoch_loss2.append(np.mean(losses2))
            epoch_loss3.append(np.mean(losses3))

            loss_valid1 = model1.loss_ave(valid_ite)
            loss_valid2 = model2.loss_ave(valid_ite)
            loss_valid3 = model3.loss_ave(valid_ite)
            epoch_valid_loss1.append(loss_valid1)
            epoch_valid_loss2.append(loss_valid2)
            epoch_valid_loss3.append(loss_valid3)

            if loss_valid1 < loss_valid_best1:
                loss_valid_best1 = loss_valid1
                epoch__loss_best1 = epoch
                model_best1 = copy.deepcopy(model1)

            if loss_valid2 < loss_valid_best2:
                loss_valid_best2 = loss_valid2
                epoch__loss_best2 = epoch
                model_best2 = copy.deepcopy(model2)

            if loss_valid3 < loss_valid_best3:
                loss_valid_best3 = loss_valid3
                epoch__loss_best3 = epoch
                model_best3 = copy.deepcopy(model3)


            # 訓練データでの結果を表示
            print()
            print("voc2012_regression_max_pooling.py")
            print("epoch:", epoch+1)
            print("time", epoch_time, "(", total_time, ")")
            print("loss1[train]:", epoch_loss1[epoch])
            print("loss2[train]:", epoch_loss2[epoch])
            print("loss3[train]:", epoch_loss3[epoch])
            print("loss1[valid]:", loss_valid1)
            print("loss2[valid]:", loss_valid2)
            print("loss3[valid]:", loss_valid3)
#            print("loss[valid_best]:", loss_valid_best)
#            print("epoch[valid_best]:", epoch__loss_best)

#            if (epoch % 10) == 0:
            plt.plot(epoch_loss1)
            plt.plot(epoch_loss2)
            plt.plot(epoch_loss3)
            plt.plot(epoch_valid_loss1)
            plt.plot(epoch_valid_loss2)
            plt.plot(epoch_valid_loss3)
            plt.ylim(0, 0.5)
            plt.title("loss")
            plt.legend(["train1", "train2", "train3", "valid1", "valid2", "valid3"], bbox_to_anchor=(1.25, 1), loc="upper right")
            plt.grid()
            plt.show()

            # 検証用のデータを取得
#            test_batch = next(test_ite)
#            X_valid = test_batch[0][0]
#            T_valid = test_batch[0][1]
#            t_loss = voc2012_regression.test_output(model_best, X_valid,
#                                                    T_valid, t_loss)

    except KeyboardInterrupt:
        print("割り込み停止が実行されました")

    train_ite.finalize()
    valid_ite.finalize()
    test_ite.finalize()

    plt.plot(epoch_loss1)
    plt.plot(epoch_loss2)
    plt.plot(epoch_loss3)
    plt.plot(epoch_valid_loss1)
    plt.plot(epoch_valid_loss2)
    plt.plot(epoch_valid_loss3)
    plt.ylim(0, 0.5)
    plt.title("loss")
    plt.legend(["train1", "train2", "train3", "valid1", "valid2", "valid3"], bbox_to_anchor=(1.25, 1), loc="upper right")
    plt.grid()
    plt.savefig(loss_filename)
    plt.show()

    serializers.save_npz(model_filename1, model_best1)

    serializers.save_npz(model_filename2, model_best2)

    serializers.save_npz(model_filename3, model_best3)

    print('max_iteration:', max_iteration)
    print('batch_size:', batch_size)
    print('train_size', num_train)
    print('valid_size', num_valid)
    print('aspect_ratio_max', aspect_ratio_max)
    print('learning_rate:', learning_rate)
    print("loss_valid_best1:", loss_valid_best1)
    print("loss_valid_best2:", loss_valid_best2)
    print("loss_valid_best3:", loss_valid_best3)
