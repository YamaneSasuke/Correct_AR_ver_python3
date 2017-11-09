# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 18:37:28 2016

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
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, Chain, serializers
from chainer.iterators import SerialIterator
from chainer.iterators import MultiprocessIterator

import utils
import load_datasets
from links import CBR

# ネットワークの定義
class Convnet(Chain):
    def __init__(self):
        super(Convnet, self).__init__(
            cbr1=CBR(3, 16, 3, 2, 1),
            cbr2=CBR(16, 16, 3, 2, 1),
            cbr3=CBR(16, 32, 3, 2, 1,),
            cbr4=CBR(32, 32, 3, 2, 1,),
            cbr5=CBR(32, 64, 3, 2, 1,),

            l1=L.Linear(3136, 1000),
            norm6=L.BatchNormalization(1000),
            l2=L.Linear(1000, 1),
        )

    def __call__(self, X):
        h = self.cbr1(X)
        h = self.cbr2(h)
        h = self.cbr3(h)
        h = self.cbr4(h)
        h = self.cbr5(h)
        h = F.relu(self.norm6(self.l1(h)))
        y = self.l2(h)
        return y

    def lossfun(self, X, t):
        y = self(X)
        loss = F.mean_squared_error(y, t)
        return loss

    def loss_ave(self, queue, num_batches, test):
        losses = []
        for i in range(num_batches):
            X_batch, T_batch = queue.get()
            X_batch = cuda.to_gpu(X_batch)
            T_batch = cuda.to_gpu(T_batch)
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                loss = self.lossfun(X_batch, T_batch)
            losses.append(cuda.to_cpu(loss.data))
        return np.mean(losses)

    def predict(self, X, test):
        X = cuda.to_gpu(X)
        y = self(X)
        y = cuda.to_cpu(y.data)
        return y


def test_output(model, X, T, t_loss):
    predict_t = model.predict(X)
    target_t = T
    predict_r = np.exp(predict_t)
    target_r = np.exp(target_t)
    predict_image = utils.fix_image(X[0:1], predict_r[0])
    original_image = utils.fix_image(X[0:1], target_r[0])
    debased_image = np.transpose(X[0], (1, 2, 0))
    predict_image = np.transpose(predict_image, (1, 2, 0))
    original_image = np.transpose(original_image, (1, 2, 0))
    t_dis = predict_t - target_t
    t_loss.append(t_dis)

    print('predict t:', predict_t[0], 'target t:', target_t[0])
    print('predict r:', predict_r[0], 'target r:', target_r[0])

    plt.plot(t_loss[0])
    plt.title("t_disdance")
    plt.grid()
    plt.show()

    plt.subplot(131)
    plt.title("debased_image")
    plt.imshow(debased_image/256.0)
    plt.subplot(132)
    plt.title("fix_image")
    plt.imshow(predict_image/256.0)
    plt.subplot(133)
    plt.title("target_image")
    plt.imshow(original_image/256.0)
    plt.show()
    return t_loss


if __name__ == '__main__':
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    time_start = time.time()
    image_list = []
    epoch_loss = []
    epoch_valid_loss = []
    loss_valid_best = np.inf
    t_loss = []

    # 超パラメータ
    max_iteration = 100  # 繰り返し回数
    batch_size = 100  # ミニバッチサイズ
    num_train = 20000  # 学習データ数
    num_test = 100  # 検証データ数
    learning_rate = 0.001  # 学習率
    output_size = 256  # 生成画像サイズ
    crop_size = 224  # ネットワーク入力画像サイズ
    aspect_ratio_min = 1.0  # 最小アスペクト比の誤り
    aspect_ratio_max = 2.0  # 最大アスペクト比の誤り
    crop = True
    hdf5_filepath = r'E:\stanford_Dogs_Dataset\raw_dataset_binary\output_size_500\output_size_500.hdf5'  # データセットファイル保存場所
    output_location = r'C:\Users\yamane\OneDrive\M1\correct_aspect_ratio'  # 学習結果保存場所
    # 学習結果保存フォルダ作成
    output_root_dir = os.path.join(output_location, file_name)
    folder_name = str(time_start) + '_asp_max_' + str(aspect_ratio_max)
    output_root_dir = os.path.join(output_root_dir, folder_name)
    if os.path.exists(output_root_dir):
        pass
    else:
        os.makedirs(output_root_dir)
    # ファイル名を作成
    model_filename = str(file_name) + str(time_start) + '.npz'
    loss_filename = 'epoch_loss' + str(time_start) + '.png'
    t_dis_filename = 't_distance' + str(time_start) + '.png'
    model_filename = os.path.join(output_root_dir, model_filename)
    loss_filename = os.path.join(output_root_dir, loss_filename)
    t_dis_filename = os.path.join(output_root_dir, t_dis_filename)
    # バッチサイズ計算
    train_data = range(0, num_train)
    test_data = range(num_train, num_train + num_test)
    num_batches_train = num_train / batch_size
    num_batches_test = num_test / batch_size
    # stream作成
    streams = load_datasets.load_voc2012_stream(
        batch_size, num_train, num_batches_valid)
    train_stream, valid_stream, test_stream = streams
    # キューを作成、プロセススタート
    queue_train = Queue(10)
    process_train = Process(target=load_datasets.load_data,
                            args=(queue_train, train_stream, crop,
                                  aspect_ratio_max, aspect_ratio_min,
                                  output_size, crop_size))
    process_train.start()
    queue_test = Queue(10)
    process_test = Process(target=load_datasets.load_data,
                           args=(queue_test, dog_stream_test, crop,
                                 aspect_ratio_max, aspect_ratio_min,
                                 output_size, crop_size))
    process_test.start()
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
            for i in tqdm.tqdm(range(num_batches_train)):
                X_batch, T_batch = queue_train.get()
                X_batch = cuda.to_gpu(X_batch)
                T_batch = cuda.to_gpu(T_batch)
                # 勾配を初期化
                model.cleargrads()
                # 順伝播を計算し、誤差と精度を取得
                with chainer.using_config('train', True), chainer.no_backprop_mode():
                    loss = model.lossfun(X_batch, T_batch)
                    # 逆伝搬を計算
                    loss.backward()
                optimizer.update()
                losses.append(cuda.to_cpu(loss.data))

            time_end = time.time()
            epoch_time = time_end - time_begin
            total_time = time_end - time_origin
            epoch_loss.append(np.mean(losses))

            loss_valid = model.loss_ave(queue_test, num_batches_test)
            epoch_valid_loss.append(loss_valid)
            if loss_valid < loss_valid_best:
                loss_valid_best = loss_valid
                epoch__loss_best = epoch
                model_best = copy.deepcopy(model)

            # 訓練データでの結果を表示
            print("dog_data_regression.py")
            print("epoch:", epoch)
            print("time", epoch_time, "(", total_time, ")")
            print("loss[train]:", epoch_loss[epoch])
            print("loss[valid]:", loss_valid)
            print("loss[valid_best]:", loss_valid_best)
            print("epoch[valid_best]:", epoch__loss_best)

            plt.plot(epoch_loss)
            plt.plot(epoch_valid_loss)
            plt.ylim(0, 0.5)
            plt.title("loss")
            plt.legend(["train", "valid"], loc="upper right")
            plt.grid()
            plt.show()

            # テスト用のデータを取得
            X_test, T_test = queue_test.get()
            t_loss = test_output(model_best, X_test, T_test, t_loss)

    except KeyboardInterrupt:
        print("割り込み停止が実行されました")

    plt.plot(epoch_loss)
    plt.plot(epoch_valid_loss)
    plt.ylim(0, 0.5)
    plt.title("loss")
    plt.legend(["train", "valid"], loc="upper right")
    plt.grid()
    plt.savefig(loss_filename)
    plt.show()

    plt.plot(t_loss)
    plt.title("t_disdance")
    plt.grid()
    plt.savefig(t_dis_filename)
    plt.show()

    model_filename = os.path.join(output_root_dir, model_filename)
    serializers.save_npz(model_filename, model_best)

    process_train.terminate()
    process_test.terminate()
    print('max_iteration:', max_iteration)
    print('learning_rate:', learning_rate)
    print('batch_size:', batch_size)
    print('train_size', num_train)
    print('valid_size', num_test)
    print('output_size', output_size)
    print('crop_size', crop_size)
    print('aspect_ratio_min', aspect_ratio_min)
    print('aspect_ratio_max', aspect_ratio_max)
