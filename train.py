# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 15:20:25 2017

@author: yamane
"""

import os
import numpy as np
import time
import tqdm
import copy
import matplotlib.pyplot as plt

import chainer
from chainer import cuda, serializers
from chainer.iterators import MultiprocessIterator

from load_datasets import Dataset

def loss_ave(model, iterator):
        losses = []
        while True:
            batch = next(iterator)
            X_batch = batch[0][0]
            T_batch = batch[0][1]
            finish = batch[0][2]
            X_batch = cuda.to_gpu(X_batch)
            T_batch = cuda.to_gpu(T_batch)
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                loss = model.lossfun(X_batch, T_batch)
            losses.append(cuda.to_cpu(loss.data))
            if finish is True:
                break
        return np.mean(losses)

def trainer(file_name, model, optimizer, params):
    """
    file_name:
        type = String
    model:
         chainer.Chain class. input = (B, C, H, W). output = (B,). have a lossfun().
    optimizer:
        chainer.optimizers.optimizer(learning_rate).setup(model)
    params:
        type = list[max_iteration, batch_size, num_train, num_valid, learning_rate, aspect_ratio_max, output_location]
    """
    time_start = time.time()
    train_losses = []
    valid_losses = []
    best_epoch = 0
    best_valid_loss = np.inf
    max_iteration = params[0]  # 繰り返し回
    batch_size = params[1]  # ミニバッチサイズ
    num_train = params[2]  # 学習データ数
    num_valid = params[3]  # 検証データ数
    learning_rate = params[4]  # 学習率 test loss順位1
    aspect_ratio_max = params[5]  # 最大アスペクト比の誤り
    output_location = params[6]  # 学習結果保存場所
    train_index = num_train
    valid_index = num_train + num_valid

    # 学習結果保存フォルダ作成
    output_root_dir = os.path.join(output_location, file_name)
    folder_name = str(time_start)
    output_root_dir = os.path.join(output_root_dir, folder_name)
    if os.path.exists(output_root_dir):
        pass
    else:
        os.makedirs(output_root_dir)

    # ファイル名を作成
    model_filename = str(file_name) + '.npz'
    loss_filename = 'epoch_loss' + str(time_start) + '.png'
    model_filename = os.path.join(output_root_dir, model_filename)
    loss_filename = os.path.join(output_root_dir, loss_filename)

    # バッチサイズ計算
    num_batches_train = int(num_train / batch_size)
    # stream作成
    train_data = Dataset(batch_size, 0, train_index, aspect_ratio_max, train=True)
    valid_data = Dataset(batch_size, train_index, valid_index, aspect_ratio_max, train=False)
    test_data = Dataset(batch_size, valid_index, 17100, aspect_ratio_max, train=False)
    train_ite = MultiprocessIterator(train_data, 1, n_processes=1)
    valid_ite = MultiprocessIterator(valid_data, 1, n_processes=1)
    test_ite = MultiprocessIterator(test_data, 1, n_processes=1)

    time_origin = time.time()
    try:
        for epoch in range(max_iteration):
            time_begin = time.time()
            losses = []
            for i in tqdm.tqdm(range(num_batches_train)):
                batch = next(train_ite)
                X_batch = batch[0][0]
                T_batch = batch[0][1]
                finish = batch[0][2]
                X_batch = cuda.to_gpu(X_batch)
                T_batch = cuda.to_gpu(T_batch)
                # 勾配を初期化
                model.cleargrads()
                with chainer.using_config('train', True):
                    # 順伝播を計算し、誤差と精度を取得
                    loss = model.lossfun(X_batch, T_batch)
                    # 逆伝搬を計算
                    loss.backward()
                optimizer.update()
                losses.append(cuda.to_cpu(loss.data))
                if finish is True:
                    break

            time_end = time.time()
            epoch_time = time_end - time_begin
            total_time = time_end - time_origin
            train_losses.append(np.mean(losses))

            valid_loss = loss_ave(model, valid_ite)
            valid_losses.append(valid_loss)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch
                best_model = copy.deepcopy(model)

            # 訓練データでの結果を表示
            print()
            print(file_name)
            print("epoch:", epoch+1)
            print("time", epoch_time, "(", total_time, ")")
            print("loss[train]:", train_losses[epoch])
            print("loss[valid]:", valid_losses[epoch])
            print("best epoch", best_epoch)
            print("loss_best:", best_valid_loss)

            plt.plot(train_losses)
            plt.plot(valid_losses)
            plt.ylim(0, 0.5)
            plt.title("loss")
            plt.legend(["train", "valid"], bbox_to_anchor=(1.2, 1), loc="upper right")
            plt.grid()
            plt.show()

    except KeyboardInterrupt:
        print("割り込み停止が実行されました")

    train_ite.finalize()
    valid_ite.finalize()
    test_ite.finalize()

    plt.plot(train_losses)
    plt.plot(valid_losses)
    plt.ylim(0, 0.5)
    plt.title("loss")
    plt.legend(["train", "valid"], bbox_to_anchor=(1.3, 1), loc="upper right")
    plt.grid()
    plt.savefig(loss_filename)
    plt.show()

    serializers.save_npz(model_filename, best_model)

    print()
    print('max_iteration:', max_iteration)
    print('batch_size:', batch_size)
    print('train_size', num_train)
    print('valid_size', num_valid)
    print('aspect_ratio_max', aspect_ratio_max)
    print('learning_rate:', learning_rate)
    print('best_epoch:', best_epoch)
    print("best_valid_loss:", best_valid_loss)

    return train_losses, valid_losses, best_model
