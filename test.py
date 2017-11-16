# -*- coding: utf-8 -*-
"""
Created on Fri Jan 06 16:59:52 2017

@author: yamane
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import chainer
from chainer import serializers, cuda

import ave_pooling, max_pooling, conv_pooling, bias_sum_pooling
import utils
from load_datasets import TestDataset

def fix(model, data_iter, t):
    batch = data_iter.get_example(t)
    x = cuda.to_gpu(batch[0])
    t = batch[1]
    with chainer.using_config('train', False):
        y = model(x)
    y = cuda.to_cpu(y.data)
    error = t - y
    error_abs = np.abs(t - y)
    return error, error_abs


def draw_graph(loss, loss_abs, success_asp, num_test, t_list, save_path):
    average_abs_file = os.path.join(save_path, 'average_abs')
    loss_file = os.path.join(save_path, 'loss')
    dot_file = os.path.join(save_path, 'dot_hist')
    average_asp_file = os.path.join(save_path, 'average_asp')
    average_asp_abs_file = os.path.join(save_path, 'average_asp_abs')
    num_t = len(t_list)
    prot_t = []
    for i in t_list:
        prot_t.append(round(i, 1))
    threshold = np.log(success_asp)
    base_line = np.ones((num_test,))
    for i in range(num_test):
        base_line[i] = threshold

    plt.rcParams["font.size"] = 14
    mean_loss_abs = np.mean(loss_abs, axis=0)
    std_abs = np.std(loss_abs, axis=0)
    plt.figure(figsize=(10, 3))
    plt.errorbar(range(100) ,mean_loss_abs, marker='o', linewidth=0, elinewidth=1.5, yerr=std_abs, label='avg. abs. error + std')
    plt.plot(base_line, label='log(1.1303)')
    plt.legend(loc="upper left")
    plt.xlabel('Image ID of test data', fontsize=20)
    plt.ylabel('Avg abs. error in log', fontsize=20)
    plt.ylim(0, max(mean_loss_abs)+max(std_abs)+0.1)
    plt.grid()
    plt.savefig(average_abs_file+'.jpg', format='jpg', bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(9, 3))
    plt.boxplot(loss)
    plt.xlim([np.log(1/3.5), np.log(3.5)])
    plt.xticks(range(num_t), prot_t)
    plt.title('Error for each aspect ratio in log scale', fontsize=24)
    plt.xlabel('Order of aspect ratio in log scale', fontsize=24)
    plt.ylabel('Error(t-y) in log scale', fontsize=24)
    plt.grid()
    plt.savefig(loss_file+'.jpg', format='jpg', bbox_inches='tight')
    plt.show()

    loss_dot = np.stack(loss, axis=0)
    loss_dot = loss_dot.reshape(num_t, num_test)
    average = np.mean(loss, axis=1)
    plt.figure(figsize=(9, 3))
    plt.plot(loss_dot, 'o', c='#348ABD')
    plt.plot(average, label='average')
    plt.xticks(range(num_t), prot_t)
    plt.title('Error for each aspect ratio in log scale', fontsize=24)
    plt.legend(loc="upper left", fontsize=20)
    plt.xlabel('Order of aspect ratio in log scale', fontsize=24)
    plt.ylabel('Error(t-y) in log scale', fontsize=24)
    plt.grid()
    plt.savefig(dot_file+'.jpg', format='jpg', bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(9, 3))
    plt.plot(average, label='average error')
    plt.plot(base_line, label='log(1.1303)')
    plt.plot(-base_line, label='log(1.1303^-1)')
    plt.xticks(range(num_t), prot_t)
    plt.xlim(0, num_t)
    plt.title('average Error for each aspect ratio in log scale', fontsize=24)
    plt.legend(loc="upper left", fontsize=20)
    plt.xlabel('Order of aspect ratio in log scale', fontsize=24)
    plt.ylabel('average Error(t-y) in log scale', fontsize=24)
    plt.grid()
    plt.savefig(average_asp_file+'.jpg', format='jpg', bbox_inches='tight')
    plt.show()

    average_abs = np.mean(loss_abs, axis=1)
    std_abs = np.std(loss_abs, axis=1)
    plt.figure(figsize=(10, 3))
    plt.errorbar(range(num_t) ,average_abs, yerr=std_abs, label='avg. abs. error + std')
    plt.plot(base_line, label='log(1.1303)')
    plt.xticks(range(num_t), prot_t)
    plt.xlim(0, num_t)
    plt.ylim(0, max(mean_loss_abs)+0.05)
    plt.legend(loc="upper right")
    plt.xlabel('Distortion of aspect ratio in log scale', fontsize=20)
    plt.ylabel('Avg. abs. error in log', fontsize=20)
    plt.grid()
    plt.savefig(average_asp_abs_file+'.jpg', format='jpg', bbox_inches='tight')
    plt.show()

    count = 0
    for i in range(num_test):
        if mean_loss_abs[i] < threshold:
            count += 1
    print('under log(1.1303) =', count, '%')
    print('num_test', num_test)
    print('model_file', model_file)


if __name__ == '__main__':
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    # テスト結果を保存するルートパス
    save_root = r'C:\Users\yamane\OneDrive\M1\correct_aspect_ratio\demo'
    # モデルのルートパス
    model_file = r'C:\Users\yamane\OneDrive\M1\correct_aspect_ratio\bias_sum_pooling\1509613391.5179036\bias_sum_pooling.npz'
    success_asp = np.exp(0.12247601469)  # 修正成功とみなすアスペクト比
    num_split = 20  # 歪み画像のアスペクト比の段階
    num_test = 100

    loss_list = []
    loss_abs_list = []
    t_list = []
    folder_name = model_file.split('\\')[-2]

    num_t = num_split + 1
    t_step = np.log(3.0) * 2 / num_split
    t = np.log(1/3.0)
    for i in range(num_t):
        t_list.append(t)
        t = t + t_step

    # 結果を保存するフォルダを作成
    folder_path = utils.create_folder(save_root, folder_name)
    # モデル読み込み
    model = bias_sum_pooling.BiasSumPooling().to_gpu()
    # Optimizerの設定
    serializers.load_npz(model_file, model)

    test_data = TestDataset(100, 17000, 17100)
    # アスペクト比ごとに歪み画像を作成し、修正誤差を計算
    for t in t_list:
        print(t)
        loss, loss_abs = fix(model, test_data, t)
        loss_list.append(loss)
        loss_abs_list.append(loss_abs)
    # 修正誤差をグラフに描画
    draw_graph(loss_list, loss_abs_list, success_asp, num_test, t_list, folder_path)
