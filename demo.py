# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:27:22 2017

@author: yamane
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import chainer
from chainer import serializers, cuda

import utils
from load_datasets import Dataset, TestDataset
import ave_pooling, max_pooling, conv_pooling, bias_sum_pooling
import make_html


def lossfun(model):
    test_data = Dataset(100, 17000, 17100, 3.0)
    batch = test_data.get_example(0)
    x = cuda.to_gpu(batch[0])
    t_l = batch[1]
    with chainer.using_config('train', True):
        y_l = model(x)
    e_l = t_l - cuda.to_cpu(y_l.data)
    e_l_abs = np.abs(t_l - cuda.to_cpu(y_l.data))
    loss = e_l
    loss_abs = e_l_abs
    target = t_l
    predict = cuda.to_cpu(y_l.data)

    return loss, loss_abs, target, predict


def show_and_save(target, predict, save_path_f, save_path_d, save_path_o):
    test_data = TestDataset(1, 17000, 17100)
    t_r = np.exp(target)
    y_r = np.exp(predict)
    for batch in range(100):
        e_l = target[batch] - predict[batch]
        e_r = t_r[batch] - y_r[batch]

        b = test_data.get_example(1.0)
        img = np.transpose(b[0][0], (1, 2, 0))
        dis_img = utils.change_aspect_ratio(img, t_r[batch], 1)
        fix_img = utils.change_aspect_ratio(dis_img, 1/y_r[batch], 1)

        print('[test_data]:', batch+1)
        print('[t_l]:', round(target[batch][0][0], 4), '\t[t_r]:', round(t_r[batch][0][0], 4))
        print('[y_l]:', round(predict[batch][0][0], 4), '\t[y_r]:', round(y_r[batch][0][0], 4))
        print('[e_l]:', round(e_l[batch][0], 4), '\t[e_r]:', round(e_r[batch][0], 4))

        plt.figure(figsize=(16, 16))
        plt.subplot(131)
        plt.title('Distorted image')
        plt.tick_params(labelbottom='off', labeltop='off', labelleft='off',
                        labelright='off')
        plt.tick_params(bottom='off', top='off', left='off', right='off')
        plt.imshow(dis_img/255)
        plt.subplot(132)
        plt.title('Fixed image')
        plt.tick_params(labelbottom='off', labeltop='off', labelleft='off',
                        labelright='off')
        plt.tick_params(bottom='off', top='off', left='off', right='off')
        plt.imshow(fix_img/255)
        plt.subplot(133)
        plt.title('Normal image')
        plt.tick_params(labelbottom='off', labeltop='off', labelleft='off',
                        labelright='off')
        plt.tick_params(bottom='off', top='off', left='off', right='off')
        plt.imshow(img/255)
        plt.show()

        utils.save_image(dis_img, save_path_d, ('%.18f' % e_l[batch]))
        utils.save_image(fix_img, save_path_f, ('%.18f' % e_l[batch]))
        utils.save_image(img, save_path_o, ('%.18f' % e_l[batch]))

    make_html.make_html(save_path_d)
    make_html.make_html(save_path_f)
    make_html.make_html(save_path_o)


def draw_graph(loss, loss_abs, success_asp, num_test, save_root):
    loss_abs_file = os.path.join(save_root, 'loss_abs')
    loss_file = os.path.join(save_root, 'loss')
    loss_hist = os.path.join(save_root, 'loss_hist')
    threshold = np.log(success_asp)
    base_line = np.ones((num_test,))
    for i in range(num_test):
        base_line[i] = threshold

    error_abs = np.stack(loss_abs, axis=0)
    error_abs = error_abs.reshape(num_test, 1)
    error = np.stack(loss, axis=0)
    error = error.reshape(num_test, 1)

    if np.abs(max(error)) > np.abs(min(error)):
        max_value = np.abs(max(error))
    else:
        max_value = np.abs(min(error))

    plt.rcParams["font.size"] = 18
    plt.figure(figsize=(8, 3))
    plt.plot(error_abs)
    plt.plot(base_line, 'r-')
    plt.legend(["Error", "log(1.1303)"], loc="upper left")
    plt.xlabel('Order of test data number', fontsize=24)
    plt.ylabel('Error(|t-y|) in log scale', fontsize=24)
    plt.ylim(0, max(error_abs)+0.01)
    plt.grid()
    plt.savefig(loss_abs_file+'.jpg', format='jpg', bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(8, 3))
    plt.plot(error, label='Error')
    plt.plot(base_line, label="log(1.1303)")
    plt.plot(-base_line, label="log(1.1303^-1)")
    plt.title('Error for each test data', fontsize=28)
    plt.legend(loc="upper left")
    plt.xlabel('Order of test data number', fontsize=28)
    plt.ylabel('Error(t-y) in log scale', fontsize=28)
    plt.ylim(-max_value-0.01, max_value+0.01)
    plt.grid()
    plt.savefig(loss_file+'.jpg', format='jpg', bbox_inches='tight')
    plt.show()

    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(error, bins=22, range=(-1.0, 1.0))
    ax.set_xlabel('Error in log scale', fontsize=20)
    ax.set_ylabel('Percentage', fontsize=20)
    plt.grid()
#    plt.xlim(-1, 1)
    plt.savefig(loss_hist+'.jpg', format='jpg', bbox_inches='tight')
    fig.show()

    count = 0
    for i in range(num_test):
        if loss_abs[i][0] < threshold:
            count += 1
    print('under log(1.1303) =', count, '%')
    print('[mean]:', np.mean(loss_abs))


if __name__ == '__main__':
    # テスト結果を保存する場所
    save_root = r'E:\demo'
    # テストに使うモデルのnpzファイルの場所
    model_file = r'C:\Users\yamane\OneDrive\M1\correct_aspect_ratio\ave_pooling\1510298648.711074\ave_pooling.npz'
    num_test = 100  # テストデータ数
    success_asp = np.exp(0.12247601469)  # 修正成功とみなす修正画像のアスペクト比の最大値
    batch_size = 100

    # モデルのファイル名をフォルダ名にする
    folder_name = model_file.split('\\')[-2]

    # テスト結果を保存するフォルダを作成
    test_folder_path = utils.create_folder(save_root, folder_name)
    fix_folder_path = utils.create_folder(test_folder_path, 'fix')
    dis_folder_path = utils.create_folder(test_folder_path, 'distorted')
    ori_folder_path = utils.create_folder(test_folder_path, 'original')

    # モデル読み込み
    model = ave_pooling.AvePooling().to_gpu()
    serializers.load_npz(model_file, model)

    # 歪み画像の修正を実行
    loss, loss_abs, target, predict = lossfun(model)

    show_and_save(target, predict, fix_folder_path, dis_folder_path, ori_folder_path)

    # 修正結果の誤差を描画
    draw_graph(loss, loss_abs, success_asp, num_test, test_folder_path)
