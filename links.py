# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 21:41:05 2017

@author: yamane
"""

import chainer
import chainer.functions as F
import chainer.links as L

class CBR(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None, **kwargs):
        super(CBR, self).__init__(
            conv=L.Convolution2D(in_channels, out_channels, ksize, stride, pad,
                                 nobias, initialW, initial_bias, **kwargs),
            bn=L.BatchNormalization(out_channels)
        )

    def __call__(self, x):
        return F.relu(self.bn(self.conv(x)))

# ネットワークの定義
class ARConvnet(chainer.Chain):
    def __init__(self):
        super(ARConvnet, self).__init__(
            cbr1_1=CBR(3, 64, 3, 2, 1),
            cbr2_1=CBR(64, 128, 3, 2, 1),
            cbr3_1=CBR(128, 128, 3, 2, 1),
            cbr4_1=CBR(128, 256, 3, 1, 1),
            cbr4_2=CBR(256, 256, 3, 2, 1),
            cbr5_1=CBR(256, 512, 3, 1, 1),
            cbr5_2=CBR(512, 512, 3, 2, 1),
        )

    def __call__(self, X):
        h = self.cbr1_1(X)
        h = self.cbr2_1(h)
        h = self.cbr3_1(h)
        h = self.cbr4_1(h)
        h = self.cbr4_2(h)
        h = self.cbr5_1(h)
        h = self.cbr5_2(h)
        return h
