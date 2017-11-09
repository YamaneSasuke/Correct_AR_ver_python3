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
