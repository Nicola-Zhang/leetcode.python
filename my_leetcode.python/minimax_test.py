#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
=================================================
@Project -> File ：my_leetcode.python -> minimax_test.py
@Author ：Nicola.Zhang
@E-mail: xuzhang0423@gmail.com
@Date ：2024/4/26 20:06 
@Desc ：
==================================================

标题
无自动求导的线性回归算法

题目描述
# 面试题目：​

**任务：** 实现一个无自动求导的线性回归算法。​

**数据集：** 生成一批随机样本，其中输入特征 `x` 的维度为 `n * d`，标签 `y` 的维度为 `n * 1`。​

**目标：** 实现简单的线性回归模型。模型训练过程中，我们希望观察到损失函数的下降并且模型参数收敛到预设的某个值。​

**超参数设置：** 可以自行设置诸如 epoch 数、学习率、batch 大小（大于 1）等超参数。​

**要求：** 仅使用基本库，不限制语言​

提示：y = w1 * x1 + w2 * x2 + b​
L2 loss: loss = (y^hat - y)**2


dLoss / dw = 2 * (w*x1+b - y1) * x1 + 2 * (w*x2+b - y2) * x2
dLoss / db = 2 * (w*x1+b - y1) + 2 * (w*x2+b - y2)

'''
from typing import Any, AnyStr, Tuple, List, Dict, Optional, Union
import numpy as np


class Linear(object):
    def __init__(self, x_in, x_out, bias=True):
        self.weight = np.random.random([x_in, x_out])
        self.bias = None
        if bias:
            self.bias = np.random.random([x_out])

    def forward(self, x):
        output = x @ self.weight
        if self.bias is not None:
            output += self.bias
        return output

    def backward(self, loss, lr):
        diff = 0


class LearningRate:
    def __init__(self, total_steps):
        self.lr = 1.
        self.diff = (self.lr-0.) / total_steps

    def forward(self):
        self.lr -= self.diff
        return self.lr


def loss_l2(y_pred, y):
    return (y-y_pred)**2


def dataloader(dataset, batch_size):
    bsz = batch_size
    d_size = len(dataset)
    for s_idx in range(0, d_size, batch_size):
        e_idx = s_idx+bsz if s_idx+bsz < d_size else d_size
        yield dataset[s_idx:e_idx]


class Model:
    def __init__(self, x_in, x_out):
        self.linear1 = Linear(x_in, 50, bias=True)
        self.linear2 = Linear(50, x_out, bias=True)

    def forward(self, x):
        x = self.linear1.forward(x)
        x = self.linear2.forward(x)
        return x

    def backward(self, loss, lr):
        return 0


def train(dataset, epoch, batch):
    x_in, x_out = 2,1
    model = Model(2,1)
    lr_sc = LearningRate(100)

    train_dataloader = dataloader(dataset, batch)

    for ep in range(epoch):
        for x, y in train_dataloader:
            y_pred = model.forward(x)
            loss = loss_l2(y_pred, y)
            model.backward(loss, lr_sc.forward())

