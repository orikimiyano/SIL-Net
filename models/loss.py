from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
import tensorflow as tf
import math


smooth = 1.

def Cross_entropy_loss(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    crossEntropyLoss = -y_true * tf.math.log(y_pred)

    return tf.reduce_sum(crossEntropyLoss, -1)

def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def CE_tversky(y_true, y_pred):
    beta = 0.4
    loss_value = beta * Cross_entropy_loss(y_true, y_pred) + (1-beta) * tversky_loss(y_true, y_pred)
    return loss_value


def f_beta_c(true, seg):
    # 完全重叠是值为1，完全不重叠时为0
    beta = 0.5

    precision = K.sum(true * seg) / (K.sum(true * seg) + K.sum(true * seg * seg) + smooth)
    recall = K.sum(true * seg) / (K.sum(true * seg) + K.sum(true * true * seg) + smooth)
    F_beta = (1 + beta ** 2) * precision * recall / ((beta ** 2 * precision + recall) + smooth)

    return F_beta


def eu_distance(true, seg):
    # 完全重叠是值为距离为0；完全不重叠时为距离为无穷大，归一化后为1
    pos_g = K.flatten(true)
    pos_p = K.flatten(seg)
    mul_p_g = pos_g * pos_p
    area_size = K.sum(pos_g - mul_p_g) + K.sum(pos_p - mul_p_g)
    normalize_c = tf.cast(area_size, dtype=tf.float32) / tf.cast(tf.size(true), dtype=tf.float32) + smooth

    return normalize_c


def dynamic_coefficient(true, seg):
    d_c = 0.5 * tf.cast((1 - f_beta_c(true, seg)) + eu_distance(true, seg), dtype=tf.float32)

    return d_c


# '''
# loss for LRCnet
# '''
#
#
# def LRCnet_loss(y_true, y_pred):
#     y_true_n = K.reshape(y_true, shape=(-1, 4))
#     y_pred_n = K.reshape(y_pred, shape=(-1, 4))
#     total_single_loss = 0.
#
#     for i in range(y_pred_n.shape[1]):
#         single_loss = CE_tversky(y_true_n[:, i], y_pred_n[:, i])
#         total_single_loss += single_loss
#         c_of_d = dynamic_coefficient(y_true_n[:, i], y_pred_n[:, i])
#         # alph = tf.cast(c_of_d, dtype=tf.float32) / tf.cast(tf.size(y_true_n[:, i]), dtype=tf.float32)
#         # alph = 0.5 * tf.cast(tf.sin(math.pi * c_of_d) + 1, dtype=tf.float32)
#         # alph = -(c_of_d - 1) ** 4 + 10 / 7
#         # alph = alph * 0.7
#         alph = (c_of_d ** 4) / 2
#         total_single_loss = alph * total_single_loss
#
#     return total_single_loss/1000
#

'''
loss for LSWnet
'''


def LSWnet_loss(y_true, y_pred):
    y_true_n = K.reshape(y_true, shape=(-1, 4))
    y_pred_n = K.reshape(y_pred, shape=(-1, 4))
    total_single_loss = 0.

    for i in range(y_pred_n.shape[1]):
        single_loss = CE_tversky(y_true_n[:, i], y_pred_n[:, i])
        total_single_loss += single_loss
        c_of_d = dynamic_coefficient(y_true_n[:, i], y_pred_n[:, i])
        # alph = tf.cast(c_of_d, dtype=tf.float32) / tf.cast(tf.size(y_true_n[:, i]), dtype=tf.float32)
        # alph = 0.5 * tf.cast(tf.sin(math.pi * c_of_d) + 1, dtype=tf.float32)
        # alph = -(c_of_d - 1) ** 4 + 10 / 7
        # alph = alph * 0.7
        alph = (c_of_d ** 4) / 2
        total_single_loss = alph * total_single_loss

    return total_single_loss/50
