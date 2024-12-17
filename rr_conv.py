from __future__ import absolute_import

from layer_utils import *
# from activations import GELU, Snake
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from keras.optimizers import *


def RR_CONV(X, channel, kernel_size=3, stack_num=2, recur_num=2, activation='ReLU', batch_norm=False, name='rr'):
    '''
    Recurrent convolutional layers with skip connection.

    RR_CONV(X, channel, kernel_size=3, stack_num=2, recur_num=2, activation='ReLU', batch_norm=False, name='rr')

    Input
    ----------
        X: input tensor.
        channel: number of convolution filters.
        kernel_size: size of 2-d convolution kernels.
        stack_num: number of stacked recurrent convolutional layers.
        recur_num: number of recurrent iterations.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.

    Output
    ----------
        X: output tensor.

    '''

    activation_func = eval(activation)

    layer_skip = Conv2D(channel, 1, name='{}_conv'.format(name))(X)
    layer_main = layer_skip

    for i in range(stack_num):

        layer_res = Conv2D(channel, kernel_size, padding='same', name='{}_conv{}'.format(name, i))(layer_main)

        if batch_norm:
            layer_res = BatchNormalization(name='{}_bn{}'.format(name, i))(layer_res)

        layer_res = activation_func(name='{}_activation{}'.format(name, i))(layer_res)

        for j in range(recur_num):

            layer_add = add([layer_res, layer_main], name='{}_add{}_{}'.format(name, i, j))

            layer_res = Conv2D(channel, kernel_size, padding='same', name='{}_conv{}_{}'.format(name, i, j))(layer_add)
            # layer_res = DepthwiseConv2D(3, depth_multiplier=1, padding='same', name='{}_conv{}_{}'.format(name, i, j))(layer_add)

            if batch_norm:
                layer_res = BatchNormalization(name='{}_bn{}_{}'.format(name, i, j))(layer_res)

            layer_res = activation_func(name='{}_activation{}_{}'.format(name, i, j))(layer_res)

        layer_main = layer_res
    # layer_skip = attention_gate(layer_skip,layer_main,channel,name='{}_att_conv'.format(name))
    out_layer = add([layer_main, layer_skip], name='{}_add{}'.format(name, i))

    return out_layer
