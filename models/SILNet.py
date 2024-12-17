import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from rr_conv import RR_CONV
from models.loss import *

IMAGE_SIZE = 256
filter = 24

activation_value = 'LeakyReLU'
batch_norm_value = False


# SIL-Net

def lswnet_encoder(inputs):
    conv_a1 = Conv2D(filter, 3, padding='same', kernel_initializer='he_normal')(inputs)
    conv_a1 = BatchNormalization()(conv_a1)
    conv_a1 = LeakyReLU(alpha=0.3)(conv_a1)

    conv_a1 = DepthwiseConv2D(3, depth_multiplier=1, padding='same')(conv_a1)
    conv_a1 = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(conv_a1)
    conv_a1 = BatchNormalization()(conv_a1)
    # shutcut1 = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(inputs)

    merge_a1 = concatenate([conv_a1, inputs], axis=3)
    conv_a1 = LeakyReLU(alpha=0.3)(merge_a1)

    # ConvBlock_2_64
    conv_a2 = Conv2D(filter, 3, padding='same', kernel_initializer='he_normal')(conv_a1)
    conv_a2 = BatchNormalization()(conv_a2)
    conv_a2 = LeakyReLU(alpha=0.3)(conv_a2)

    conv_a2 = DepthwiseConv2D(3, depth_multiplier=1, padding='same')(conv_a2)
    conv_a2 = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(conv_a2)
    conv_a2 = BatchNormalization()(conv_a2)

    # shutcut2 = Conv2D(filter, 1,  padding='same', kernel_initializer='he_normal')(merge_a1)
    merge_a2 = concatenate([conv_a2, conv_a1], axis=3)
    conv_a2 = LeakyReLU(alpha=0.3)(merge_a2)

    # Aggre_agationBlock_3_64
    merge_a_unit1 = concatenate([conv_a1, conv_a2], axis=3)
    conv_a_root3 = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(merge_a_unit1)
    Aggre_a3 = LeakyReLU(alpha=0.3)(conv_a_root3)

    #####-----Hierarchical-Block-1------#####

    pool3 = MaxPool2D(pool_size=(2, 2))(Aggre_a3)

    #####-----Hierarchical-Block-2------#####
    # ConvBlock_4_128
    conv_a4 = Conv2D(filter * 2, 3, padding='same', kernel_initializer='he_normal')(pool3)
    conv_a4 = BatchNormalization()(conv_a4)
    conv_a4 = LeakyReLU(alpha=0.3)(conv_a4)

    conv_a4 = DepthwiseConv2D(3, depth_multiplier=1, padding='same')(conv_a4)
    conv_a4 = Conv2D(filter * 2, 1, padding='same', kernel_initializer='he_normal')(conv_a4)
    conv_a4 = BatchNormalization()(conv_a4)

    # shutcut1 = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(inputs)

    merge_a4 = concatenate([conv_a4, pool3], axis=3)
    conv_a4 = LeakyReLU(alpha=0.3)(merge_a4)

    # ConvBlock_5_128
    conv_a5 = Conv2D(filter * 2, 3, padding='same', kernel_initializer='he_normal')(conv_a4)
    conv_a5 = BatchNormalization()(conv_a5)
    conv_a5 = LeakyReLU(alpha=0.3)(conv_a5)

    conv_a5 = DepthwiseConv2D(3, depth_multiplier=1, padding='same')(conv_a5)
    conv_a5 = Conv2D(filter * 2, 1, padding='same', kernel_initializer='he_normal')(conv_a5)
    conv_a5 = BatchNormalization()(conv_a5)

    # shutcut2 = Conv2D(filter, 1,  padding='same', kernel_initializer='he_normal')(merge_a1)
    merge_a5 = concatenate([conv_a4, conv_a5], axis=3)
    conv_a5 = LeakyReLU(alpha=0.3)(merge_a5)

    # Aggre_agationBlock_6_128
    merge_a_unit2 = concatenate([conv_a4, conv_a5], axis=3)
    conv_a_root6 = Conv2D(filter * 2, 1, padding='same', kernel_initializer='he_normal')(merge_a_unit2)
    Aggre_a6 = LeakyReLU(alpha=0.3)(conv_a_root6)

    #####-----Hierarchical-Block-2------#####

    pool6 = MaxPool2D(pool_size=(2, 2))(Aggre_a6)

    #####-----Hierarchical-Block-3------#####

    # ConvBlock_7_256
    conv_a7 = Conv2D(filter * 4, 3, padding='same', kernel_initializer='he_normal')(pool6)
    conv_a7 = BatchNormalization()(conv_a7)
    conv_a7 = LeakyReLU(alpha=0.3)(conv_a7)

    conv_a7 = DepthwiseConv2D(3, depth_multiplier=1, padding='same')(conv_a7)
    conv_a7 = Conv2D(filter * 4, 1, padding='same', kernel_initializer='he_normal')(conv_a7)
    conv_a7 = BatchNormalization()(conv_a7)

    # shutcut1 = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(inputs)

    merge_a7 = concatenate([conv_a7, pool6], axis=3)
    conv_a7 = LeakyReLU(alpha=0.3)(merge_a7)

    # ConvBlock_8_256
    conv_a8 = Conv2D(filter * 4, 3, padding='same', kernel_initializer='he_normal')(conv_a7)
    conv_a8 = BatchNormalization()(conv_a8)
    conv_a8 = LeakyReLU(alpha=0.3)(conv_a8)

    conv_a8 = DepthwiseConv2D(3, depth_multiplier=1, padding='same')(conv_a8)
    conv_a8 = Conv2D(filter * 4, 1, padding='same', kernel_initializer='he_normal')(conv_a8)
    conv_a8 = BatchNormalization()(conv_a8)

    # shutcut2 = Conv2D(filter, 1,  padding='same', kernel_initializer='he_normal')(merge_a1)
    merge_a8 = concatenate([conv_a7, conv_a8], axis=3)
    conv_a8 = LeakyReLU(alpha=0.3)(merge_a8)

    # Aggre_agationBlock_9_256
    merge_a_unit3 = concatenate([conv_a7, conv_a8], axis=3)
    conv_a_root9 = Conv2D(filter * 4, 1, padding='same', kernel_initializer='he_normal')(merge_a_unit3)
    Aggre_a9 = LeakyReLU(alpha=0.3)(conv_a_root9)

    conv_a9_1 = Conv2D(filter * 2, 1, padding='same', kernel_initializer='he_normal')(Aggre_a3)
    conv_a9_1 = MaxPool2D(pool_size=(2, 2))(conv_a9_1)
    skip_a9_1 = concatenate([conv_a9_1, Aggre_a6], axis=3)
    conv_a_skip_a9 = Conv2D(filter * 4, 1, padding='same', kernel_initializer='he_normal')(skip_a9_1)
    conv_a_skip_a9 = LeakyReLU(alpha=0.3)(conv_a_skip_a9)

    conv_a_skip_a9 = MaxPool2D(pool_size=(2, 2))(conv_a_skip_a9)
    skip_a9_2 = concatenate([Aggre_a9, conv_a_skip_a9], axis=3)
    conv_a_skip_a9 = Conv2D(filter * 4, 1, padding='same', kernel_initializer='he_normal')(skip_a9_2)
    conv_a_skip_a9 = LeakyReLU(alpha=0.3)(conv_a_skip_a9)

    #####-----Hierarchical-Block-3------#####
    return conv_a_skip_a9

def lswnet_decoder(conv_a_skip_a9):

    up_a9 = UpSampling2D(size=(2, 2))(conv_a_skip_a9)
        #####-----Hierarchical-Block-4------#####

    # ConvBlock_10_128
    conv_a10 = Conv2D(filter * 2, 3, padding='same', kernel_initializer='he_normal')(up_a9)
    conv_a10 = BatchNormalization()(conv_a10)
    conv_a10 = LeakyReLU(alpha=0.3)(conv_a10)

    conv_a10 = DepthwiseConv2D(3, depth_multiplier=1, padding='same')(conv_a10)
    conv_a10 = Conv2D(filter * 2, 1, padding='same', kernel_initializer='he_normal')(conv_a10)
    conv_a10 = BatchNormalization()(conv_a10)

    # shutcut1 = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(inputs)

    merge_a10 = concatenate([conv_a10, up_a9], axis=3)
    conv_a10 = LeakyReLU(alpha=0.3)(merge_a10)

    # ConvBlock_11_128
    conv_a11 = Conv2D(filter * 2, 3, padding='same', kernel_initializer='he_normal')(conv_a10)
    conv_a11 = BatchNormalization()(conv_a11)
    conv_a11 = LeakyReLU(alpha=0.3)(conv_a11)

    conv_a11 = DepthwiseConv2D(3, depth_multiplier=1, padding='same')(conv_a11)
    conv_a11 = Conv2D(filter * 2, 1, padding='same', kernel_initializer='he_normal')(conv_a11)
    conv_a11 = BatchNormalization()(conv_a11)

    # shutcut2 = Conv2D(filter, 1,  padding='same', kernel_initializer='he_normal')(merge_a1)
    merge_a11 = concatenate([conv_a10, conv_a11], axis=3)
    conv_a11 = LeakyReLU(alpha=0.3)(merge_a11)

    # Aggre_agationBlock_12_128
    merge_a_unit4 = concatenate([conv_a10, conv_a11], axis=3)
    conv_a_root12 = Conv2D(filter * 2, 1, padding='same', kernel_initializer='he_normal')(merge_a_unit4)
    Aggre_a12 = LeakyReLU(alpha=0.3)(conv_a_root12)
    # Aggre_a12 = Conv2D(filter*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
    #     UpSampling2D(size=(2, 2))(Aggre_a12))

    con_up_a12 = UpSampling2D(size=(2, 2))(conv_a_skip_a9)
    skip_a12 = concatenate([Aggre_a12, con_up_a12], axis=3)
    conv_a_skip_a12 = Conv2D(filter * 2, 1, padding='same', kernel_initializer='he_normal')(skip_a12)
    conv_a_skip_a12 = LeakyReLU(alpha=0.3)(conv_a_skip_a12)

    #####-----Hierarchical-Block-4------#####

    up_a12 = UpSampling2D(size=(2, 2))(conv_a_skip_a12)

    #####-----Hierarchical-Block-5------#####

    # ConvBlock_13_64
    conv_a13 = Conv2D(filter, 3, padding='same', kernel_initializer='he_normal')(up_a12)
    conv_a13 = BatchNormalization()(conv_a13)
    conv_a13 = LeakyReLU(alpha=0.3)(conv_a13)

    conv_a13 = DepthwiseConv2D(3, depth_multiplier=1, padding='same')(conv_a13)
    conv_a13 = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(conv_a13)
    conv_a13 = BatchNormalization()(conv_a13)

    # shutcut1 = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(inputs)

    merge_a13 = concatenate([conv_a13, up_a12], axis=3)
    conv_a13 = LeakyReLU(alpha=0.3)(merge_a13)

    # ConvBlock_14_64
    conv_a14 = Conv2D(filter, 3, padding='same', kernel_initializer='he_normal')(conv_a13)
    conv_a14 = BatchNormalization()(conv_a14)
    conv_a14 = LeakyReLU(alpha=0.3)(conv_a14)

    conv_a14 = DepthwiseConv2D(3, depth_multiplier=1, padding='same')(conv_a14)
    conv_a14 = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(conv_a14)
    conv_a14 = BatchNormalization()(conv_a14)

    # shutcut2 = Conv2D(filter, 1,  padding='same', kernel_initializer='he_normal')(merge_a1)
    merge_a14 = concatenate([conv_a13, conv_a14], axis=3)
    conv_a14 = LeakyReLU(alpha=0.3)(merge_a14)

    # Aggre_agationBlock_15_64
    merge_a_unit5 = concatenate([conv_a13, conv_a14], axis=3)
    conv_a_root15 = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(merge_a_unit5)
    Aggre_a15 = LeakyReLU(alpha=0.3)(conv_a_root15)

    con_up_a15 = UpSampling2D(size=(4, 4))(conv_a_skip_a9)
    skip_a15 = concatenate([Aggre_a15, con_up_a15], axis=3)
    conv_a_skip_a15 = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(skip_a15)
    lsw_out = LeakyReLU(alpha=0.3)(conv_a_skip_a15)

    #####-----Hierarchical-Block-5------#####

    # lsw_out = Conv2D(num_class, 1, activation='softmax')(lsw_out)

    return lsw_out


def lrcnet_encoder(inputs):
    #####-----Hierarchical-Block-1------#####

    conv1 = RR_CONV(inputs, filter, stack_num=2, recur_num=2,
                    activation=activation_value, batch_norm=batch_norm_value, name='rr1')
    merge1 = concatenate([conv1, inputs], axis=3)
    conv1 = LeakyReLU(alpha=0.3)(merge1)

    # ConvBlock_2_64
    conv2 = RR_CONV(conv1, filter, stack_num=2, recur_num=2,
                    activation=activation_value, batch_norm=batch_norm_value, name='rr2')
    merge2 = concatenate([conv2, conv1], axis=3)
    conv2 = LeakyReLU(alpha=0.3)(merge2)

    # AggregationBlock_3_64
    merge_unit1 = concatenate([conv1, conv2], axis=3)
    conv_root3 = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(merge_unit1)
    Aggre3 = LeakyReLU(alpha=0.3)(conv_root3)

    #####-----Hierarchical-Block-1------#####

    pool3 = MaxPool2D(pool_size=(2, 2))(Aggre3)

    #####-----Hierarchical-Block-2------#####
    # ConvBlock_4_128
    conv4 = RR_CONV(pool3, filter * 2, stack_num=2, recur_num=2,
                    activation=activation_value, batch_norm=batch_norm_value, name='rr3')
    merge4 = concatenate([conv4, pool3], axis=3)
    conv4 = LeakyReLU(alpha=0.3)(merge4)

    # ConvBlock_5_128
    conv5 = RR_CONV(conv4, filter * 2, stack_num=2, recur_num=2,
                    activation=activation_value, batch_norm=batch_norm_value, name='rr4')
    merge5 = concatenate([conv4, conv5], axis=3)
    conv5 = LeakyReLU(alpha=0.3)(merge5)

    # AggregationBlock_6_128
    merge_unit2 = concatenate([conv4, conv5], axis=3)
    conv_root6 = Conv2D(filter * 2, 1, padding='same', kernel_initializer='he_normal')(merge_unit2)

    mergeA1_A2 = concatenate([conv_root6, pool3], axis=3)
    conv_root6 = Conv2D(filter * 2, 1, padding='same', kernel_initializer='he_normal')(mergeA1_A2)

    Aggre6 = LeakyReLU(alpha=0.3)(conv_root6)

    #####-----Hierarchical-Block-2------#####

    pool6 = MaxPool2D(pool_size=(2, 2))(Aggre6)

    #####-------Center-Block-------#####

    # ConvC1_Block_16
    merge_unit_C1 = concatenate([pool3, Aggre6], axis=3)
    convC1 = Conv2D(filter * 2, 1, padding='same', kernel_initializer='he_normal')(merge_unit_C1)
    convC1 = BatchNormalization()(convC1)
    AggreC1 = LeakyReLU(alpha=0.3)(convC1)

    AggreC1 = RR_CONV(AggreC1, filter, stack_num=2, recur_num=2,
                      activation=activation_value, batch_norm=batch_norm_value, name='rr_c1')

    # ConvC3_Block_18
    merge_unit_C3_1 = concatenate([pool3, AggreC1], axis=3)
    AggreC3_1 = RR_CONV(merge_unit_C3_1, filter * 2, stack_num=2, recur_num=2,
                        activation=activation_value, batch_norm=batch_norm_value, name='rr_c31')
    AggreC3_1 = MaxPool2D(pool_size=(2, 2))(AggreC3_1)

    merge_unit_C3_2 = concatenate([pool6, AggreC3_1], axis=3)
    AggreC3_2 = RR_CONV(merge_unit_C3_2, filter * 4, stack_num=2, recur_num=2,
                        activation=activation_value, batch_norm=batch_norm_value, name='rr_c32')
    # convC3_3 = RR_CONV(AggreC3_2, filter * 2, stack_num=2, recur_num=2,
    #                  activation=activation_value, batch_norm=batch_norm_value, name='rr_c33')
    # ConvC2_Block_17
    return AggreC3_2

def lrcnet_decoder(AggreC3_2):
    convC3 = Conv2D(filter * 2, 1, padding='same', kernel_initializer='he_normal')(AggreC3_2)
    convC3 = BatchNormalization()(convC3)
    convC3 = LeakyReLU(alpha=0.3)(convC3)
    AggreC3 = RR_CONV(convC3, filter * 2, stack_num=2, recur_num=2,
                      activation=activation_value, batch_norm=batch_norm_value, name='rr_c2')
    #####-------Center-Block-------#####

    up9 = UpSampling2D(size=(2, 2))(AggreC3_2)
    # up10 = UpSampling2D(size=(2, 2))(AggreC3)

    #####-----Hierarchical-Block-3------#####

    # ConvBlock_10_128
    conv10 = RR_CONV(up9, filter * 2, stack_num=2, recur_num=2,
                     activation=activation_value, batch_norm=batch_norm_value, name='rr5')
    merge10 = concatenate([conv10, up9], axis=3)
    conv10 = LeakyReLU(alpha=0.3)(merge10)

    # ConvBlock_11_128
    conv11 = RR_CONV(conv10, filter * 2, stack_num=2, recur_num=2,
                     activation=activation_value, batch_norm=batch_norm_value, name='rr6')
    merge11 = concatenate([conv10, conv11], axis=3)
    conv11 = LeakyReLU(alpha=0.3)(merge11)

    # AggregationBlock_12_128
    merge_unit4 = concatenate([conv10, conv11], axis=3)
    conv_root12 = Conv2D(filter * 2, 1, padding='same', kernel_initializer='he_normal')(merge_unit4)
    Aggre12 = LeakyReLU(alpha=0.3)(conv_root12)

    skip12 = concatenate([Aggre12, up9], axis=3)
    conv_skip12 = Conv2D(filter * 2, 1, padding='same', kernel_initializer='he_normal')(skip12)
    conv_skip12 = LeakyReLU(alpha=0.3)(conv_skip12)

    skip14 = concatenate([conv_skip12, up9], axis=3)
    conv_skip14 = Conv2D(filter * 2, 1, padding='same', kernel_initializer='he_normal')(skip14)
    conv_skip14 = LeakyReLU(alpha=0.3)(conv_skip14)

    #####-----Hierarchical-Block-3------#####

    up12 = UpSampling2D(size=(2, 2))(conv_skip14)

    #####-----Hierarchical-Block-4------#####

    # ConvBlock_13_64
    conv13 = RR_CONV(up12, filter * 2, stack_num=2, recur_num=2,
                     activation=activation_value, batch_norm=batch_norm_value, name='rr7')
    merge13 = concatenate([conv13, up12], axis=3)
    conv13 = LeakyReLU(alpha=0.3)(merge13)

    # ConvBlock_14_64
    conv14 = RR_CONV(conv13, filter * 2, stack_num=2, recur_num=2,
                     activation=activation_value, batch_norm=batch_norm_value, name='rr8')
    merge14 = concatenate([conv13, conv14], axis=3)
    conv14 = LeakyReLU(alpha=0.3)(merge14)

    # AggregationBlock_15_64
    merge_unit5 = concatenate([conv13, conv14], axis=3)
    conv_root15 = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(merge_unit5)
    Aggre15 = LeakyReLU(alpha=0.3)(conv_root15)

    # con_up15 = UpSampling2D(size=(4, 4))(conv_skip9)
    skip15 = concatenate([Aggre15, up12], axis=3)
    conv_skip15 = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(skip15)
    conv_skip15 = LeakyReLU(alpha=0.3)(conv_skip15)

    skip16 = concatenate([AggreC3, AggreC3_2], axis=3)
    skip16 = UpSampling2D(size=(4, 4))(skip16)
    conv_skip16 = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(skip16)
    conv_skip16 = LeakyReLU(alpha=0.3)(conv_skip16)

    skip17 = concatenate([conv_skip15, conv_skip16], axis=3)
    lrc_out = RR_CONV(skip17, filter, stack_num=2, recur_num=2,
                      activation=activation_value, batch_norm=batch_norm_value, name='conv_out')

    #####-----Hierarchical-Block-4------#####

    # lrc_out = Conv2D(num_class, 1, activation='softmax')(lrc_out)

    return lrc_out


def net(pretrained_weights=None, input_size=(IMAGE_SIZE, IMAGE_SIZE, 1), num_class=20):
    input_1 = Input(input_size)
    input_2 = Input(input_size)
    input_3 = Input(input_size)
    input_B = Input(input_size)

    #####-----inputs_plug------#####
    conv_input_1 = Conv2D(filter, 3, padding='same', kernel_initializer='he_normal')(input_1)
    conv_input_1 = LeakyReLU(alpha=0.3)(conv_input_1)
    conv_input_2 = Conv2D(filter, 3, padding='same', kernel_initializer='he_normal')(input_2)
    conv_input_2 = LeakyReLU(alpha=0.3)(conv_input_2)
    conv_input_3 = Conv2D(filter, 3, padding='same', kernel_initializer='he_normal')(input_3)
    conv_input_3 = LeakyReLU(alpha=0.3)(conv_input_3)

    merge_ip1 = concatenate([conv_input_1, conv_input_2], axis=3)
    merge_ip2 = concatenate([conv_input_2, conv_input_3], axis=3)
    merge_ip3 = concatenate([conv_input_1, conv_input_3], axis=3)

    conv_input_4 = Conv2D(filter, 3, padding='same', kernel_initializer='he_normal')(merge_ip1)
    conv_input_4 = LeakyReLU(alpha=0.3)(conv_input_4)
    conv_input_5 = Conv2D(filter, 3, padding='same', kernel_initializer='he_normal')(merge_ip2)
    conv_input_5 = LeakyReLU(alpha=0.3)(conv_input_5)
    conv_input_6 = Conv2D(filter, 3, padding='same', kernel_initializer='he_normal')(merge_ip3)
    conv_input_6 = LeakyReLU(alpha=0.3)(conv_input_6)

    fused_1 = Concatenate()([conv_input_4, conv_input_5, conv_input_6])
    #####-----inputs_plug------#####

    lrcnet_rail_line = lrcnet_encoder(fused_1)
    lswnet_rail_line = lswnet_encoder(input_B)

    rail_line_union=Concatenate()([lrcnet_rail_line,lswnet_rail_line])

    lrcnet_tail=lrcnet_decoder(rail_line_union)
    lswnet_tail=lswnet_decoder(rail_line_union)


    LRCnet_out = Conv2D(num_class, 1, activation='softmax', name='LRCnet_out')(lrcnet_tail)


    LSWnet_out = Conv2D(num_class, 1, activation='softmax', name='LSWnet_out')(lswnet_tail)

    model = Model(inputs=[input_1, input_2, input_3, input_B], outputs=[LRCnet_out, LSWnet_out])

    model.compile(loss={'LRCnet_out': 'categorical_crossentropy',
                        'LSWnet_out': LSWnet_loss},
                  loss_weights={
                      'LRCnet_out': 1.,
                      'LSWnet_out': 1.
                  },
                  metrics=['accuracy'])
    # model.summary()

    return model
