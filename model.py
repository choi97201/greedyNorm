# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
from abc import abstractmethod, ABCMeta
import keras
import tensorflow as tf
import keras.backend as K
from keras.layers import DepthwiseConv2D
from keras.layers import AveragePooling2D
from keras.layers import Add
from keras.layers import Lambda
from keras.layers import Concatenate
from keras.layers import GlobalAveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import Conv2DTranspose
from keras.layers import Multiply
from keras.engine.topology import Layer
from keras_contrib.layers import InstanceNormalization
from keras_layer_normalization import LayerNormalization
from keras.layers.merge import _Merge
from keras.constraints import Constraint
from group_norm import GroupNormalization
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
# spectral norm
import matplotlib.pyplot as plt
from SpectralNormalizationKeras import DenseSN, ConvSN2D, ConvSN2DTranspose, _ConvSN

session = tf.Session()
session.run(tf.initialize_all_variables())

class BilinearUpSampling(Layer):
    def __init__(self, size, name, **kwargs):
        super(BilinearUpSampling, self).__init__(**kwargs)
        self.size = size
        self.name = name

    def call(self, x):
        return tf.image.resize_bilinear(x, size=self.size, align_corners=True, name=self.name)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.size[0], self.size[1], input_shape[3])


def Conv2D_same(x, filters, part, strides=1, kernel_size=3, rate=1):
    if strides == 1:
        return Conv2D(filters, (kernel_size, kernel_size),
                      strides=(strides, strides),
                      padding='same',
                      use_bias=False,
                      dilation_rate=(rate, rate),
                      name=part)(x)

    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        return Conv2D(filters, (kernel_size, kernel_size),
                      strides=(strides, strides),
                      padding='valid', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=part)(x)


def SeparableConvBlock(x, filters, kernel_size=3, strides=1, rate=1, part=None, last_activation=False):
    '''
    Separable Convolution Block used in MobileNet and Xception
    '''
    if strides == 1:
        padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        padding = 'valid'

    if not last_activation:
        x = Activation(activation='relu', name=part + '_early_act')(x)

    # Depwise Convolution
    x = DepthwiseConv2D((3, 3), strides=(strides, strides), use_bias=False, padding=padding,
                        dilation_rate=(rate, rate),
                        name=part + '_depthwise_conv')(x)
    x = BatchNormalization(name=part + '_depthwise_bn')(x)
    x = Activation(activation='relu', name=part + '_depthwise_act')(x)

    # Pointwise Convolution
    x = Conv2D(filters, (1, 1), use_bias=False, padding='same', name=part + '_pointwise_conv')(x)
    x = BatchNormalization(name=part + '_pointwise_bn')(x)
    if last_activation:
        x = Activation(activation='relu', name=part + '_pointwise_act')(x)

    return x


def xception_block(inputs, filters, strides, rate=1, part=None, skip_connection=False, mode='conv',
                   last_activation=False):
    '''
    Xception Block in DeepLabv3+
    '''

    x = inputs
    for i in range(3):
        x = SeparableConvBlock(x, filters[i],
                               strides=strides if i == 2 else 1,
                               rate=rate,
                               part=part + '_seprable' + str(i + 1),
                               last_activation=last_activation)

        if i == 1 and skip_connection:
            skip = x

    if mode == 'conv':
        residual = Conv2D_same(inputs, filters[1], part=part + '_residual_conv', kernel_size=1, strides=strides)
        residual = BatchNormalization(name=part + '_residual_bn')(residual)
        x = Add()([x, residual])
    elif mode == 'sum':
        x = Add()([x, inputs])

    if skip_connection:
        return x, skip
    else:
        return x

#
def DeepLab_Encoder(img_input, part='encoder', divide=1):
    '''
    Encoder Module using Modified Xception Baseline
    '''
    # Entry flow
    x = Conv2D(32 // divide, (3, 3), strides=(2, 2), use_bias=False, padding='same', name=part + '_block1_conv1')(
        img_input)
    x = BatchNormalization(name=part + '_block1_conv1_bn1')(x)
    x = Activation(activation='relu', name=part + '_block1_conv1_act1')(x)

    x = Conv2D_same(x, 64 // divide, part + '_block1_conv2', kernel_size=3, strides=1)
    x = BatchNormalization(name=part + '_block1_conv2_bn2')(x)
    x = Activation(activation='relu', name=part + '_block1_conv2_act2')(x)

    x = xception_block(x, [128 // divide, 128 // divide, 128 // divide], 2, 1, part + '_block2', mode='conv')
    x, skip = xception_block(x, [256 // divide, 256 // divide, 256 // divide], 2, 1, part + '_block3', True,
                             mode='conv')
    x = xception_block(x, [728 // divide, 728 // divide, 728 // divide], 2, 1, part + '_block4', mode='conv')

    # Middle flow
    for i in range(16):
        part_ = part + '_block' + str(i + 5)
        x = xception_block(x, [728 // divide, 728 // divide, 728 // divide], 1, 1, part_, mode='sum')

    # Exit flow
    x = xception_block(x, [728 // divide, 1024 // divide, 1024 // divide], 1, 1, part + '_block21', mode='conv')
    x = xception_block(x, [1536 // divide, 1536 // divide, 2048 // divide], 1, 2, part + '_block22', mode='none',
                       last_activation=True)

    '''
    Encoder Module using Atrous Spatial Pyramid Pooling(ASPP) in DeepLabv3 and ParseNet
    '''
    b0 = Conv2D(256 // divide, (1, 1), use_bias=False, padding='same', name=part + '_aspp0')(x)
    b0 = BatchNormalization(name=part + '_aspp0_bn')(b0)
    b0 = Activation(activation='relu', name=part + '_aspp0_act')(b0)

    b1 = SeparableConvBlock(x, 256 // divide, part=part + '_aspp1', rate=6, last_activation=True)
    b2 = SeparableConvBlock(x, 256 // divide, part=part + '_aspp2', rate=12, last_activation=True)
    b3 = SeparableConvBlock(x, 256 // divide, part=part + '_aspp3', rate=18, last_activation=True)
    print()
    b4 = AveragePooling2D(pool_size=(K.int_shape(x)[1], K.int_shape(x)[2]), name=part + '_image_pooling')(x)
    b4 = Conv2D(256 // divide, (1, 1), use_bias=False, padding='same', name=part + '_image_pooling_conv')(b4)
    b4 = BatchNormalization(name=part + '_image_pooling_bn')(b4)
    b4 = Activation(activation='relu', name=part + '_image_pooling_act')(b4)
    b4 = BilinearUpSampling(size=(K.int_shape(x)[1], K.int_shape(x)[2]), name=part + '_image_pooling_unpool')(b4)

    x = Concatenate()([b0, b1, b2, b3, b4])
    x = Conv2D(256 // divide, (1, 1), use_bias=False, padding='same', name=part + '_concat_conv')(x)
    x = BatchNormalization(name=part + '_concat_projection_bn')(x)
    x = Activation(activation='relu', name=part + '_concat_projection_act')(x)

    # Skip Connection
    x = BilinearUpSampling(size=(K.int_shape(skip)[1], K.int_shape(skip)[2]), name=part + '_upsampling')(x)

    skip = Conv2D(48 // divide, (1, 1), use_bias=False, padding='same', name=part + '_feature_conv')(skip)
    skip = BatchNormalization(name=part + '_feature_bn')(skip)
    skip = Activation(activation='relu', name=part + '_feature_act')(skip)

    return x, skip


#수정해야함
# def DeepLab_Encoder(img_input, part='encoder', divide=1, is_instancenorm=False, is_gropnorm=False, is_spectralnorm=False):
#     '''
#     Encoder Module using Modified Xception Baseline
#     '''
#     # Entry flow
#     x = Conv2D(32 // divide, (3, 3), strides=(2, 2), use_bias=False, padding='same', name=part + '_block1_conv1')(
#         img_input)
#     bn = BatchNormalization(name=part + '_block1_conv1_bn1')(x)
#     if is_instancenorm:
#         ins = InstanceNormalization(x)
#         x = Add()([bn,ins])
#     elif is_gropnorm:
#         gn = GroupNormalization(x)
#         x = Add()([bn,gn])
#     elif is_spectralnorm:
#         sn = _ConvSN(x)
#         x = Add()([bn,sn])
#     x = Activation(activation='relu', name=part + '_block1_conv1_act1')(x)
#
#     x = Conv2D_same(x, 64 // divide, part + '_block1_conv2', kernel_size=3, strides=1)
#     bn = BatchNormalization(name=part + '_block1_conv2_bn2')(x)
#     if is_instancenorm:
#         ins = InstanceNormalization(x)
#         x = Add()([bn,ins])
#     elif is_gropnorm:
#         gn = GroupNormalization(x)
#         x = Add()([bn,gn])
#     elif is_spectralnorm:
#         sn = _ConvSN(x)
#         x = Add()([bn,sn])
#     x = Activation(activation='relu', name=part + '_block1_conv1_act2')(x)
#
#     x = xception_block(x, [128 // divide, 128 // divide, 128 // divide], 2, 1, part + '_block2', mode='conv')
#     x, skip = xception_block(x, [256 // divide, 256 // divide, 256 // divide], 2, 1, part + '_block3', True,
#                              mode='conv')
#     x = xception_block(x, [728 // divide, 728 // divide, 728 // divide], 2, 1, part + '_block4', mode='conv')
#
#     # Middle flow
#     for i in range(16):
#         part_ = part + '_block' + str(i + 5)
#         x = xception_block(x, [728 // divide, 728 // divide, 728 // divide], 1, 1, part_, mode='sum')
#
#     # Exit flow
#     x = xception_block(x, [728 // divide, 1024 // divide, 1024 // divide], 1, 1, part + '_block21', mode='conv')
#     x = xception_block(x, [1536 // divide, 1536 // divide, 2048 // divide], 1, 2, part + '_block22', mode='none',
#                        last_activation=True)
#
#     '''
#     Encoder Module using Atrous Spatial Pyramid Pooling(ASPP) in DeepLabv3 and ParseNet
#     '''
#     b0 = Conv2D(256 // divide, (1, 1), use_bias=False, padding='same', name=part + '_aspp0')(x)
#     bn = BatchNormalization(name=part + '_aspp0_bn')(b0)
#     if is_instancenorm:
#         ins = InstanceNormalization(b0)
#         b0 = Add()([bn,ins])
#     elif is_gropnorm:
#         gn = GroupNormalization(b0)
#         b0 = Add()([bn,gn])
#     elif is_spectralnorm:
#         sn = _ConvSN(b0)
#         b0 = Add()([bn,sn])
#     b0 = Activation(activation='relu', name=part + '_aspp0_act')(b0)
#
#     b1 = SeparableConvBlock(x, 256 // divide, part=part + '_aspp1', rate=6, last_activation=True)
#     b2 = SeparableConvBlock(x, 256 // divide, part=part + '_aspp2', rate=12, last_activation=True)
#     b3 = SeparableConvBlock(x, 256 // divide, part=part + '_aspp3', rate=18, last_activation=True)
#
#     b4 = AveragePooling2D(pool_size=(K.int_shape(x)[1], K.int_shape(x)[2]), name=part + '_image_pooling')(x)
#     b4 = Conv2D(256 // divide, (1, 1), use_bias=False, padding='same', name=part + '_image_pooling_conv')(b4)
#     bn = BatchNormalization(name=part + '_image_pooling_bn')(b4)
#     if is_instancenorm:
#         ins = InstanceNormalization(b4)
#         b4 = Add()([bn,ins])
#     if is_gropnorm:
#         gn = GroupNormalization(b4)
#         b4 = Add()([bn,gn])
#     if is_spectralnorm:
#         sn = _ConvSN(b4)
#         b4 = Add()([bn,sn])
#     b4 = Activation(activation='relu', name=part + '_image_pooling_act')(b4)
#     b4 = BilinearUpSampling(size=(K.int_shape(x)[1], K.int_shape(x)[2]), name=part + '_image_pooling_unpool')(b4)
#
#     x = Concatenate()([b0, b1, b2, b3, b4])
#     x = Conv2D(256 // divide, (1, 1), use_bias=False, padding='same', name=part + '_concat_conv')(x)
#     bn = BatchNormalization(name=part + '_concat_projection_bn')(x)
#     if is_instancenorm:
#         ins = InstanceNormalization(x)
#         x = Add()([bn,ins])
#     if is_gropnorm:
#         gn = GroupNormalization(x)
#         x = Add()([bn,gn])
#     if is_spectralnorm:
#         sn = _ConvSN(x)
#         x = Add()([bn,sn])
#     x = Activation(activation='relu', name=part + '_concat_projection_act')(x)
#
#     # Skip Connection
#     x = BilinearUpSampling(size=(K.int_shape(skip)[1], K.int_shape(skip)[2]), name=part + '_upsampling')(x)
#
#     skip = Conv2D(48 // divide, (1, 1), use_bias=False, padding='same', name=part + '_feature_conv')(skip)
#     bn = BatchNormalization(name=part + '_feature_bn')(skip)
#     if is_instancenorm:
#         ins = InstanceNormalization(skip)
#         skip = Add()([bn,ins])
#     if is_gropnorm:
#         gn = GroupNormalization(skip)
#         skip = Add()([bn,gn])
#     if is_spectralnorm:
#         sn = _ConvSN(skip)
#         skip = Add()([bn,sn])
#     skip = Activation(activation='relu', name=part + '_feature_act')(skip)
#
#     return x, skip


def DeepLab_Decoder(img_input, input_shape, classes=21, part='decoder', divide=1):
    '''
    Decoder Module for Semantic Segmentation
    '''
    x = SeparableConvBlock(img_input, 256 // divide, part=part + '_conv0', last_activation=True)
    x = SeparableConvBlock(x, 256 // divide, part=part + '_conv1', last_activation=True)
    x = Conv2D(classes, (1, 1), padding='same', name=part + '_logits')(x)
    x = BilinearUpSampling(size=(input_shape[0], input_shape[1]), name=part + '_upsampling')(x)

    return x


# def DeepLabv3plus(input_shape=(None, None, 3), classes=21):
#     img_input = Input(shape=input_shape)
#
#     encoder, skip = DeepLab_Encoder(img_input, part='encoder')
#     x = Concatenate(name='concat_encoder_skip')([encoder, skip])
#     x = DeepLab_Decoder(x, input_shape, classes, part='decoder')
#     x = Reshape((-1, classes))(x)
#     logits = Activation(activation='softmax', name='final_logits')(x)
#
#     model = Model(img_input, logits, name='DeepLabv3plus')
#
#     return model

def DeepLabv3plus(input_shape=(None, None, 3), classes=21,name='DeepLabv3plus'):
    img_input = Input(shape=input_shape)
    if name == 'DeepLabv3plus':
        encoder, skip = DeepLab_Encoder(img_input, part='encoder', is_instancenorm=False, is_gropnorm=False, is_spectralnorm=False)
    elif name == 'DeepLabv3plusBIN':
        encoder, skip = DeepLab_Encoder(img_input, part='encoder', is_instancenorm=True, is_gropnorm=False, is_spectralnorm=False)
    elif name == 'DeepLabv3plusGN':
        encoder, skip = DeepLab_Encoder(img_input, part='encoder', is_instancenorm=False, is_gropnorm=True, is_spectralnorm=False)
    elif name == 'DeepLabv3plusSN':
        encoder, skip = DeepLab_Encoder(img_input, part='encoder', is_instancenorm=False, is_gropnorm=True, is_spectralnorm=True)
    x = Concatenate(name='concat_encoder_skip')([encoder, skip])
    x = DeepLab_Decoder(x, input_shape, classes, part='decoder')
    x = Reshape((-1, classes))(x)
    logits = Activation(activation='softmax', name='final_logits')(x)
    model = Model(img_input, logits, name=name)

    return model

def Encoder_Model(img_input, part, divide=1):
    e, es = DeepLab_Encoder(img_input, part, divide)
    encoder_model = Model(img_input, e, name=part)
    skip_model = Model(img_input, es, name=part + '_skip')
    return encoder_model, skip_model


def Decoder_Model(img_input, input_shape, classes, part, divide):
    d = DeepLab_Decoder(img_input, input_shape, classes, part, divide)
    decoder_model = Model(img_input, d, name=part)
    return decoder_model


def Adversarial_1(input_shape=(None, None, 3), classes=21, divide=1, include_z=True):
    '''
    Input : Image, Mask, Z-noise based standard normal distribution
    Architecture : Generator(Encoder + Mask + Generator -> Decoder) + Discriminator(Encoder + Mask + Generator Feature map)
    Generator Output : Generated Mask
    Discriminator Output : Whether Real or Generation
    '''

    total_input = Input(shape=input_shape)
    img_input = Input(shape=input_shape, name='img_input')
    mask_input = Input(shape=input_shape, name='mask_input')
    z_noise = Input(shape=input_shape, name='z_noise_input')

    # Encoder
    encoder, encoder_skip = Encoder_Model(total_input, 'encoder', divide)
    encoder_output = encoder(img_input)
    encoder_skip_output = encoder_skip(img_input)

    # Mask
    mask_output = encoder(mask_input)

    # Generator in Generator
    g, gs = DeepLab_Encoder(z_noise, 'generator', divide)

    total_concat = Concatenate()([mask_output, gs, encoder_skip_output, encoder_output, g])

    # Decoder
    decoder = DeepLab_Decoder(total_concat, input_shape, classes, 'decoder', divide)
    pred = Reshape((-1, classes))(decoder)
    pred = Activation(activation='softmax', name='generator_output')(pred)

    generator = Model([img_input, mask_input, z_noise], pred, name='generator')

    total_concat_conv = Conv2D(K.int_shape(mask_output)[-1], (1, 1), use_bias=False, padding='same',
                               name='totconcat_conv')(total_concat)
    print('total concat conv :', K.int_shape(total_concat_conv))

    # Discriminator
    disc_input = Input(shape=K.int_shape(total_concat_conv)[1:])
    disc, _ = DeepLab_Encoder(disc_input, 'discriminator', divide)
    clsf = GlobalAveragePooling2D()(disc)
    clsf = Dense(1)(clsf)
    clsf = Activation(activation='sigmoid', name='discriminator_output')(clsf)
    discriminator = Model(disc_input, clsf, name='discriminator')

    discriminator.trainable = False

    inputs = [img_input, mask_input, z_noise]
    outputs = [g, pred, discriminator(total_concat_conv)]

    model = Model(inputs, outputs)

    return {
        'generator': generator,
        'discriminator': discriminator,
        'model': model
    }


def Adversarial_2(input_shape=(None, None, 3), classes=21, divide=1, include_z=True):
    '''
    Input : Image, Mask, Z-noise based standard normal distribution
    Architecture : Generator(Encoder + Mask + Generator -> Decoder) + Discriminator(Generated Mask)
    Generator Output : Generated Mask
    Discriminator Output : Whether Real or Generation
    '''

    total_input = Input(shape=input_shape)
    img_input = Input(shape=input_shape, name='img_input')
    mask_input = Input(shape=input_shape, name='mask_input')
    z_noise = Input(shape=input_shape, name='z_noise_input')

    # Encoder
    encoder, encoder_skip = Encoder_Model(total_input, 'encoder', divide)
    encoder_output = encoder(img_input)
    encoder_skip_output = encoder_skip(img_input)

    # Mask
    mask_output = encoder(mask_input)

    # Generator in Generator
    g, gs = DeepLab_Encoder(z_noise, 'generator', divide)

    total_concat = Concatenate()([mask_output, gs, encoder_skip_output, encoder_output, g])

    # Decoder
    decoder = DeepLab_Decoder(total_concat, input_shape, classes, 'decoder', divide)
    pred = Reshape((-1, classes))(decoder)
    pred = Activation(activation='softmax')(pred)
    generator = Model([img_input, mask_input, z_noise], pred, name='generator')

    _to_disc = Reshape((input_shape[0], input_shape[1], classes))(pred)
    _to_disc = Lambda(lambda x: K.cast(K.expand_dims(K.argmax(x)), dtype=pred.dtype))(_to_disc)

    # Discriminator
    disc_input = Input(shape=K.int_shape(_to_disc)[1:])
    disc, _ = DeepLab_Encoder(disc_input, 'discriminator', divide)
    clsf = GlobalAveragePooling2D()(disc)
    clsf = Dense(1)(clsf)
    clsf = Activation(activation='sigmoid')(clsf)
    discriminator = Model(disc_input, clsf, name='discriminator')

    discriminator.trainable = False

    inputs = [img_input, mask_input, z_noise]
    outputs = [pred, discriminator(_to_disc)]

    model = Model(inputs, outputs)

    return {
        'generator': generator,
        'discriminator': discriminator,
        'model': model
    }


class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''

    def __init__(self, c=1):
        self.c = c

    def __call__(self, p):
        return K.clip(p, 0, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}


class Weighted_Add(_Merge):
    def build(self, input_shape):
        super(Weighted_Add, self).build(input_shape)
        dim = (input_shape[0][-1],)
        self.rho = self.add_weight(name='kernel',
                                   shape=dim,
                                   initializer=keras.initializers.RandomUniform(0., 1.),
                                   constraint=WeightClip(1))

    def _merge_function(self, inputs):
        self.rho = K.clip(self.rho, 0, 1)
        bn = inputs[0] * self.rho
        ins = inputs[1] * (1 - self.rho)
        return (bn + ins) / 2





def Unet_Attention_BIN(input_shape=(None, None, 3), classes=21):
    def expend_as(tensor, rep):
        my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)
        return my_repeat

    def UnetGatingSignal(input):
        shape = K.int_shape(input)
        # print('gating signal >>>>> ', shape)
        x = Conv2D(shape[3] * 2, (1, 1), strides=(1, 1), padding="same")(input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def AttentionGate(x, g, inter_shape):
        '''
        x : input
        g : gate
        '''
        shape_x = K.int_shape(x)
        shape_g = K.int_shape(g)
        theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)
        shape_theta_x = K.int_shape(theta_x)
        # print('shape_theta_x >>>>> ', shape_theta_x)

        phi_g = Conv2D(inter_shape, (1, 1), padding='same', bias_initializer='glorot_uniform')(g)
        # print('phi_g >>>>> ', phi_g)
        upsample_g = Conv2DTranspose(inter_shape, (3, 3),
                                     strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                     padding='same', bias_initializer='glorot_uniform')(phi_g)  # 16
        # print('upsample_g >>>>> ', upsample_g)

        concat_xg = Add()([upsample_g, theta_x])
        act_xg = Activation('relu')(concat_xg)
        psi = Conv2D(1, (1, 1), padding='same', bias_initializer='glorot_uniform')(act_xg)
        sigmoid_xg = Activation('sigmoid')(psi)
        shape_sigmoid = K.int_shape(sigmoid_xg)
        upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(
            sigmoid_xg)  # 32

        # my_repeat=Lambda(lambda xinput:K.repeat_elements(xinput[0],shape_x[1],axis=1))
        # upsample_psi=my_repeat([upsample_psi])
        upsample_psi = expend_as(upsample_psi, shape_x[3])
        y = Multiply()([upsample_psi, x])

        # print(K.is_keras_tensor(upsample_psi))

        result = Conv2D(shape_x[3], (1, 1), padding='same')(y)
        result_bn = BatchNormalization()(result)
        return result_bn

    img_input = Input(shape=input_shape)
    divide = 0.1
    # Block 1
    c1 = UnetConv2D(img_input, int(32 // divide), True, True)
    p1 = MaxPooling2D(pool_size=(2, 2), strides=2)(c1)

    # Block 2
    c2 = UnetConv2D(p1, int(64 // divide), True, True)
    p2 = MaxPooling2D(pool_size=(2, 2), strides=2)(c2)

    # Block 3
    c3 = UnetConv2D(p2, int(128 // divide), True, True)
    p3 = MaxPooling2D(pool_size=(2, 2), strides=2)(c3)

    # Block 4
    c4 = UnetConv2D(p3, int(256 // divide), True, True)
    p4 = MaxPooling2D(pool_size=(2, 2), strides=2)(c4)

    # Block 5
    c5 = UnetConv2D(p4, int(256 // divide), True, True)

    # Block 6
    g6 = UnetGatingSignal(c5)
    att6 = AttentionGate(c4, g6, int(256 // divide))
    up6 = Conv2DTranspose(int(256 // divide), (3, 3), activation='relu', strides=(2, 2), padding='same')(c5)
    up6 = Concatenate()([up6, att6])
    c6 = UnetConv2D(up6, int(256 // divide), True, True)

    # Block 7
    g7 = UnetGatingSignal(c6)
    att7 = AttentionGate(c3, g7, int(128 // divide))
    up7 = Conv2DTranspose(int(128 // divide), (3, 3), activation='relu', strides=(2, 2), padding='same')(c6)
    up7 = Concatenate()([up7, att7])
    c7 = UnetConv2D(up7, int(128 // divide), True, True)

    # Block 8
    g8 = UnetGatingSignal(c7)
    att8 = AttentionGate(c2, g8, int(64 // divide))
    up8 = Conv2DTranspose(int(64 // divide), (3, 3), activation='relu', strides=(2, 2), padding='same')(c7)
    up8 = Concatenate()([up8, att8])
    c8 = UnetConv2D(up8, int(64 // divide), True, True)

    # Block 9
    g9 = UnetGatingSignal(c8)
    att9 = AttentionGate(c1, g9, int(32 // divide))
    up9 = Conv2DTranspose(int(32 // divide), (3, 3), activation='relu', strides=(2, 2), padding='same')(c8)
    up9 = Concatenate()([up9, att9])
    c9 = UnetConv2D(up9, int(32 // divide), True, True)

    c10 = Conv2D(classes, (1, 1))(c9)
    r10 = Reshape((-1, classes))(c10)
    logits = Activation('softmax')(r10)

    model = Model(img_input, logits, name='Unet_Attention_BIN')

    return model


def Unet_Attention_BN(input_shape=(None, None, 3), classes=21):
    def expend_as(tensor, rep):
        my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)
        return my_repeat

    def UnetGatingSignal(input):
        shape = K.int_shape(input)
        # print('gating signal >>>>> ', shape)
        x = Conv2D(shape[3] * 2, (1, 1), strides=(1, 1), padding="same")(input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def AttentionGate(x, g, inter_shape):
        '''
        x : input
        g : gate
        '''
        shape_x = K.int_shape(x)
        shape_g = K.int_shape(g)
        theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)
        shape_theta_x = K.int_shape(theta_x)
        # print('shape_theta_x >>>>> ', shape_theta_x)

        phi_g = Conv2D(inter_shape, (1, 1), padding='same', bias_initializer='glorot_uniform')(g)
        # print('phi_g >>>>> ', phi_g)
        upsample_g = Conv2DTranspose(inter_shape, (3, 3),
                                     strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                     padding='same', bias_initializer='glorot_uniform')(phi_g)  # 16
        # print('upsample_g >>>>> ', upsample_g)

        concat_xg = Add()([upsample_g, theta_x])
        act_xg = Activation('relu')(concat_xg)
        psi = Conv2D(1, (1, 1), padding='same', bias_initializer='glorot_uniform')(act_xg)
        sigmoid_xg = Activation('sigmoid')(psi)
        shape_sigmoid = K.int_shape(sigmoid_xg)
        upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(
            sigmoid_xg)  # 32

        # my_repeat=Lambda(lambda xinput:K.repeat_elements(xinput[0],shape_x[1],axis=1))
        # upsample_psi=my_repeat([upsample_psi])
        upsample_psi = expend_as(upsample_psi, shape_x[3])
        y = Multiply()([upsample_psi, x])

        # print(K.is_keras_tensor(upsample_psi))

        result = Conv2D(shape_x[3], (1, 1), padding='same')(y)
        result_bn = BatchNormalization()(result)
        return result_bn

    divide = 1.0
    img_input = Input(shape=input_shape)

    # Block 1
    c1 = UnetConv2D(img_input, int(64 // divide), True)
    p1 = MaxPooling2D(pool_size=(2, 2), strides=2)(c1)

    # Block 2
    c2 = UnetConv2D(p1, int(128 // divide), True)
    p2 = MaxPooling2D(pool_size=(2, 2), strides=2)(c2)

    # Block 3
    c3 = UnetConv2D(p2, int(256 // divide), True)
    p3 = MaxPooling2D(pool_size=(2, 2), strides=2)(c3)

    # Block 4
    c4 = UnetConv2D(p3, int(512 // divide), True)
    p4 = MaxPooling2D(pool_size=(2, 2), strides=2)(c4)

    # Block 5
    c5 = UnetConv2D(p4, int(1024 // divide), True)

    # Block 6
    g6 = UnetGatingSignal(c5)
    att6 = AttentionGate(c4, g6, int(512 // divide))
    up6 = Conv2DTranspose(int(512 // divide), (3, 3), activation='relu', strides=(2, 2), padding='same')(c5)
    up6 = Concatenate()([up6, att6])
    c6 = UnetConv2D(up6, int(512 // divide), True)

    # Block 7
    g7 = UnetGatingSignal(c6)
    att7 = AttentionGate(c3, g7, int(256 // divide))
    up7 = Conv2DTranspose(int(256 // divide), (3, 3), activation='relu', strides=(2, 2), padding='same')(c6)
    up7 = Concatenate()([up7, att7])
    c7 = UnetConv2D(up7, int(256 // divide), True)

    # Block 8
    g8 = UnetGatingSignal(c7)
    att8 = AttentionGate(c2, g8, int(128 // divide))
    up8 = Conv2DTranspose(int(128 // divide), (3, 3), activation='relu', strides=(2, 2), padding='same')(c7)
    up8 = Concatenate()([up8, att8])
    c8 = UnetConv2D(up8, int(128 // divide), True)

    # Block 9
    g9 = UnetGatingSignal(c8)
    att9 = AttentionGate(c1, g9, int(64 // divide))
    up9 = Conv2DTranspose(int(64 // divide), (3, 3), activation='relu', strides=(2, 2), padding='same')(c8)
    up9 = Concatenate()([up9, att9])
    c9 = UnetConv2D(up9, int(64 // divide), True)

    c10 = Conv2D(classes, (1, 1))(c9)
    r10 = Reshape((-1, classes))(c10)
    logits = Activation('softmax')(r10)

    model = Model(img_input, logits, name='Unet_Attention_BN')
    return model


def Unet_Attention(input_shape=(None, None, 3), classes=21):
    def expend_as(tensor, rep):
        my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)
        return my_repeat

    def UnetGatingSignal(input):
        shape = K.int_shape(input)
        # print('gating signal >>>>> ', shape)
        x = Conv2D(shape[3] * 2, (1, 1), strides=(1, 1), padding="same")(input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def AttentionGate(x, g, inter_shape):
        '''
        x : input
        g : gate
        '''
        shape_x = K.int_shape(x)
        shape_g = K.int_shape(g)
        theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)
        shape_theta_x = K.int_shape(theta_x)
        # print('shape_theta_x >>>>> ', shape_theta_x)

        phi_g = Conv2D(inter_shape, (1, 1), padding='same', bias_initializer='glorot_uniform')(g)
        # print('phi_g >>>>> ', phi_g)
        upsample_g = Conv2DTranspose(inter_shape, (3, 3),
                                     strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                     padding='same', bias_initializer='glorot_uniform')(phi_g)  # 16
        # print('upsample_g >>>>> ', upsample_g)

        concat_xg = Add()([upsample_g, theta_x])
        act_xg = Activation('relu')(concat_xg)
        psi = Conv2D(1, (1, 1), padding='same', bias_initializer='glorot_uniform')(act_xg)
        sigmoid_xg = Activation('sigmoid')(psi)
        shape_sigmoid = K.int_shape(sigmoid_xg)
        upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(
            sigmoid_xg)  # 32

        # my_repeat=Lambda(lambda xinput:K.repeat_elements(xinput[0],shape_x[1],axis=1))
        # upsample_psi=my_repeat([upsample_psi])
        upsample_psi = expend_as(upsample_psi, shape_x[3])
        y = Multiply()([upsample_psi, x])

        # print(K.is_keras_tensor(upsample_psi))

        result = Conv2D(shape_x[3], (1, 1), padding='same')(y)
        result_bn = BatchNormalization()(result)
        return result_bn

    img_input = Input(shape=input_shape)

    # Block 1
    c1 = UnetConv2D(img_input, 64)
    p1 = MaxPooling2D(pool_size=(2, 2), strides=2)(c1)

    # Block 2
    c2 = UnetConv2D(p1, 128)
    p2 = MaxPooling2D(pool_size=(2, 2), strides=2)(c2)

    # Block 3
    c3 = UnetConv2D(p2, 256)
    p3 = MaxPooling2D(pool_size=(2, 2), strides=2)(c3)

    # Block 4
    c4 = UnetConv2D(p3, 512)
    p4 = MaxPooling2D(pool_size=(2, 2), strides=2)(c4)

    # Block 5
    c5 = UnetConv2D(p4, 1024)

    # Block 6
    g6 = UnetGatingSignal(c5)
    att6 = AttentionGate(c4, g6, 512)
    up6 = Conv2DTranspose(512, (3, 3), activation='relu', strides=(2, 2), padding='same')(c5)
    up6 = Concatenate()([up6, att6])
    c6 = UnetConv2D(up6, 512)

    # Block 7
    g7 = UnetGatingSignal(c6)
    att7 = AttentionGate(c3, g7, 256)
    up7 = Conv2DTranspose(256, (3, 3), activation='relu', strides=(2, 2), padding='same')(c6)
    up7 = Concatenate()([up7, att7])
    c7 = UnetConv2D(up7, 256)

    # Block 8
    g8 = UnetGatingSignal(c7)
    att8 = AttentionGate(c2, g8, 128)
    up8 = Conv2DTranspose(128, (3, 3), activation='relu', strides=(2, 2), padding='same')(c7)
    up8 = Concatenate()([up8, att8])
    c8 = UnetConv2D(up8, 128)

    # Block 9
    g9 = UnetGatingSignal(c8)
    att9 = AttentionGate(c1, g9, 64)
    up9 = Conv2DTranspose(64, (3, 3), activation='relu', strides=(2, 2), padding='same')(c8)
    up9 = Concatenate()([up9, att9])
    c9 = UnetConv2D(up9, 64)

    c10 = Conv2D(classes, (1, 1))(c9)
    r10 = Reshape((-1, classes))(c10)
    logits = Activation('softmax')(r10)

    model = Model(img_input, logits, name='Unet_Attention')
    return model

def linear_transform(x):
    v1 = tf.Variable(1., name='multiplier')
    v2 = tf.Variable(0., name='bias')
    # tmp = tf.math.divide(input, 64)
    # splits = tf.split(tmp[0].shape,64,x)
    # for i in range(len(splits)):
    #     splits[i]*v1
    # x = tf.concat(0,splits)
    # return splits
    return x*v1

def FCNConv2D(input, outdim, num=2, is_batchnorm=False, is_batchnorm2=False, is_instancenorm=False, is_groupnorm=False, is_layernorm=False, is_spectralnorm=False):
    
    x = Conv2D(outdim, (3, 3), strides=(1, 1), padding="same")(input)
    if is_batchnorm:
        bn = BatchNormalization()(x)
        if is_instancenorm:
            ins = InstanceNormalization()(x)
            x = Weighted_Add()([bn, ins])
        elif is_groupnorm:
            gn = GroupNormalization(groups=32, axis=-1)(x)
            x = Weighted_Add()([bn, gn])
        elif is_layernorm:
            ln = LayerNormalization()(x)
            x = Weighted_Add()([bn, ln])
        elif is_spectralnorm:
            sn = ConvSN2D(outdim, (3, 3), strides=(1, 1), padding="same")(x)
            x = Weighted_Add()([bn, sn])
        elif is_batchnorm2:
            bn2 = BatchNormalization()(x)
            x = Weighted_Add()([bn, bn2])
        else:
            print("If you want to use just Batch normalization. Use parameter 'is_batchnorm2'.")
            return
    else:
        if is_instancenorm:
            x = InstanceNormalization()(x)
        elif is_groupnorm:
            x = GroupNormalization(groups=32, axis=-1)(x)
        elif is_layernorm:
            x = LayerNormalization()(x)
        elif is_spectralnorm:
            x = ConvSN2D(outdim, (3, 3), strides=(1, 1), padding="same")(x)
        elif is_batchnorm2:
            x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(outdim, (3, 3), strides=(1, 1), padding="same")(x)
    if is_batchnorm:
        bn = BatchNormalization()(x)
        if is_instancenorm:
            ins = InstanceNormalization()(x)
            x = Weighted_Add()([bn, ins])
        elif is_groupnorm:
            gn = GroupNormalization(groups=32, axis=-1)(x)
            x = Weighted_Add()([bn, gn])
        elif is_layernorm:
            ln = LayerNormalization()(x)
            x = Weighted_Add()([bn, ln])
        elif is_spectralnorm:
            sn = ConvSN2D(outdim, (3, 3), strides=(1, 1), padding="same")(x)
            x = Weighted_Add()([bn, sn])
        elif is_batchnorm2:
            bn2 = BatchNormalization()(x)
            x = Weighted_Add()([bn, bn2])
    else:
        if is_instancenorm:
            x = InstanceNormalization()(x)
        elif is_groupnorm:
            x = GroupNormalization(groups=32, axis=-1)(x)
        elif is_layernorm:
            x = LayerNormalization()(x)
        elif is_spectralnorm:
            x = ConvSN2D(outdim, (3, 3), strides=(1, 1), padding="same")(x)
        elif is_batchnorm2:
            x = BatchNormalization()(x)
    x = Activation('relu')(x)

    if num==3:
        x = Conv2D(outdim, (3, 3), strides=(1, 1), padding="same")(x)
        if is_batchnorm:
            bn = BatchNormalization()(x)
            if is_instancenorm:
                ins = InstanceNormalization()(x)
                x = Weighted_Add()([bn, ins])
            elif is_groupnorm:
                gn = GroupNormalization(groups=32, axis=-1)(x)
                x = Weighted_Add()([bn, gn])
            elif is_layernorm:
                ln = LayerNormalization()(x)
                x = Weighted_Add()([bn, ln])
            elif is_spectralnorm:
                sn = ConvSN2D(outdim, (3, 3), strides=(1, 1), padding="same")(x)
                x = Weighted_Add()([bn, sn])
            elif is_batchnorm2:
                bn2 = BatchNormalization()(x)
                x = Weighted_Add()([bn, bn2])
        else:
            if is_instancenorm:
                x = InstanceNormalization()(x)
            elif is_groupnorm:
                x = GroupNormalization(groups=32, axis=-1)(x)
            elif is_layernorm:
                x = LayerNormalization()(x)
            elif is_spectralnorm:
                x = ConvSN2D(outdim, (3, 3), strides=(1, 1), padding="same")(x)
            elif is_batchnorm2:
                x = BatchNormalization()(x)
        x = Activation('relu')(x)
    return x

def FCNModel(input_shape=(None, None, 3), classes=21, is_batchnorm=False, is_batchnorm2=False, is_instancenorm=False, is_groupnorm=False, is_layernorm=False,
              is_spectralnorm=False):
    img_input = Input(shape=input_shape)
    # Block 1
    c1 = FCNConv2D(img_input, 64, 2, is_batchnorm, is_batchnorm2, is_instancenorm, is_groupnorm, is_layernorm, is_spectralnorm)
    p1 = MaxPooling2D(pool_size=(2, 2), strides=2)(c1)

    # Block 2
    c2 = FCNConv2D(p1, 128, 2, is_batchnorm, is_batchnorm2, is_instancenorm, is_groupnorm, is_layernorm, is_spectralnorm)
    p2 = MaxPooling2D(pool_size=(2, 2), strides=2)(c2)

    # Block 3
    c3 = FCNConv2D(p2, 256, 3, is_batchnorm, is_batchnorm2, is_instancenorm, is_groupnorm, is_layernorm, is_spectralnorm)
    p3 = MaxPooling2D(pool_size=(2, 2), strides=2)(c3)

    # Block 4
    c4 = FCNConv2D(p3, 512, 3, is_batchnorm, is_batchnorm2, is_instancenorm, is_groupnorm, is_layernorm, is_spectralnorm)
    p4 = MaxPooling2D(pool_size=(2, 2), strides=2)(c4)

    # Block 5
    c5 = FCNConv2D(p4, 512, 3, is_batchnorm, is_batchnorm2, is_instancenorm, is_groupnorm, is_layernorm, is_spectralnorm)
    p5 = MaxPooling2D(pool_size=(2, 2), strides=2)(c5)

    # Convolutional layers transfered from fully-connected layers
    fc6 = Conv2D(4096, (7, 7), activation='relu', padding='same', bias_initializer='glorot_uniform')(p5)
    d6 = Dropout(0.5)(fc6)
    fc6 = Conv2D(4096, (1, 1), activation='relu', padding='same', bias_initializer='glorot_uniform')(d6)
    d6 = Dropout(0.5)(fc6)

    # Classifying layer
    c7 = Conv2D(classes, (1, 1), padding='same', bias_initializer='glorot_uniform')(d6)

    # upscale to actual image size
    # print(K.int_shape(p4))
    c8 = Conv2DTranspose(classes, (4, 4), strides=(2, 2), padding='same')(c7)
    p4 = Conv2D(classes, (1, 1), padding='same')(p4)
    f8 = Add()([c8, p4])

    c9 = Conv2DTranspose(classes, (4, 4), strides=(2, 2), padding='same')(f8)
    p3 = Conv2D(classes, (1, 1), padding='same')(p3)
    f9 = Add()([c9, p3])

    c10 = Conv2DTranspose(classes, (16, 16), strides=(8, 8), padding='same')(f9)
    r10 = Reshape((-1, classes))(c10)
    logits = Activation(activation='softmax')(r10)
    if is_batchnorm:
        if is_instancenorm:
            modelName = 'FCNBIN'
        elif is_groupnorm:
            modelName = 'FCNBGN'
        elif is_layernorm:
            modelName = 'FCNBLN'
        elif is_spectralnorm:
            modelName = 'FCNBSN'
        elif is_batchnorm2:
            modelName = 'FCNBBN'
    else:
        modelName = 'FCN'
        if is_instancenorm:
            modelName = 'FCNIN'
        elif is_groupnorm:
            modelName = 'FCNGN'
        elif is_layernorm:
            modelName = 'FCNLN'
        elif is_spectralnorm:
            modelName = 'FCNSN'
        elif is_batchnorm2:
            modelName = 'FCNBN'
    model = Model(img_input, logits, name=modelName)

    return model


def UnetConv2D(input, outdim, is_batchnorm=False, is_batchnorm2=False, is_instancenorm=False, is_groupnorm=False, is_layernorm=False, is_spectralnorm=False):
    x = Conv2D(outdim, (3, 3), strides=(1, 1), padding="same")(input)
    if is_batchnorm:
        bn = BatchNormalization()(x)
        if is_instancenorm:
            ins = InstanceNormalization()(x)
            x = Weighted_Add()([bn, ins])
        elif is_groupnorm:
            gn = GroupNormalization(32)(x)
            x = Weighted_Add()([bn, gn])
        elif is_layernorm:
            ln = LayerNormalization()(x)
            x = Weighted_Add()([bn, ln])
        elif is_spectralnorm:
            sn = ConvSN2D(outdim, (3,3), strides=(1,1), padding='same')(x)
            x = Weighted_Add()([bn, sn])
        elif is_batchnorm2:
            bn2 = BatchNormalization()(x)
            x = Weighted_Add()([bn, bn2])
        else:
            print("If you want to use just Batch normalization. Use parameter 'is_batchnorm2'.")
            return
    else:
        if is_instancenorm:
            x = InstanceNormalization()(x)
        elif is_groupnorm:
            x = GroupNormalization(groups=32, axis=-1)(x)
        elif is_layernorm:
            x = LayerNormalization()(x)
        elif is_spectralnorm:
            x = ConvSN2D(outdim, (3, 3), strides=(1, 1), padding="same")(x)
        elif is_batchnorm2:
            x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(outdim, (3, 3), strides=(1, 1), padding="same")(x)
    if is_batchnorm:
        bn = BatchNormalization()(x)
        if is_instancenorm:
            ins = InstanceNormalization()(x)
            x = Weighted_Add()([bn, ins])
        elif is_groupnorm:
            gn = GroupNormalization(32)(x)
            x = Weighted_Add()([bn, gn])
        elif is_layernorm:
            ln = LayerNormalization()(x)
            x = Weighted_Add()([bn, ln])
        elif is_spectralnorm:
            sn = ConvSN2D(outdim, (3,3), strides=(1,1), padding='same')(x)
            x = Weighted_Add()([bn, sn])
        elif is_batchnorm2:
            bn2 = BatchNormalization()(x)
            x = Weighted_Add()([bn, bn2])
        else:
            print("If you want to use just Batch normalization. Use parameter 'is_batchnorm2'.")
            return
    else:
        if is_instancenorm:
            x = InstanceNormalization()(x)
        elif is_groupnorm:
            x = GroupNormalization(groups=32, axis=-1)(x)
        elif is_layernorm:
            x = LayerNormalization()(x)
        elif is_spectralnorm:
            x = ConvSN2D(outdim, (3, 3), strides=(1, 1), padding="same")(x)
        elif is_batchnorm2:
            x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x



def UnetModel(input_shape=(None, None, 3), classes=21, is_batchnorm=False, is_batchnorm2=False, is_instancenorm=False, is_layernorm=False, is_groupnorm=False,
              is_spectralnorm=False):
    img_input = Input(shape=input_shape)
    size = 64
    c1 = UnetConv2D(img_input, size, is_batchnorm, is_batchnorm2, is_instancenorm, is_groupnorm, is_layernorm, is_spectralnorm)
    p1 = MaxPooling2D(pool_size=(2, 2), strides=2)(c1)

    # Block 2
    c2 = UnetConv2D(p1, size * 2, is_batchnorm, is_batchnorm2, is_instancenorm, is_groupnorm, is_layernorm, is_spectralnorm)
    p2 = MaxPooling2D(pool_size=(2, 2), strides=2)(c2)

    # Block 3
    c3 = UnetConv2D(p2, size * 3, is_batchnorm, is_batchnorm2, is_instancenorm, is_groupnorm, is_layernorm, is_spectralnorm)
    p3 = MaxPooling2D(pool_size=(2, 2), strides=2)(c3)

    # Block 4
    c4 = UnetConv2D(p3, size * 4, is_batchnorm, is_batchnorm2, is_instancenorm, is_groupnorm, is_layernorm, is_spectralnorm)
    p4 = MaxPooling2D(pool_size=(2, 2), strides=2)(c4)

    # Block 5
    c5 = UnetConv2D(p4, size * 5, is_batchnorm, is_batchnorm2, is_instancenorm, is_groupnorm, is_layernorm, is_spectralnorm)

    # Block 6
    u6 = Conv2DTranspose(size * 4, (2, 2), activation='relu', strides=(2, 2), padding='same')(c5)
    u6 = Concatenate()([u6, c4])
    c6 = UnetConv2D(u6, size * 4, is_batchnorm, is_batchnorm2, is_instancenorm, is_groupnorm, is_layernorm, is_spectralnorm)

    # Block 7
    u7 = Conv2DTranspose(size * 3, (2, 2), activation='relu', strides=(2, 2), padding='same')(c6)
    u7 = Concatenate()([u7, c3])
    c7 = UnetConv2D(u7, size * 3, is_batchnorm, is_batchnorm2, is_instancenorm, is_groupnorm, is_layernorm, is_spectralnorm)

    # Block 8
    u8 = Conv2DTranspose(size * 2, (2, 2), activation='relu', strides=(2, 2), padding='same')(c7)
    u8 = Concatenate()([u8, c2])
    c8 = UnetConv2D(u8, size * 2, is_batchnorm, is_batchnorm2, is_instancenorm, is_groupnorm, is_layernorm, is_spectralnorm)

    # Block 9
    u9 = Conv2DTranspose(size, (2, 2), activation='relu', strides=(2, 2), padding='same')(c8)
    u9 = Concatenate()([u9, c1])
    c9 = UnetConv2D(u9, 64, is_batchnorm, is_batchnorm2, is_instancenorm, is_groupnorm, is_layernorm, is_spectralnorm)

    c10 = Conv2D(classes, (1, 1))(c9)
    r10 = Reshape((-1, classes))(c10)
    logits = Activation('softmax')(r10)

    if is_batchnorm:
        if is_instancenorm:
            modelName = 'UnetBIN'
        elif is_groupnorm:
            modelName = 'UnetBGN'
        elif is_layernorm:
            modelName = 'UnetBLN'
        elif is_spectralnorm:
            modelName = 'UnetBSN'
        elif is_batchnorm2:
            modelName = 'UnetBBN'
    else:
        modelName = 'Unet'
        if is_instancenorm:
            modelName = 'UnetIN'
        elif is_groupnorm:
            modelName = 'UnetGN'
        elif is_layernorm:
            modelName = 'UnetLN'
        elif is_spectralnorm:
            modelName = 'UnetSN'
        elif is_batchnorm:
            modelName = 'UnetBN'
    model = Model(img_input, logits, name=modelName)
    return model

def Vgg16Conv2d(input, outdim, is_batchnorm=False, is_instancenorm=False, is_layernorm=False, is_groupnorm=False, is_spectralnorm=False):
    x = Conv2D(outdim, (3, 3), strides=(1, 1), padding="same")(input)
    if is_batchnorm:
        x = BatchNormalization()(x)
    elif is_instancenorm:
        x = InstanceNormalization()(x)
    elif is_layernorm:
        x = LayerNormalization()(x)
    elif is_groupnorm:
        x = GroupNormalization(groups=32, axis=-1)(x)
    elif is_spectralnorm:
        x = ConvSN2D(outdim, (3, 3), strides=(1, 1), padding="same")(x)
    x = Activation('relu')(x)
    return x


def Vgg16Model(input_shape=(None, None, 3), classes=10, is_batchnorm=False, is_instancenorm=False, is_layernorm=False, is_groupnorm=False,
              is_spectralnorm=False):
    img_input = Input(shape=input_shape)

    c1 = Vgg16Conv2d(img_input, 64, is_batchnorm, is_instancenorm, is_layernorm, is_groupnorm, is_spectralnorm)
    c1 = Vgg16Conv2d(c1,64, is_batchnorm, is_instancenorm, is_layernorm, is_groupnorm, is_spectralnorm)
    p1 = MaxPooling2D(pool_size=(2, 2), strides=2)(c1)

    c2 = Vgg16Conv2d(p1, 128, is_batchnorm, is_instancenorm, is_layernorm, is_groupnorm, is_spectralnorm)
    c2 = Vgg16Conv2d(c2, 128, is_batchnorm, is_instancenorm, is_layernorm, is_groupnorm, is_spectralnorm)
    p2 = MaxPooling2D(pool_size=(2, 2), strides=2)(c2)

    c3 = Vgg16Conv2d(p2, 256, is_batchnorm, is_instancenorm, is_layernorm, is_groupnorm, is_spectralnorm)
    c3 = Vgg16Conv2d(c3, 256, is_batchnorm, is_instancenorm, is_layernorm, is_groupnorm, is_spectralnorm)
    c3 = Vgg16Conv2d(c3, 256, is_batchnorm, is_instancenorm, is_layernorm, is_groupnorm, is_spectralnorm)
    p3 = MaxPooling2D(pool_size=(2, 2), strides=2)(c3)

    c4 = Vgg16Conv2d(p3, 512, is_batchnorm, is_instancenorm, is_layernorm, is_groupnorm, is_spectralnorm)
    c4 = Vgg16Conv2d(c4, 512, is_batchnorm, is_instancenorm, is_layernorm, is_groupnorm, is_spectralnorm)
    c4 = Vgg16Conv2d(c4, 512, is_batchnorm, is_instancenorm, is_layernorm, is_groupnorm, is_spectralnorm)
    p4 = MaxPooling2D(pool_size=(2, 2), strides=2)(c4)

    c5 = Vgg16Conv2d(p4, 512, is_batchnorm, is_instancenorm, is_layernorm, is_groupnorm, is_spectralnorm)
    c5 = Vgg16Conv2d(c5, 512, is_batchnorm, is_instancenorm, is_layernorm, is_groupnorm, is_spectralnorm)
    c5 = Vgg16Conv2d(c5, 512, is_batchnorm, is_instancenorm, is_layernorm, is_groupnorm, is_spectralnorm)
    p5 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(c5)

    f = Flatten()(p5)

    d1 = Dense(4096)(f)
    d1 = Activation(activation='relu')(d1)
    d2 = Dense(4096)(d1)
    d2 = Activation(activation='relu')(d2)
    d3 = Dense(classes)(d2)
    logits = Activation(activation='softmax')(d3)


    modelName = 'Vgg16'
    if is_batchnorm:
        modelName = 'Vgg16BN'
    elif is_instancenorm:
        modelName = 'Vgg16IN'
    elif is_groupnorm:
        modelName = 'Vgg16GN'
    elif is_layernorm:
        modelName = 'Vgg16LN'
    elif is_spectralnorm:
        modelName = 'Vgg16SN'

    model = Model(img_input, logits, name=modelName)
    model.summary()
    return model

def conv_block(input, outdim, strides=(2,2), is_batchnorm=False, is_instancenorm=False, is_layernorm=False, is_groupnorm=False,
              is_spectralnorm=False):

    outdim1, outdim2, outdim3 = outdim

    x = Conv2D(outdim1, (1,1), strides=strides)(input)
    if is_batchnorm:
        x = BatchNormalization()(x)
    if is_instancenorm:
        x = InstanceNormalization()(x)
    if is_layernorm:
        x = LayerNormalization()(x)
    if is_groupnorm:
        x = GroupNormalization(groups=32, axis=-1)(x)
    if is_spectralnorm:
        x = ConvSN2D(outdim1, (3, 3), strides=(1, 1), padding="same")(x)
    x = Activation('relu')(x)

    x = Conv2D(outdim2, (3,3), strides=(1,1), padding='same')(x)
    if is_batchnorm:
        x = BatchNormalization()(x)
    if is_instancenorm:
        x = InstanceNormalization()(x)
    if is_layernorm:
        x = LayerNormalization()(x)
    if is_groupnorm:
        x = GroupNormalization(groups=32, axis=-1)(x)
    if is_spectralnorm:
        x = ConvSN2D(outdim2, (3, 3), strides=(1, 1), padding="same")(x)
    x = Activation('relu')(x)

    x = Conv2D(outdim3, (1,1), strides=(1,1))(x)
    if is_batchnorm:
        x = BatchNormalization()(x)
    if is_instancenorm:
        x = InstanceNormalization()(x)
    if is_layernorm:
        x = LayerNormalization()(x)
    if is_groupnorm:
        x = GroupNormalization(groups=32, axis=-1)(x)
    if is_spectralnorm:
        x = ConvSN2D(outdim3, (3, 3), strides=(1, 1), padding="same")(x)

    shortcut = Conv2D(outdim3, (1,1), strides=strides)(input)
    if is_batchnorm:
        shortcut = BatchNormalization()(shortcut)
    if is_instancenorm:
        shortcut = InstanceNormalization()(shortcut)
    if is_layernorm:
        shortcut = LayerNormalization()(shortcut)
    if is_groupnorm:
        shortcut = GroupNormalization(groups=32, axis=-1)(shortcut)
    if is_spectralnorm:
        shortcut = ConvSN2D(outdim3, (3, 3), strides=(1, 1), padding="same")(shortcut)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def identity_block(input, outdim, is_batchnorm=False, is_instancenorm=False, is_layernorm=False, is_groupnorm=False,
              is_spectralnorm=False):
    outdim1, outdim2, outdim3 = outdim

    x = Conv2D(outdim1, (1,1), strides=(1,1))(input)
    if is_batchnorm:
        x = BatchNormalization()(x)
    if is_instancenorm:
        x = InstanceNormalization()(x)
    if is_layernorm:
        x = LayerNormalization()(x)
    if is_groupnorm:
        x = GroupNormalization(groups=32, axis=-1)(x)
    if is_spectralnorm:
        x = ConvSN2D(outdim1, (3, 3), strides=(1, 1), padding="same")(x)
    x = Activation('relu')(x)

    x = Conv2D(outdim2, (3,3), strides=(1,1), padding='same')(x)
    if is_batchnorm:
        x = BatchNormalization()(x)
    if is_instancenorm:
        x = InstanceNormalization()(x)
    if is_layernorm:
        x = LayerNormalization()(x)
    if is_groupnorm:
        x = GroupNormalization(groups=32, axis=-1)(x)
    if is_spectralnorm:
        x = ConvSN2D(outdim2, (3, 3), strides=(1, 1), padding="same")(x)
    x = Activation('relu')(x)

    x = Conv2D(outdim3, (1,1), strides=(1,1))(x)
    if is_batchnorm:
        x = BatchNormalization()(x)
    if is_instancenorm:
        x = InstanceNormalization()(x)
    if is_layernorm:
        x = LayerNormalization()(x)
    if is_groupnorm:
        x = GroupNormalization(groups=32, axis=-1)(x)
    if is_spectralnorm:
        x = ConvSN2D(outdim3, (3, 3), strides=(1, 1), padding="same")(x)

    x = Add()([x, input])
    x = Activation('relu')(x)
    return x

def ResNet50(input_shape=(None, None, 3), classes=10, is_batchnorm=False, is_instancenorm=False, is_layernorm=False, is_groupnorm=False,
              is_spectralnorm=False):
    img_input = Input(shape=input_shape)

    x = ZeroPadding2D(padding=(3,3))(img_input)
    x = Conv2D(64, (7,7), strides=(2,2))(x)
    if is_batchnorm:
        x = BatchNormalization()(x)
    if is_instancenorm:
        x = InstanceNormalization()(x)
    if is_layernorm:
        x = LayerNormalization()(x)
    if is_groupnorm:
        x = GroupNormalization(groups=32, axis=-1)(x)
    if is_spectralnorm:
        x = ConvSN2D(64, (3, 3), strides=(1, 1), padding="same")(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1,1))(x)
    x = MaxPooling2D((3,3), strides=(2,2))(x)

    x = conv_block(x, [64,64,256], strides=(1,1),is_batchnorm=is_batchnorm,is_instancenorm=is_instancenorm,is_layernorm=is_layernorm,is_groupnorm=is_groupnorm,
                   is_spectralnorm=is_spectralnorm)
    x = identity_block(x, [64,64,256],is_batchnorm=is_batchnorm,is_instancenorm=is_instancenorm,is_layernorm=is_layernorm,is_groupnorm=is_groupnorm,
                   is_spectralnorm=is_spectralnorm)
    x = identity_block(x, [64,64,256],is_batchnorm=is_batchnorm,is_instancenorm=is_instancenorm,is_layernorm=is_layernorm,is_groupnorm=is_groupnorm,
                   is_spectralnorm=is_spectralnorm)

    x = conv_block(x, [128,128,512],is_batchnorm=is_batchnorm,is_instancenorm=is_instancenorm,is_layernorm=is_layernorm,is_groupnorm=is_groupnorm,
                   is_spectralnorm=is_spectralnorm)
    x = identity_block(x, [128,128,512],is_batchnorm=is_batchnorm,is_instancenorm=is_instancenorm,is_layernorm=is_layernorm,is_groupnorm=is_groupnorm,
                   is_spectralnorm=is_spectralnorm)
    x = identity_block(x, [128,128,512],is_batchnorm=is_batchnorm,is_instancenorm=is_instancenorm,is_layernorm=is_layernorm,is_groupnorm=is_groupnorm,
                   is_spectralnorm=is_spectralnorm)
    x = identity_block(x, [128,128,512],is_batchnorm=is_batchnorm,is_instancenorm=is_instancenorm,is_layernorm=is_layernorm,is_groupnorm=is_groupnorm,
                   is_spectralnorm=is_spectralnorm)

    x = conv_block(x, [256,256,1024],is_batchnorm=is_batchnorm,is_instancenorm=is_instancenorm,is_layernorm=is_layernorm,is_groupnorm=is_groupnorm,
                   is_spectralnorm=is_spectralnorm)
    x = identity_block(x, [256,256,1024],is_batchnorm=is_batchnorm,is_instancenorm=is_instancenorm,is_layernorm=is_layernorm,is_groupnorm=is_groupnorm,
                   is_spectralnorm=is_spectralnorm)
    x = identity_block(x, [256,256,1024],is_batchnorm=is_batchnorm,is_instancenorm=is_instancenorm,is_layernorm=is_layernorm,is_groupnorm=is_groupnorm,
                   is_spectralnorm=is_spectralnorm)
    x = identity_block(x, [256,256,1024],is_batchnorm=is_batchnorm,is_instancenorm=is_instancenorm,is_layernorm=is_layernorm,is_groupnorm=is_groupnorm,
                   is_spectralnorm=is_spectralnorm)
    x = identity_block(x, [256,256,1024],is_batchnorm=is_batchnorm,is_instancenorm=is_instancenorm,is_layernorm=is_layernorm,is_groupnorm=is_groupnorm,
                   is_spectralnorm=is_spectralnorm)
    x = identity_block(x, [256,256,1024],is_batchnorm=is_batchnorm,is_instancenorm=is_instancenorm,is_layernorm=is_layernorm,is_groupnorm=is_groupnorm,
                   is_spectralnorm=is_spectralnorm)

    x = conv_block(x, [512,512,2048],is_batchnorm=is_batchnorm,is_instancenorm=is_instancenorm,is_layernorm=is_layernorm,is_groupnorm=is_groupnorm,
                   is_spectralnorm=is_spectralnorm)
    x = identity_block(x, [512,512,2048],is_batchnorm=is_batchnorm,is_instancenorm=is_instancenorm,is_layernorm=is_layernorm,is_groupnorm=is_groupnorm,
                   is_spectralnorm=is_spectralnorm)
    x = identity_block(x, [512,512,2048],is_batchnorm=is_batchnorm,is_instancenorm=is_instancenorm,is_layernorm=is_layernorm,is_groupnorm=is_groupnorm,
                   is_spectralnorm=is_spectralnorm)

    x = GlobalAveragePooling2D()(x)
    x = Dense(classes)(x)
    logits = Activation('softmax')(x)

    modelName = 'ResNet50'
    if is_batchnorm:
        modelName = 'ResNet50BN'
    elif is_instancenorm:
        modelName = 'ResNet50IN'
    elif is_groupnorm:
        modelName = 'ResNet50GN'
    elif is_layernorm:
        modelName = 'ResNet50LN'
    elif is_spectralnorm:
        modelName = 'ResNet50SN'

    model = Model(img_input, logits, name=modelName)
    return model

class GAN():
    def __init__(self, is_batchnorm=False, is_instancenorm=False, is_layernorm=False, is_groupnorm=False, is_spectralnorm=False):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.is_batchnorm = is_batchnorm
        self.is_instancenorm = is_instancenorm
        self.is_layernorm = is_layernorm
        self.is_groupnorm = is_groupnorm
        self.is_spectralnorm = is_spectralnorm
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = True

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        if self.is_batchnorm:
            model.add(BatchNormalization(momentum=0.8))
        elif self.is_instancenorm:
            model.add(InstanceNormalization())
        elif self.is_instancenorm:
            model.add(InstanceNormalization())
        elif self.is_groupnorm:
            model.add(GroupNormalization(groups=32, axis=-1))
        elif self.is_spectralnorm:
            model.add(ConvSN2D(256, (3, 3), strides=(1, 1), padding="same"))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        if self.is_batchnorm:
            model.add(BatchNormalization(momentum=0.8))
        elif self.is_instancenorm:
            model.add(InstanceNormalization())
        elif self.is_instancenorm:
            model.add(InstanceNormalization())
        elif self.is_groupnorm:
            model.add(GroupNormalization(groups=32, axis=-1))
        elif self.is_spectralnorm:
            model.add(ConvSN2D(512, (3, 3), strides=(1, 1), padding="same"))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        if self.is_batchnorm:
            model.add(BatchNormalization(momentum=0.8))
        elif self.is_instancenorm:
            model.add(InstanceNormalization())
        elif self.is_instancenorm:
            model.add(InstanceNormalization())
        elif self.is_groupnorm:
            model.add(GroupNormalization(groups=32, axis=-1))
        elif self.is_spectralnorm:
            model.add(ConvSN2D(1024, (3, 3), strides=(1, 1), padding="same"))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        g = Model(noise, img, name='generator')
        return g

    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))

        img = Input(shape=self.img_shape)
        validity = model(img)

        d = Model(img, validity)
        d.summary()
        return d

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()
class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        print("====Generator====")
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        print("====Discriminator====")
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()

if __name__ == '__main__':
    # from keras.utils import plot_model
    # model = Adversarial_3((512, 512, 3))
    # model.summary()
    # print(model.layers)
    # model = Adversarial((512, 512, 3))
    # model['model'].summary()
    # plot_model(model['model'], to_file='./Adversarial_33.png', show_shapes=True, show_layer_names=True)

    model = FCNModel()
    model.summary()