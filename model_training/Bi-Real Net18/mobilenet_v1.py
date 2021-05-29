# -*- coding: utf-8 -*-

# mobilenet的一种实现，用于与Bi-Real Net进行比较

from __future__ import print_function
#from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
if tf.__version__<'2.0':
    import keras
else:
    from tensorflow import keras

# [128, 160, 192, 224]:
def MobileNet(input_shape=(224,224,3),alpha=1.0, depth_multiplier=1,
              dropout=1e-3, classes=1000,include_top=True,pooling='avg'):
    if alpha not in [0.25, 0.50, 0.75, 1.0]:
        raise ValueError('If imagenet weights are being loaded, alpha can be one of'
                             '`0.25`, `0.50`, `0.75` or `1.0` only.')

    img_input = keras.layers.Input(shape=input_shape)
    # cifar100 strides=(2, 2)=>(1,1)
    x = _conv_block(img_input, 32, alpha, strides=(1, 1))
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,strides=(1, 1), block_id=2) #strides=(2, 2)=>(1,1)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)

    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)

    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier,strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)

    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier,  strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)

    #if keras.backend.image_data_format() == 'channels_first':
    #    shape = (int(1024 * alpha), 1, 1)
    #else:
    #    shape = (1, 1, int(1024 * alpha))
    if pooling=='avg':
        x = keras.layers.GlobalAveragePooling2D()(x)
    else:
        x=keras.layers.GlobalMaxPooling2D()(x)
    if include_top:
        x=keras.layers.Dense(classes,activation='softmax')(x)

    # x = keras.layers.Reshape(shape, name='reshape_1')(x)
    # x = keras.layers.Dropout(dropout, name='dropout')(x)
    # x = keras.layers.Conv2D(classes, (1, 1), padding='same',kernel_regularizer=keras.regularizers.l2(1e-5), use_bias=False, name='conv_preds')(x)
    # if include_top:
    #     x = keras.layers.Reshape((classes,), name='reshape_2')(x)
    #     x = keras.layers.Activation('softmax', name='act_softmax')(x)
    # else:
    #     x = keras.layers.GlobalAveragePooling2D()(x)
    #elif pooling == 'max':
    #    x = keras.layers.GlobalMaxPooling2D()(x)

    model = keras.models.Model(inputs=img_input, outputs=x, name='mobilenet_%0.2f_%s' % (alpha, input_shape[0]))
    return model


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    
    channel_axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv1_pad')(inputs)
    x = keras.layers.Conv2D(filters, kernel,padding='valid',use_bias=False, strides=strides,kernel_regularizer=keras.regularizers.l2(1e-5),name='conv1')(x)
    x = keras.layers.BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return keras.layers.ReLU(6., name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    channel_axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = keras.layers.ZeroPadding2D(((0, 1), (0, 1)),name='conv_pad_%d' % block_id)(inputs)
    x = keras.layers.DepthwiseConv2D((3, 3),padding='same' if strides == (1, 1) else 'valid',
                               depth_multiplier=depth_multiplier, strides=strides,kernel_regularizer=keras.regularizers.l2(1e-5),
                               use_bias=False,name='conv_dw_%d' % block_id)(x)
    x = keras.layers.BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = keras.layers.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)

    x = keras.layers.Conv2D(pointwise_conv_filters, (1, 1),padding='same',use_bias=False,strides=(1, 1),kernel_regularizer=keras.regularizers.l2(1e-5),name='conv_pw_%d' % block_id)(x)
    x = keras.layers.BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
    return keras.layers.ReLU(6., name='conv_pw_%d_relu' % block_id)(x)

def getMobileNet():
    model=MobileNet(input_shape=(32,32,3),classes=100)
    return model
if __name__ == "__main__":
    #3,331,364
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    model=MobileNet(input_shape=(32,32,3),classes=100)
    model.summary()
    keras.utils.plot_model(model, 'MobileNet-1.0.png', show_shapes=True)