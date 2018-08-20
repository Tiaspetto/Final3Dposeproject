# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import keras.backend as K
from keras.engine import Layer

from keras.layers import Input, Dropout, merge
from keras.layers.convolutional import Convolution2D, UpSampling2D, ZeroPadding2D, Cropping2D, Deconvolution2D
from keras.layers.core import Activation

from keras.applications.resnet50 import ResNet50
from keras.models import Model, load_model
from posenet_2d import *


import numpy as np

class Softmax4D(Layer):
    def __init__(self, axis=-1,**kwargs):
        self.axis=axis
        super(Softmax4D, self).__init__(**kwargs)

    def build(self,input_shape):
        pass

    def call(self, x,mask=None):
        e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        s = K.sum(e, axis=self.axis, keepdims=True)
        return e / s

    def get_output_shape_for(self, input_shape):
        return input_shape

def bilinear_interpolation(w):
    frac = w[0].shape[0]
    n_predict = w[0].shape[-1]
    w_bilinear = np.zeros(w[0].shape)

    for i in range(n_predict):
        w_bilinear[:,:,i,i] = 1.0/(frac*frac) * np.ones((frac,frac))

    return w_bilinear

def resnet50_32s(input_shape = (224, 224, 3), model_input = ''):
    base_model = PoseNet_50(input_shape)
    
    #base_model.summary()
    if model_input != '':
        base_model.load_weights(model_input)
    
    #add predictor
    X = base_model.get_layer('leaky_re_lu_4').output
    X = Conv2D(14, (1, 1), name = 'pred_32', padding = 'valid', kernel_initializer = glorot_uniform(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    
    # add upsampler
    stride = 16
    X = UpSampling2D(size = (int(stride/2), int(stride/2)))(X)
    X = Conv2D(256, (5, 5), strides = (1, 1), name = 'pred_32s_p1', padding = 'valid', kernel_initializer = glorot_uniform(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    X = LeakyReLU(alpha=.001)(X)
    
    X = Conv2D(512, (5, 5), strides = (1, 1), name = 'pred_32s_p2', padding = 'valid', kernel_initializer = glorot_uniform(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    X = LeakyReLU(alpha=.001)(X)
    
    X = Conv2D(1024, (3, 3), strides = (1, 1), name = 'pred_32s', padding = 'valid', kernel_initializer = glorot_uniform(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    X = LeakyReLU(alpha=.001)(X)

    # output layer
    X = Flatten()(X)
    X = Dense(42, activation='linear', name='fc_'  + str('pred_32s'), kernel_initializer = glorot_uniform(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    
    model = Model(input=base_model.input,output=X)
    
    # create bilinear interpolation
    #w = model.get_layer('pred_32s').get_weights()
    #model.get_layer('pred_32s').set_weights([bilinear_interpolation(w), w[1]])
    
    # fine-tune 
    train_layers = ['pred_32',
                    'pred_32s_p1'
                    'pred_32s_p2'
                    'pred_32s',
                    'fc_pred_32s',

                    'bn4b_branch2c', 
                    'res4b_branch2c',
                    'bn4b_branch2b', 
                    'res4b_branch2b',
                    'bn4b_branch2a', 
                    'res4b_branch2a'

                    'bn4c_branch2c', 
                    'res4c_branch2c',
                    'bn4c_branch2b', 
                    'res4c_branch2b',
                    'bn4c_branch2a', 
                    'res4c_branch2a'

                    'bn5a_branch2c', 
                    'res5a_branch2c',
                    'bn5a_branch2b', 
                    'res5a_branch2b',
                    'bn5a_branch2a', 
                    'res5a_branch2a']

    for l in model.layers:
        if l.name in train_layers:
            l.trainable = True
        else :
            l.trainable = False

    return model, stride

def resnet50_16s(input_shape = (224, 224, 3), model_input = ''):
    # load 32s base model
    base_model, stride = resnet50_32s(input_shape)

    if model_input != '':
        base_model.load_weights(model_input)
    
    # add 16s classifier
    X = base_model.get_layer('activation_40').output
    X = Conv2D(42, (1, 1), name = 'pred_16', padding = 'valid', kernel_initializer = glorot_uniform(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    X = UpSampling2D(name='upsampling_16', size = (int(stride/2), int(stride/2)))(X)
    X = Conv2D(256, (5, 5), strides = (1, 1), name = 'pred_16s_p1', padding = 'valid', kernel_initializer = glorot_uniform(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    X = LeakyReLU(alpha=.001)(X)
    
    X = Conv2D(512, (5, 5), strides = (1, 1), name = 'pred_16s_p2', padding = 'valid', kernel_initializer = glorot_uniform(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    X = LeakyReLU(alpha=.001)(X)
    
    X = Conv2D(1024, (3, 3), strides = (1, 1), name = 'pred_16s', padding = 'valid', kernel_initializer = glorot_uniform(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    X = LeakyReLU(alpha=.001)(X)
    # merge classifiers
    X = merge([X, base_model.get_layer('pred_32s').output],mode = 'sum')
    
    #X = AveragePooling2D(pool_size=(2, 2), padding='valid', name='avg_pool')(X)
    
    # output layer
    X = Flatten()(X)

    X = Dense(42, activation='linear', name='fc_'  + str('pred_16s'), kernel_initializer = glorot_uniform(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    
    # create bilinear interpolation
    #w = model.get_layer('pred_16s').get_weights()
    #model.get_layer('pred_16s').set_weights([bilinear_interpolation(w), w[1]])

    # fine-tune 
    train_layers = ['pred_32',
                    'pred_32s_p1'
                    'pred_32s_p2'
                    'pred_32s',
                    'pred_16',
                    'pred_16s_p1'
                    'pred_16s_p2'
                    'pred_16s',
                    'fc_pred_16s',


                    'bn4b_branch2c', 
                    'res4b_branch2c',
                    'bn4b_branch2b', 
                    'res4b_branch2b',
                    'bn4b_branch2a', 
                    'res4b_branch2a'

                    'bn4c_branch2c', 
                    'res4c_branch2c',
                    'bn4c_branch2b', 
                    'res4c_branch2b',
                    'bn4c_branch2a', 
                    'res4c_branch2a'

                    'bn5a_branch2c', 
                    'res5a_branch2c',
                    'bn5a_branch2b', 
                    'res5a_branch2b',
                    'bn5a_branch2a', 
                    'res5a_branch2a']


    for l in model.layers:
        if l.name in train_layers:
            l.trainable = True
        else :
            l.trainable = False

    return model, strid
   
def resnet50_8s(input_shape = (224, 224, 3), model_input = ''):
    # load 16s base model
    base_model, stride = resnet50_16s(n_classes)

    if model_input != '':
        base_model.load_weights(model_input)
    
    # add 16s classifier
    X = base_model.get_layer('activation_22').output
    X = Conv2D(42, (1, 1), name = 'pred_8', padding = 'valid', kernel_initializer = glorot_uniform(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    X = UpSampling2D(name='upsampling_8',size=(int(stride/8), int(stride/8)))(X)
    X = Conv2D(256, (5, 5), strides = (1, 1), name = 'pred_8s_p1', padding = 'valid', kernel_initializer = glorot_uniform(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    X = LeakyReLU(alpha=.001)(X)
    
    X = Conv2D(512, (5, 5), strides = (1, 1), name = 'pred_8s_p2', padding = 'valid', kernel_initializer = glorot_uniform(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    X = LeakyReLU(alpha=.001)(X)
    
    X = Conv2D(1024, (3, 3), strides = (1, 1), name = 'pred_8s', padding = 'valid', kernel_initializer = glorot_uniform(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    X = LeakyReLU(alpha=.001)(X)

    # merge classifiers
    X = merge([X, base_model.get_layer('pred_16s').output],mode = 'sum')
    
    # output layer
    X = Flatten()(X)

    X = Dense(42, activation='linear', name='fc2_'  + str('pred_8s'), kernel_initializer = glorot_uniform(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)

    # create bilinear interpolation
    #w = model.get_layer('pred_8s').get_weights()
    #model.get_layer('pred_8s').set_weights([bilinear_interpolation(w), w[1]])

    # fine-tune 
    train_layers = ['pred_32',
                    'pred_32s_p1'
                    'pred_32s_p2'
                    'pred_32s',
                    'pred_16',
                    'pred_16s_p1'
                    'pred_16s_p2'
                    'pred_16s',
                    'pred_8',
                    'pred_8s_p1'
                    'pred_8s_p2'
                    'pred_8s',
                    'fc_pred_8s',

                    'bn4b_branch2c', 
                    'res4b_branch2c',
                    'bn4b_branch2b', 
                    'res4b_branch2b',
                    'bn4b_branch2a', 
                    'res4b_branch2a'

                    'bn4c_branch2c', 
                    'res4c_branch2c',
                    'bn4c_branch2b', 
                    'res4c_branch2b',
                    'bn4c_branch2a', 
                    'res4c_branch2a'

                    'bn5a_branch2c', 
                    'res5a_branch2c',
                    'bn5a_branch2b', 
                    'res5a_branch2b',
                    'bn5a_branch2a', 
                    'res5a_branch2a']


    for l in model.layers:
        if l.name in train_layers:
            l.trainable = True
        else :
            l.trainable = False

    return model, strid
   
