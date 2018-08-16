# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import keras.backend as K
from keras.engine import Layer

from keras.layers import Input, Dropout, merge
from keras.layers.convolutional import Convolution2D, UpSampling2D, ZeroPadding2D, Cropping2D, Deconvolution2D
from keras.layers.core import Activation

from keras.applications.resnet50 import ResNet50
from keras.models import Model, load_model

import numpy as np

def bilinear_interpolation(w):
    frac = w[0].shape[0]
    n_predict = w[0].shape[-1]
    w_bilinear = np.zeros(w[0].shape)

    for i in range(n_predict):
        w_bilinear[:,:,i,i] = 1.0/(frac*frac) * np.ones((frac,frac))

    return w_bilinear

def resnet50_32s(input_shape = (224, 224, 15)):
    predict_shape = input_shape[3] - 1 
    predict_shape = predict_shape * 3
    #Input tensor 
    X_input = Input(input_shape)
    base_model = ResNet50(weights = "imagenet", include_top = False, input_tensor = X_input)
    
    #add predictor
    X = base_model.get_layer('activation_49').output
    X = Convolution2D(predict_shape, 1, 1, name = 'pred_32', init = 'zero', border_mode = 'valid')(x)
    
    # add upsampler
    stride = 32
    X = UpSampling2D(size = (stride,stride))(x)
    X = Convolution2D(predict_shape, 5, 5, name = 'pred_32s', init = 'zero', border_mode = 'same')(x)
    
    X = AveragePooling2D(pool_size = (2, 2), padding = 'valid', name = 'avg_pool')(X)
    
    # output layer
    X = Flatten()(X)
    X = Dense(predict_shape , activation='linear', name='fc' + str('pred_32s'), kernel_initializer = glorot_uniform(seed=0))(X)
    
    model = Model(input=base_model.input,output=X)
    
    # create bilinear interpolation
    w = model.get_layer('pred_32s').get_weights()
    model.get_layer('pred_32s').set_weights([bilinear_interpolation(w), w[1]])
    
    # fine-tune 
    train_layers = ['pred_32',
                    'pred_32s',

                    'bn5c_branch2c', 
                    'res5c_branch2c',
                    'bn5c_branch2b', 
                    'res5c_branch2b',
                    'bn5c_branch2a', 
                    'res5c_branch2a',

                    'bn5b_branch2c', 
                    'res5b_branch2c',
                    'bn5b_branch2b', 
                    'res5b_branch2b',
                    'bn5b_branch2a', 
                    'res5b_branch2a',

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

def resnet50_16s(input_shape = (224, 224, 15), model_input = ''):

    predict_shape = input_shape[3] - 1 
    predict_shape = predict_shape * 3

    # load 32s base model
    base_model, stride = resnet50_32s(input_shape)

    if model_input != '':
        base_model.load_weights(model_input)
    
    # add 16s classifier
    X = base_model.get_layer('activation_40').output
    X = Convolution2D(predict_shape, 1, 1, name = 'pred_16', init = 'zero', border_mode = 'valid')(x)
    X = UpSampling2D(name='upsampling_16', size = (stride/2, stride/2))(x)
    X = Convolution2D(predict_shape, 5, 5, name = 'pred_16s', init = 'zero', border_mode = 'same')(x)
    
    # merge classifiers
    X = merge([X, base_model.get_layer('pred_32s').output],mode = 'sum')
    
    X = AveragePooling2D(pool_size=(2, 2), padding='valid', name='avg_pool')(X)
    
    # output layer
    X = Flatten()(X)
    X = Dense(predict_shape, activation='linear', name='fc' + str('pred_16s'), kernel_initializer = glorot_uniform(seed=0))(X)
    
    # create bilinear interpolation
    w = model.get_layer('pred_16s').get_weights()
    model.get_layer('pred_16s').set_weights([bilinear_interpolation(w), w[1]])

    # fine-tune 
    train_layers = ['pred_32',
                    'pred_32s',
                    'pred_16',
                    'pred_16s',

                    'bn5c_branch2c', 
                    'res5c_branch2c',
                    'bn5c_branch2b', 
                    'res5c_branch2b',
                    'bn5c_branch2a', 
                    'res5c_branch2a',

                    'bn5b_branch2c', 
                    'res5b_branch2c',
                    'bn5b_branch2b', 
                    'res5b_branch2b',
                    'bn5b_branch2a', 
                    'res5b_branch2a',

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
   
def resnet50_8s(input_shape = (224, 224, 15), model_input = ''):
    predict_shape = input_shape[3] - 1 
    predict_shape = predict_shape * 3
    # load 16s base model
    base_model, stride = resnet50_16s(n_classes)

    if model_input != '':
        base_model.load_weights(model_input)
    
    # add 16s classifier
    x = base_model.get_layer('activation_22').output
    x = Convolution2D(predict_shape,1,1,name = 'pred_8',init='zero',border_mode = 'valid')(x)
    x = UpSampling2D(name='upsampling_8',size=(stride/4,stride/4))(x)
    x = Convolution2D(predict_shape,5,5,name = 'pred_8s',init='zero',border_mode = 'same')(x)
    
    # merge classifiers
    x = merge([x, base_model.get_layer('pred_16s').output],mode = 'sum')
    
    # output layer
    X = Flatten()(X)
    X = Dense(predict_shape , activation='linear', name='fc' + str('pred_8s'), kernel_initializer = glorot_uniform(seed=0))(X)
    model = Model(input=base_model.input,output=x)

    # create bilinear interpolation
    w = model.get_layer('pred_8s').get_weights()
    model.get_layer('pred_8s').set_weights([bilinear_interpolation(w), w[1]])

    # fine-tune 
    train_layers = ['pred_32',
                    'pred_32s',
                    'pred_16',
                    'pred_16s',
                    'pred_8',
                    'pred_8s',

                    'bn5c_branch2c', 
                    'res5c_branch2c',
                    'bn5c_branch2b', 
                    'res5c_branch2b',
                    'bn5c_branch2a', 
                    'res5c_branch2a',

                    'bn5b_branch2c', 
                    'res5b_branch2c',
                    'bn5b_branch2b', 
                    'res5b_branch2b',
                    'bn5b_branch2a', 
                    'res5b_branch2a',

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
   
