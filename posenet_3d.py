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
	n_classes = w[0].shape[-1]
	w_bilinear = np.zeros(w[0].shape)

	for i in range(n_classes):
		w_bilinear[:,:,i,i] = 1.0/(frac*frac) * np.ones((frac,frac))

	return w_bilinear

def resnet50_fcn(input_shape = (224, 224, 15)):
    #Input tensor 
    X_input = Input(input_shape)
    base_model = ResNet50(weights = "imagenet", include_top = False, input_tensor = X_input)
    
    #add predictor
    X = base_model.get_layer('activation_49').output
    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = Convolution2D(joints * 3,1,1,name = 'pred_32',init='zero',border_mode = 'valid')(x)
    
    # add upsampler
	stride = 32
	X = UpSampling2D(size=(stride,stride))(x)
	X = Convolution2D(joints * 3,5,5,name = 'pred_32s',init='zero',border_mode = 'same')(x)
    
    X = AveragePooling2D(pool_size=(2, 2), padding='valid', name='avg_pool')(X)
    
    # output layer
    X = Flatten()(X)
    X = Dense(joints * 3 , activation='linear', name='fc' + str(pred_32s), kernel_initializer = glorot_uniform(seed=0))(X)
    
    model = Model(input=base_model.input,output=X)
    
    # create bilinear interpolation
	w = model.get_layer('pred_32s').get_weights()
	model.get_layer('pred_32s').set_weights([bilinear_interpolation(w), w[1]])
    
    return model, stride

def resnet50_16s_fcn(input_shape = (224, 224, 15), model_input = ''):
    # load 32s base model
	base_model, stride = resnet50_fcn(n_classes)

	if model_input != '':
		base_model.load_weights(model_input)
    
    # add 16s classifier
	X = base_model.get_layer('activation_40').output
	X = Convolution2D(joints * 3,1,1,name = 'pred_16',init='zero',border_mode = 'valid')(x)
	X = UpSampling2D(name='upsampling_16',size=(stride/2,stride/2))(x)
	X = Convolution2D(n_classes,5,5,name = 'pred_up_16',init='zero',border_mode = 'same')(x)
    
    # merge classifiers
	X = merge([X, base_model.get_layer('pred_32s').output],mode = 'sum')
    
    X = AveragePooling2D(pool_size=(2, 2), padding='valid', name='avg_pool')(X)
    
    # output layer
    X = Flatten()(X)
    X = Dense(joints * 3 , activation='linear', name='fc' + str(pred_32s), kernel_initializer = glorot_uniform(seed=0))(X)
    
   
    


