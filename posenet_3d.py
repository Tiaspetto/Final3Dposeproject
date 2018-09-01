# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import keras.backend as K
from keras.engine import Layer

from keras.layers import Input, Dropout, merge, TimeDistributed, ZeroPadding2D, Conv3D
from keras.layers.convolutional import Convolution2D, UpSampling2D, ZeroPadding2D, Cropping2D, Deconvolution2D
from keras.layers.core import Activation

from keras.applications.resnet50 import ResNet50
from keras.models import Model, load_model
from posenet_2d import *
from keras.initializers import glorot_normal


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
    X = Conv2D(256, (1, 1), name = 'pred_32', padding = 'valid', kernel_initializer = glorot_normal(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    
    # add upsampler
    stride = 4
    X = UpSampling2D(size = (int(stride), int(stride)))(X)

    X = Conv2D(256, (3, 3), strides = (1, 1), name = 'pred_32s_feature1', padding = 'same', kernel_initializer = glorot_normal(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    X = Activation('tanh')(X)

    X = Conv2D(128, (3, 3), strides = (1, 1), name = 'pred_32s_feature2', padding = 'same', kernel_initializer = glorot_normal(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    X = Activation('tanh')(X)

    X = Conv2D(128, (5, 5), strides = (2, 2), name = 'pred_32s_p1', padding = 'valid', kernel_initializer = glorot_normal(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    X = Activation('tanh')(X)
    
    X = Conv2D(128, (5, 5), strides = (2, 2), name = 'pred_32s', padding = 'valid', kernel_initializer = glorot_normal(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    X = Activation('tanh')(X)

    # output layer
    X = Flatten()(X)
    X = Dense(1024, activation='linear', name='fc_'  + str('pred_32s_1024'), kernel_initializer = glorot_normal(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    X = Activation('tanh')(X)
    X = Dense(42, activation='linear', name='fc_'  + str('pred_32s'), kernel_initializer = glorot_normal(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    
    model = Model(input=base_model.input,output=X)
    
    # # fine-tune 
    # train_layers = ['pred_32',
    #                 'pred_32s_feature1',
    #                 'pred_32s_feature2',
    #                 'pred_32s_p1',
    #                 'pred_32s',
    #                 'fc_pred_32s_1024',
    #                 'fc_pred_32s',

    #                 'bn4b_branch2c', 
    #                 'res4b_branch2c',
    #                 'bn4b_branch2b', 
    #                 'res4b_branch2b',
    #                 'bn4b_branch2a', 
    #                 'res4b_branch2a'

    #                 'bn4c_branch2c', 
    #                 'res4c_branch2c',
    #                 'bn4c_branch2b', 
    #                 'res4c_branch2b',
    #                 'bn4c_branch2a', 
    #                 'res4c_branch2a'

    #                 'bn5a_branch2c', 
    #                 'res5a_branch2c',
    #                 'bn5a_branch2b', 
    #                 'res5a_branch2b',
    #                 'bn5a_branch2a', 
    #                 'res5a_branch2a',
    #                 'res5b_branch2a',
    #                 'bn5b_branch2a',
    #                 ]

    # for l in model.layers:
    #     if l.name in train_layers:
    #         l.trainable = True
    #     else :
    #         l.trainable = False

    return model, stride

def resnet50_16s(input_shape = (224, 224, 3), model_input = ''):
    # load 32s base model
    base_model, stride = resnet50_32s(input_shape)

    if model_input != '':
        base_model.load_weights(model_input)
    
    # add 16s classifier
    X = base_model.get_layer('activation_40').output
    X = Conv2D(256, (1, 1), name = 'pred_16', padding = 'valid', kernel_initializer = glorot_normal(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    X = UpSampling2D(name='upsampling_16', size = (int(stride), int(stride)))(X)
    X = Conv2D(256, (3, 3), strides = (1, 1), name = 'pred_16s_feature1', padding = 'same', kernel_initializer = glorot_normal(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    
    X = Add()([X, base_model.get_layer('pred_32s_feature1').output])
    X = Activation('tanh')(X)

    X = Conv2D(128, (3, 3), strides = (1, 1), name = 'pred_16s_feature2', padding = 'same', kernel_initializer = glorot_normal(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    X = Activation('tanh')(X)

    X = Conv2D(128, (5, 5), strides = (2, 2), name = 'pred_16s_p1', padding = 'valid', kernel_initializer = glorot_normal(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    X = Activation('tanh')(X)

    X = Conv2D(128, (5, 5), strides = (2, 2), name = 'pred_16s', padding = 'valid', kernel_initializer = glorot_normal(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    X = Activation('tanh')(X)
    #X = AveragePooling2D(pool_size=(2, 2), padding='valid', name='avg_pool')(X)
    
    # output layer
    X = Flatten()(X)
    
    X = Dense(1024, activation='linear', name='fc_'  + str('pred_16s_1024'), kernel_initializer = glorot_normal(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    X = Activation('tanh')(X)
    X = Dense(42, activation='linear', name='fc_'  + str('pred_16s'), kernel_initializer = glorot_normal(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    
    model = Model(input=base_model.input,output=X)
    # create bilinear interpolation
    #w = model.get_layer('pred_16s').get_weights()
    #model.get_layer('pred_16s').set_weights([bilinear_interpolation(w), w[1]])

    #fine-tune 
    train_layers = ['pred_32',
                    'pred_32s_feature1',
                    'pred_16',
                    'pred_16s_feature1',
                    'pred_16s_feature2',
                    'pred_16s_p1',
                    'pred_16s',
                    'fc_pred_16s_1024',
                    'fc_pred_16s',


                    'bn4b_branch2c', 
                    'res4b_branch2c',
                    'bn4b_branch2b', 
                    'res4b_branch2b',
                    'bn4b_branch2a', 
                    'res4b_branch2a',

                    'bn4c_branch2c', 
                    'res4c_branch2c',
                    'bn4c_branch2b', 
                    'res4c_branch2b',
                    'bn4c_branch2a', 
                    'res4c_branch2a',

                    'bn5a_branch2c', 
                    'res5a_branch2c',
                    'bn5a_branch2b', 
                    'res5a_branch2b',
                    'bn5a_branch2a', 
                    'res5a_branch2a',
                    'res5b_branch2a',
                    'bn5b_branch2a'
                    ]

    for l in model.layers:
        if l.name in train_layers:
            l.trainable = True
        else :
            l.trainable = False

    return model, stride
   
def resnet50_8s(input_shape = (224, 224, 3), model_input = 'None'):
    # load 16s base model
    base_model, stride = resnet50_16s(input_shape)

    if model_input != 'None':
        base_model.load_weights(model_input)
    
    # add 16s classifier
    X = base_model.get_layer('activation_22').output
    X = Conv2D(256, (1, 1), name = 'pred_8', padding = 'valid', kernel_initializer = glorot_normal(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    X = UpSampling2D(name='upsampling_8',size=(int(stride/2), int(stride/2)))(X)
    #X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(256, (3, 3), strides = (1, 1), name = 'pred_8s_feature1', padding = 'same', kernel_initializer = glorot_normal(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    
    # merge classifiers
    X = Add()([X, base_model.get_layer('pred_16s_feature1').output])
    X = Activation('tanh')(X)

    X = Conv2D(128, (3, 3), strides = (1, 1), name = 'pred_8s_feature2', padding = 'same', kernel_initializer = glorot_normal(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('tanh')(X)

    X = Conv2D(128, (5, 5), strides = (2, 2), name = 'pred_8s_p1', padding = 'valid', kernel_initializer = glorot_normal(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    X = Activation('tanh')(X)

    X = Conv2D(128, (5, 5), strides = (2, 2), name = 'pred_8s', padding = 'valid', kernel_initializer = glorot_normal(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    X = Activation('tanh')(X)

    
    # output layer
    X = Flatten()(X)
    X = Dense(1024, activation='linear', name='fc_'  + str('pred_8s_1024'), kernel_initializer = glorot_normal(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    X = Activation('tanh')(X)
    X = Dense(42, activation='linear', name='fc_'  + str('pred_8s'), kernel_initializer = glorot_normal(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    
    model = Model(input=base_model.input,output=X)
    # create bilinear interpolation
    #w = model.get_layer('pred_8s').get_weights()
    #model.get_layer('pred_8s').set_weights([bilinear_interpolation(w), w[1]])

    #fine-tune 
    train_layers = ['pred_32',
                    'pred_32s_feature1',
                    'pred_16',
                    'pred_16s_feature1',
                    'pred_8',
                    'pred_8s_feature1',
                    'pred_8s_feature2',
                    'pred_8s_p1',
                    'pred_8s',
                    'fc_pred_8s_1024',
                    'fc_pred_8s',

                    'bn4b_branch2c', 
                    'res4b_branch2c',
                    'bn4b_branch2b', 
                    'res4b_branch2b',
                    'bn4b_branch2a', 
                    'res4b_branch2a',

                    'bn4c_branch2c', 
                    'res4c_branch2c',
                    'bn4c_branch2b', 
                    'res4c_branch2b',
                    'bn4c_branch2a', 
                    'res4c_branch2a',

                    'bn5a_branch2c', 
                    'res5a_branch2c',
                    'bn5a_branch2b', 
                    'res5a_branch2b',
                    'bn5a_branch2a', 
                    'res5a_branch2a',
                    'res5b_branch2a',
                    'bn5b_branch2a',
                    ]


    for l in model.layers:
        if l.name in train_layers:
            l.trainable = True
        else :
            l.trainable = False

    return model, stride


def make_seq_model(model_input):
    base_model, _ = resnet50_16s(input_shape=(224, 224, 3))
    base_model.load_weights(model_input)
    x = base_model.get_layer('pred_16s').output

    model = Model(input=base_model.input,output=x)
    #model.summary()
    main_input = Input(shape=(8, 224, 224, 3), dtype='float32', name='cnn_input')

    X = TimeDistributed(model)(main_input)
    X = Conv3D(64, (3, 3, 3), name = "3D_conv_1", padding = 'same',  kernel_initializer = glorot_normal(seed=0))(X)
    X = Activation('tanh')(X)
    X = Conv3D(64, (3, 3, 3), name = "3D_conv_2", padding = 'same',  kernel_initializer = glorot_normal(seed=0))(X)
    X = Activation('tanh')(X)
    X = Conv3D(128, (3, 3, 3), name = "3D_conv_3", padding = 'same',  kernel_initializer = glorot_normal(seed=0))(X)
    X = Activation('tanh')(X)
    X = Conv3D(128, (3, 3, 3), name = "3D_conv_4", padding = 'same',  kernel_initializer = glorot_normal(seed=0))(X)
    X = Activation('tanh')(X)
    X = Flatten()(X)
    X = Dense(1024, activation='linear', name='fc_'  + str('3D_pred_1024'), kernel_initializer = glorot_normal(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    X = Activation('tanh')(X)
    X = Dense(42, activation='linear', name='fc_'  + str('3D_pred'), kernel_initializer = glorot_normal(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    
    time_model = Model(input=main_input, output=X)
    
    train_layers = ["3D_conv_1",
                    "3D_conv_2",
                    "3D_conv_3",
                    "3D_conv_4",
                    "fc_3D_pred_1024",
                    "fc_3D_pred"
                    ]

    for l in model.layers:
        if l.name in train_layers:
            l.trainable = True
        else :
            l.trainable = False

    return time_model
   
