import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input
from keras import regularizers
from matplotlib.pyplot import imshow
from resnets_utils import *
from keras.layers.advanced_activations import LeakyReLU

import scipy.misc
import keras.backend as K

# GRADED FUNCTION: identity_block

def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = LeakyReLU(alpha=.001)(X)
    
    
    # Second component of main path 
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), padding = 'same', name =  conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = LeakyReLU(alpha=.001)(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', name =  conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0), kernel_regularizer = regularizers.l2(0.01))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation 
    X = Add()([X_shortcut, X])
    X = LeakyReLU(alpha=.001)(X)
    
    
    return X

def non_short_cut_identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1 = filters[0]    
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0), kernel_regularizer = regularizers.l2(0.05))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = LeakyReLU(alpha=.001)(X)
    
    
    
    return X

# GRADED FUNCTION: convolutional_block

def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0), kernel_regularizer = regularizers.l2(0.05))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = LeakyReLU(alpha=.001)(X)
    

    # Second component of main path
    X = Conv2D(F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0), kernel_regularizer = regularizers.l2(0.05))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = LeakyReLU(alpha=.001)(X)

    # Third component of main path
    X = Conv2D(F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0), kernel_regularizer = regularizers.l2(0.05))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### 
    X_shortcut =  Conv2D(F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0), kernel_regularizer = regularizers.l2(0.05))(X_shortcut)
    X_shortcut =  BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation 
    X = Add()([X_shortcut, X])
    X = LeakyReLU(alpha=.001)(X)
    
    
    return X

# GRADED FUNCTION: ResNet50

def __PoesNet50(input_shape = (224, 224, 3)):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f=3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f=3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f=3, filters = [512, 512, 1024], stage = 5, block='a', s = 1)
    X = non_short_cut_identity_block(X, 1, [64], stage=5, block='b')
    
    X = UpSampling2D(size = (2,2))(X)
    X = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding = 'same', use_bias = False, kernel_regularizer = regularizers.l2(0.05))(X)
    X = UpSampling2D(size = (2,2))(X)
    X = Conv2DTranspose(14, (4, 4), strides=(2, 2), padding = 'same', use_bias = False, kernel_regularizer = regularizers.l2(0.05))(X)
    
    model = Model(inputs = X_input, outputs = X, name = "PoseNet_50")

    return model

def PoseNet_50(input_shape = (224, 224, 3)):
    # Define input tensor with input_shape
    X_input = Input(input_shape)
    base_model = ResNet50(weights = 'imagenet', include_top = False, input_tensor = X_input)

    X = base_model.get_layer('activation_40').output
    X = convolutional_block(X, f=3, filters = [512, 512, 1024], stage = 5, block='a', s = 1)
    X = non_short_cut_identity_block(X, 1, [64], stage = 5, block = 'b')

    X = UpSampling2D(size = (2,2))(X)
    X = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding = 'same', use_bias = False, kernel_regularizer = regularizers.l2(0.05))(X)
    X = UpSampling2D(size = (2,2))(X)
    X = Conv2DTranspose(14, (4, 4), strides=(2, 2), padding = 'same', use_bias = False, kernel_regularizer = regularizers.l2(0.05))(X)
    
    model = Model(inputs = X_input, outputs = X, name = "PoseNet_50")

    return model

if __name__ == "__main__":
    printpath()


