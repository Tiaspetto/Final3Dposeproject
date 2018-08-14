from data.data_scripts.readmat import *
from posenet_2d import *
from keras.models import Model, load_model
from keras import backend as K
from keras import optimizers
from keras.callbacks import LearningRateScheduler
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def euc_dist_keras(y_true, y_pred):
	return K.sqrt(K.sum(K.square(y_true - y_pred), axis = -1, keepdims = True))

if __name__ == '__main__':

    model = PoseNet_50(input_shape=(224, 224, 3))
    adadelta = optimizers.Adadelta(lr = 0.05, rho = 0.9, decay = 0.0)
    model.compile(optimizer = adadelta, loss = euc_dist_keras,
                  metrics=['mae'])
    model.load_weights('D:/dissertation/model_data/weights-0.2056.hdf5')
    
    prediction = model.predict(np.array([read_image(1)]))

    result = prediction[0]
    result = result.transpose((2, 0, 1))
    plt.figure()
    for i in range(14):
        plt.subplot(4,4,i+1)
        imshow(result[i])
    plt.show()