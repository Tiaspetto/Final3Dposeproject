from data.data_scripts.readmat import *
from posenet_2d import *
from keras.models import Model, load_model
from keras import backend as K
from keras import optimizers
from keras.callbacks import LearningRateScheduler
import random
import tensorflow as tf


def get_img_batch(batch_array):
    imgs = []
    for index in batch_array:
        img = read_image(index)
        imgs.append(img)

    return np.array(imgs)


def get_heat_batch(batch_array):
    heats = []
    for index in batch_array:
        heat = read_heat_info(index)
        heats.append(heat)

    return np.array(heats)


def get_train_batch(index_array, batch_size):

    while 1:
        for i in range(0, len(index_array), batch_size):
            batch_array = index_array[i:i+batch_size]
            x = get_img_batch(batch_array)
            y = get_heat_batch(batch_array)
            #print(np.shape(x), np.shape(y))
            yield (x,y)

def shuffle(index_array):
    for i in range(0, len(index_array)-1):
        index = random.randint(i, len(index_array)-1)
        index_array[index], index_array[i] = index_array[i], index_array[index]

    return index_array

def euc_dist_keras(y_true, y_pred):
	return K.sqrt(K.sum(K.square(y_true - y_pred), axis = -1, keepdims = True))

def step_decay(epochs):
	initial_lrate = 0.05
	drop = 0.5
	epochs_drop = 15
	lrate = initial_lrate * math.pow(drop, math.floor((1+epochs)/epochs_drop))
	print("learning rate drop to:", lrate)
	return lrate

if __name__ == '__main__':

    index_array = range(1, 1901)

    index_array = list(index_array)

    index_array = shuffle(index_array)

    validation_array = list(range(1901,2001))


    ckpt_path = 'log/weights-{val_loss:.4f}.hdf5'
    ckpt = tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                              monitor='val_loss',
                                              verbose=1,
                                              save_best_only=True,
                                              mode='min')

    model = PoseNet_50(input_shape=(224, 224, 3))
    adadelta = optimizers.Adadelta(lr = 0.05, rho = 0.9, decay = 0.0)
    model.compile(optimizer = adadelta, loss = euc_dist_keras,
                  metrics=['mae'])
    lrate = LearningRateScheduler(step_decay)
    result = model.fit_generator(generator=get_train_batch(index_array, 8),
                                 steps_per_epoch=238,
                                 callbacks=[ckpt, lrate],
                                 epochs=60000, verbose=1,
                                 validation_data=get_train_batch(validation_array, 8),
                                 validation_steps=52,
                                 workers=1)