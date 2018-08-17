from data.data_scripts.data_utils import *
from posenet_2d import *
from posenet_3d import *
from keras.models import Model, load_model
from keras import backend as K
from keras import optimizers
from keras.callbacks import LearningRateScheduler
import random
import tensorflow as tf
import cv2


def pose2d_get_img_batch(batch_array):
    imgs = []
    for index in batch_array:
        img = read_image(index)
        imgs.append(img)

    return np.array(imgs)


def pose2d_get_heat_batch(batch_array):
    heats = []
    for index in batch_array:
        heat = read_heat_info(index)
        heats.append(heat)

    return np.array(heats)


def pose2d_get_train_batch(index_array, batch_size):
    while 1:
        for i in range(0, len(index_array), batch_size):
            batch_array = index_array[i:i+batch_size]
            x = pose2d_get_img_batch(batch_array)
            y = pose2d_get_heat_batch(batch_array)
            #print(np.shape(x), np.shape(y))
            yield (x, y)


def pose3d_get_img_batch(batch_array, isTrain):
    imgs = []
    for index in batch_array:
        img = read_image(index, "ECCV", isTrain)
        imgs.append(img)
    return np.array(imgs)


def pose3d_get_pose_batch(batch_array, isTrain):
    poses = []
    for index in batch_array:
        pose = read_pose_data(index, isTrain)
        poses.append(pose)
    return np.array(poses)


def pose3d_get_train_batch(array, batch_size, isTrain):
    while 1:
        for i in range(0, len(array), batch_size):
            batch_array = array[i:i+batch_size]
            x = pose3d_get_img_batch(batch_array, isTrain)
            y = pose3d_get_pose_batch(batch_array, isTrain)
            #print(np.shape(x), np.shape(y))
            yield (x, y)


def shuffle(index_array):
    for i in range(0, len(index_array)-1):
        index = random.randint(i, len(index_array)-1)
        index_array[index], index_array[i] = index_array[i], index_array[index]

    return index_array


def euc_dist_keras(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True))


def step_decay(epochs):
    initial_lrate = 0.05
    drop = 0.5
    epochs_drop = 8
    lrate = initial_lrate * math.pow(drop, math.floor((1+epochs)/epochs_drop))

    lrate = max(float('3.3e-5'), lrate)
    print("learning rate drop to:", lrate)

    return lrate


if __name__ == '__main__':

    # index_array = range(1, 1901)

    # index_array = list(index_array)

    # index_array = shuffle(index_array)

    # validation_array = list(range(1901,2001))

    # ckpt_path = 'log/weights-{val_loss:.4f}.hdf5'
    # ckpt = tf.keras.callbacks.ModelCheckpoint(ckpt_path,
    #                                           monitor='val_loss',
    #                                           verbose=1,
    #                                           save_best_only=True,
    #                                           mode='min')

    # model = PoseNet_50(input_shape=(224, 224, 3))
    # adadelta = optimizers.Adadelta(lr = 0.05, rho = 0.9, decay = 0.0)
    # model.compile(optimizer = adadelta, loss = euc_dist_keras,
    #               metrics=['mae'])
    # lrate = LearningRateScheduler(step_decay)
    # result = model.fit_generator(generator=pose2d_get_train_batch(index_array, 8),
    #                              steps_per_epoch=238,
    #                              callbacks=[ckpt, lrate],
    #                              epochs=60000, verbose=1,
    #                              validation_data=pose2d_get_train_batch(validation_array, 8),
    #                              validation_steps=52,
    #                              workers=1)

    #====train 3d pose =======#
    train_array = list(range(1, 35833))
    train_array = shuffle(train_array)
    val_array = list(range(1, 19313))
    val_array = shuffle(val_array)

    ckpt_path = 'log/3d_weights-{val_loss:.4f}.hdf5'
    ckpt = tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                              monitor='val_loss',
                                              verbose=1,
                                              save_best_only=True,
                                              mode='min')

    model, stride = resnet50_32s(input_shape=(
        224, 224, 3), model_input="D:/dissertation/model_data/weights-0.2056.hdf5")
    adadelta = optimizers.Adadelta(lr=0.05, rho=0.9, decay=0.0)
    model.compile(optimizer=adadelta, loss=euc_dist_keras,
                  metrics=['mae'])
    lrate = LearningRateScheduler(step_decay)
    result = model.fit_generator(generator=pose3d_get_train_batch(train_array, 8, True),
                                 steps_per_epoch=4480,
                                 callbacks=[ckpt, lrate],
                                 epochs=60000, verbose=1,
                                 validation_data=pose3d_get_train_batch(val_array, 8, False),
                                 validation_steps=2415,
                                 workers=1)
