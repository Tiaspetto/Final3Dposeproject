from data.data_scripts.data_utils import *
from posenet_2d import *
from posenet_3d import *
from keras.models import Model, load_model
from keras import backend as K
from keras import optimizers
from keras.callbacks import LearningRateScheduler
from cyclical_learning_rate import CyclicLR
import random
import tensorflow as tf
import cv2
import sys

joint_parents = [-1, 0, 1, -1, 3, 4, -1, 6, -1, 8, 9, -1, 11, 12]

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

def pose2d_get_further_train_batch(index_array, batch_size, isTrain):
    data = get_MPII_data(isTrain)
    while 1:
        for i in range(0, len(index_array), batch_size):
            batch_array = index_array[i:i+batch_size]
            imgs =  []
            heats = []   
            for index in batch_array:
                img, height, width = MPI_read_img(data[index][0])
                imgs.append(img)
                heats.append(MPI_read_heat_info(isTrain, data[index][0]))

            yield (np.array(imgs), np.array(heats))




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
    return K.sum(K.sqrt(K.sum(K.square(y_true - y_pred), axis=2)), axis =1)


def calc_parant(y_true, y_pred):
    py_true = K.concatenate([y_true[:,i,:] if joint_parents[i] == -1 else (y_true[:,i,:] - y_true[:,joint_parents[i],:] ) for i in range(0,14)])
    py_pred = K.concatenate([y_pred[:,i,:] if joint_parents[i] == -1 else (y_pred[:,i,:] - y_pred[:,joint_parents[i],:] ) for i in range(0,14)])

    return (K.reshape(py_true, [-1, 14, 3]), K.reshape(py_pred, [-1, 14, 3]))

def calc_parent_loss1_loss2(py_true, py_pred):
    inner_true = K.sqrt(K.sum(K.square(py_true), axis=2))
    inner_pred = K.sqrt(K.sum(K.square(py_pred), axis=2))
  
    loss1 = K.mean(K.square(inner_true - inner_pred)  ,axis = 1)
    inner_mul = inner_true * inner_pred + float("1e-8")
    inner_vec = K.sum((py_true * py_pred), axis = 2) 
    inner_vec = 1- inner_vec/inner_mul
    loss2 = K.mean(inner_vec, axis = 1)

    return (loss1, loss2)

def euc_joint_dist_loss(y_true, y_pred):
    y_pred = K.reshape(y_pred, [-1, 14, 3])
    y_true = K.reshape(y_true, [-1, 14, 3])
    
    loss = K.mean(K.sqrt(K.sum(K.square(y_true - y_pred), axis=2)), axis = 1)
    py_true, py_pred = calc_parant(y_true, y_pred)
    # shape = K.int_shape(py_true)
    # print(shape)
    loss1, loss2 = calc_parent_loss1_loss2(py_true, py_pred)
    return loss+4*loss1+loss2

def cal_pckh(py_true, euc_dist):
    inner_true = K.sqrt(K.sum(K.square(py_true), axis=2))
    half_head = inner_true[:, 7]
    half_head = K.tile(half_head, [14])
    half_head = K.reshape(half_head, [-1, 14])
    pckh_vec = half_head - euc_dist
    greater = K.greater_equal(pckh_vec, 0)
    greater = K.cast(greater, "float32")
    greater = K.mean(greater, axis = 1)
    return greater


def metrics_pckh(y_true, y_pred):
    y_pred = K.reshape(y_pred, [-1, 14, 3])
    y_true = K.reshape(y_true, [-1, 14, 3])

    py_true, py_pred = calc_parant(y_true, y_pred)
    euc_dist = K.sqrt(K.sum(K.square(y_true - y_pred), axis=2))
    pckh = cal_pckh(py_true, euc_dist)
    return pckh

def euc_joint_metrics_dist_keras(y_true, y_pred):
    y_pred = K.reshape(y_pred, [-1, 14, 3])
    y_true = K.reshape(y_true, [-1, 14, 3])
    y_pred = y_pred * 1000.0
    y_true = y_true * 1000.0

    loss = K.mean(K.sqrt(K.sum(K.square(y_true - y_pred), axis=2)), axis = 1)
    return loss

def step_decay(epochs):
    initial_lrate = float('0.05')
    drop = 0.5
    epochs_drop = 8
    lrate = initial_lrate * math.pow(drop, math.floor((1+epochs)/epochs_drop))

    print("learning rate drop to:", lrate)
    return lrate

def read_skip():
    train_skip = []
    val_skip = []
    f = open("no_bbox.txt")
    train_skip = f.read().split(',')
    train_skip = [int(i) for i in train_skip if len(i)==5]
    f.close()
    f = open("no_val_bbox.txt")
    val_skip = f.read().split(',')
    val_skip = [int(i) for i in val_skip if len(i)==5]
    f.close()
    return (train_skip, val_skip)

def train_2d():
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
    clr = CyclicLR(base_lr = float("3.3e-4"), max_lr = 0.05, step_size = 1900, mode = 'triangular')
    model.compile(optimizer = adadelta, loss = euc_dist_keras,
                  metrics=['mae'])
    model.load_weights('D:/dissertation/log/weights-118.9920.hdf5')
    #lrate = LearningRateScheduler(step_decay)
    result = model.fit_generator(generator=pose2d_get_train_batch(index_array, 8),
                                 steps_per_epoch=238,
                                 callbacks=[ckpt, clr],
                                 epochs=60000, verbose=1,
                                 validation_data=pose2d_get_train_batch(validation_array, 8),
                                 validation_steps=52,
                                 workers=1)

def feature_train_2d():
    train_array = range(0, 17928)

    train_array = list(train_array)

    train_array = shuffle(train_array)

    validation_array = list(range(0,1991))


    ckpt_path = 'log/weights-{val_loss:.4f}.hdf5'
    ckpt = tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                              monitor='val_loss',
                                              verbose=1,
                                              save_best_only=True,
                                              mode='min')

    model = PoseNet_50(input_shape=(224, 224, 3))
    adadelta = optimizers.Adadelta(lr = 0.05, rho = 0.9, decay = 0.0)
    # model.load_weights("model_data/weights-0.0753.hdf5")
    model.compile(optimizer = adadelta, loss = euc_dist_keras,
                  metrics=['mae'])
    #lrate = LearningRateScheduler(step_decay)
    clr = CyclicLR(base_lr = float("3.3e-4"), max_lr = 0.001, step_size = 1900, mode = 'triangular')
    result = model.fit_generator(generator=pose2d_get_further_train_batch(train_array, 8, True),
                                 steps_per_epoch=2241,
                                 callbacks=[ckpt, clr],
                                 epochs=60000, verbose=1,
                                 validation_data=pose2d_get_further_train_batch(validation_array, 8, False),
                                 validation_steps=249,
                                 workers=1)

def train_3d(base_model = '', ckpt_model = ''):
    train_skip, val_skip = read_skip()
    train_array = list(range(1, 16545)) 
    train_array = [i for i in train_array if i not in train_skip]
    train_array = shuffle(train_array)

    val_array = list(range(1, 4740))
    val_array = [i for i in val_array if i not in val_skip]
    val_array = shuffle(val_array)

    ckpt_path = 'log/3d_weights-{val_loss:.4f}.hdf5'
    ckpt = tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                              monitor='val_loss',
                                              verbose=1,
                                              save_best_only=True,
                                              mode='min')

    model, stride = resnet50_32s(input_shape=(
        224, 224, 3), model_input=base_model)
    #adadelta = optimizers.Adadelta(lr=0.05, rho=0.9, decay=0.0)
    adam = optimizers.adam(lr=float("1e-4"))
    model.compile(optimizer=adam, loss=euc_joint_dist_loss,
                  metrics= [euc_joint_metrics_dist_keras, metrics_pckh])
    #lrate = LearningRateScheduler(step_decay)
    clr = CyclicLR(base_lr = float("1e-7"), max_lr = float("1e-4"), step_size = 2069, mode = 'triangular')
    model.summary()
    if ckpt_model != '':
        model.load_weights(ckpt_model)
    result = model.fit_generator(generator=pose3d_get_train_batch(train_array, 8, True),
                                 steps_per_epoch=2069,
                                 callbacks=[ckpt, clr],
                                 epochs=60000, verbose=1,
                                 validation_data=pose3d_get_train_batch(val_array, 8, False),
                                 validation_steps=593,
                                 workers=1)

def train_3d_16s(base_model = '', ckpt_model = 'None'):
    train_skip, val_skip = read_skip()
    train_array = list(range(1, 16545)) 
    train_array = [i for i in train_array if i not in train_skip]
    train_array = shuffle(train_array)

    val_array = list(range(1, 4740))
    val_array = [i for i in val_array if i not in val_skip]
    val_array = shuffle(val_array)

    ckpt_path = 'log/3d_weights_16s-{val_loss:.4f}.hdf5'
    ckpt = tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                              monitor='val_loss',
                                              verbose=1,
                                              save_best_only=True,
                                              mode='min')

    model, stride = resnet50_16s(input_shape=(
        224, 224, 3), model_input=base_model)
    #adadelta = optimizers.Adadelta(lr=0.05, rho=0.9, decay=0.0)
    adam = optimizers.adam(lr=float("1e-4"))
    model.compile(optimizer=adam, loss=euc_joint_dist_loss,
                  metrics= [euc_joint_metrics_dist_keras, metrics_pckh])
    #lrate = LearningRateScheduler(step_decay)
    clr = CyclicLR(base_lr = float("1e-7"), max_lr = float("1e-4"), step_size = 2069, mode = 'triangular')
    model.summary()
    if ckpt_model != 'None':
        model.load_weights(ckpt_model)
    result = model.fit_generator(generator=pose3d_get_train_batch(train_array, 8, True),
                                 steps_per_epoch=2069,
                                 callbacks=[ckpt, clr],
                                 epochs=60000, verbose=1,
                                 validation_data=pose3d_get_train_batch(val_array, 8, False),
                                 validation_steps=593,
                                 workers=1)

def train_3d_8s(base_model = '', ckpt_model = 'None'):
    train_skip, val_skip = read_skip()
    train_array = list(range(1, 16545)) 
    train_array = [i for i in train_array if i not in train_skip]
    train_array = shuffle(train_array)

    val_array = list(range(1, 4740))
    val_array = [i for i in val_array if i not in val_skip]
    val_array = shuffle(val_array)

    ckpt_path = 'log/3d_weights_8s-{val_loss:.4f}.hdf5'
    ckpt = tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                              monitor='val_loss',
                                              verbose=1,
                                              save_best_only=True,
                                              mode='min')

    model, stride = resnet50_8s(input_shape=(
        224, 224, 3), model_input=base_model)
    #adadelta = optimizers.Adadelta(lr=0.05, rho=0.9, decay=0.0)
    adam = optimizers.adam(lr=float("1e-4"))
    model.compile(optimizer=adam, loss=euc_joint_dist_loss,
                  metrics= [euc_joint_metrics_dist_keras, metrics_pckh])
    #lrate = LearningRateScheduler(step_decay)
    clr = CyclicLR(base_lr = float("1e-7"), max_lr = float("1e-4"), step_size = 2069, mode = 'triangular')
    model.summary()
    print(ckpt)
    if ckpt_model != 'None':
        model.load_weights(ckpt_model)
    result = model.fit_generator(generator=pose3d_get_train_batch(train_array, 8, True),
                                 steps_per_epoch=2069,
                                 callbacks=[ckpt, clr],
                                 epochs=60000, verbose=1,
                                 validation_data=pose3d_get_train_batch(val_array, 8, False),
                                 validation_steps=593,
                                 workers=1)

def train_3d_conv(base_model = '', ckpt_model = "None" ):
    img_path = "/media/disk1/human3.6/H36M-images/images/"
    pose_path = "/media/disk1/human3.6/Annot/"
    ckpt_path = 'log/3d_conv_weights-{val_loss:.4f}.hdf5'
    ckpt = tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                              monitor='val_loss',
                                              verbose=1,
                                              save_best_only=True,
                                              mode='min')
    model = make_seq_model(base_model)
    adam = optimizers.adam(lr=float("1e-4"))
    model.compile(optimizer=adam, loss=euc_joint_dist_loss,
                  metrics= [euc_joint_metrics_dist_keras, metrics_pckh])
    clr = CyclicLR(base_lr = float("1e-7"), max_lr = float("1e-4"), step_size = 45288, mode = 'triangular')
    model.summary()

    if ckpt_model != 'None':
        model.load_weights(ckpt_model)

    result = model.fit_generator(generator=get_3d_train_batch(img_path, pose_path),
                                 steps_per_epoch=45288,
                                 callbacks=[ckpt, clr],
                                 epochs=60000, verbose=1,
                                 validation_data=get_3d_Val_batch(img_path, pose_path),
                                 validation_steps=839,
                                 workers=1)


def main(argv):
    if argv[1] == "2d":
        train_2d()
    elif argv[1] == "2d_feature":
        feature_train_2d()
    elif argv[1] == "3d":
        if len(argv) == 3:
            train_3d(argv[2], '')
        elif len(argv) == 4:
            train_3d(argv[2], argv[3])
        else:
            print("WRONG ARGS NUM")
    elif argv[1] == "3d_16s":
        if len(argv) == 3:
            train_3d_16s(argv[2], 'None')
        elif len(argv) == 4:
            train_3d_16s(argv[2], argv[3])
        else:
            print("WRONG ARGS NUM")
    elif argv[1] == "3d_8s":
        if len(argv) == 3:
            train_3d_8s(argv[2], 'None')
        elif len(argv) == 4:
            train_3d_8s(argv[2], argv[3])
        else:
            print("WRONG ARGS NUM")
    elif argv[1] == "conv_3d":
        if len(argv) == 3:
            train_3d_conv(argv[2], "None")
        elif len(argv) == 4:
            train_3d_conv(argv[2], argv[3])
    else:
        print("you got wrong argv!!")
    
if __name__ == '__main__':
    main(sys.argv)
    
    # #result = euc_joint_dist_loss(y_pred, y_true)
    # pckh = metrics_pckh(y_pred, y_true)
    # sess = K.get_session()
    # print(sess.run(pckh))
