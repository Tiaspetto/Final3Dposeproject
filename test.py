from data.data_scripts.data_utils import *
from posenet_2d import *
from posenet_3d import *
from keras.models import Model, load_model
from keras import backend as K
from keras import optimizers
from keras.callbacks import LearningRateScheduler
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
import numpy as np
import os
from data.data_scripts.data_utils import *
from train import *
from keras.utils import plot_model

def euc_dist_keras(y_true, y_pred):
	return K.sqrt(K.sum(K.square(y_true - y_pred), axis = -1, keepdims = True))

def plot_pose(pose3D):

    buff_large = np.zeros((32, 3));
    buff_large[(1,2,3,6,7,8,13,14,17,18,19,25,26,27), :] = pose3D;

    pose3D = buff_large.transpose();
    kin = np.array([[13, 14], [13, 17], [17, 18], [18, 19], [13, 25], [25, 26], [26, 27], [0, 1], [1, 2], [2, 3], [0, 6], [6, 7], [7, 8]]);
    order = np.array([0,2,1]);

    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.view_init(azim=-90, elev=15)

    for link in kin:
       ax.plot(pose3D[0, link], pose3D[2, link], -pose3D[1, link], linewidth=5.0);

    ax.legend()
  
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')

    X = pose3D[0, :]
    Y = pose3D[2, :]
    Z = -pose3D[1, :]
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    plt.show()

def test_3d_conv(model_weights = "None" ):
    img_path = "D:/dissertation/data/human3.6/H36M-images/images/"
    pose_path = "D:/dissertation/data/human3.6/Annot/"
    ckpt_path = 'log/3d_conv_weights-{val_loss:.4f}.hdf5'
    ckpt = tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                              monitor='val_loss',
                                              verbose=1,
                                              save_best_only=True,
                                              mode='min')
    model = make_seq_model("None")
    adam = optimizers.adam(lr=float("1e-4"))
    model.compile(optimizer=adam, loss=euc_joint_dist_loss,
                  metrics= [euc_joint_metrics_dist_keras, metrics_pckh])
    clr = CyclicLR(base_lr = float("1e-7"), max_lr = float("1e-4"), step_size = 45288, mode = 'triangular')
    model.summary()
    plot_model(model, to_file='model.png')
    model.load_weights('D:/dissertation/model_data/3d_conv_weights-0.2407.hdf5')

    result = model.predict_generator(generator=get_3d_Test_batch(img_path, pose_path),
                                 steps = 15,
                                 verbose = 1,
                                 workers=1)

    for i in range(15):
        pose3D = np.reshape(result[i], (14, 3))
        plot_pose(pose3D*1000.0)

def plot_3d_Pose_grid():
    action_s = {2:"Directions", 3:"Discussion", 4:"Eating", 5:"Greeting", 6:"Phoning", 7:"Posing", 8:"Purchases", 9:"Sitting", 10:"SittingDown", 11:"Smoking", 12:"Photo", 13:"Waiting", 14:"Walking", 15:"WalkDog", 16:"WalkTogether"}
    plt.figure()
    for i in range(1,16):
        plt.subplot(3,5,i)
        predict_path = "C:/Users/Administrator/Desktop/3dpose/Figure_{num}.png"
        predict_path = predict_path.format(num = str(i))
        img=mpimg.imread(predict_path)
        fig = imshow(img)
        plt.title(action_s[i+1])
        plt.axis('off')
    plt.show()

    plt.figure()
    for i in range(1,16):
        plt.subplot(3,5,i)
        predict_path = "C:/Users/Administrator/Desktop/3dpose/g{num}.jpg"
        predict_path = predict_path.format(num = str(i))
        img=mpimg.imread(predict_path)
        plt.title(action_s[i+1])
        imshow(img)
    plt.show()
def show_sequence():
    plt.figure()
    for i in range(1,10):
        plt.subplot(1,9,i)
        predict_path = "C:/Users/Administrator/Desktop/seqence/{num}.jpg"
        if i == 9:
            predict_path = "C:/Users/Administrator/Desktop/seqence/{num}.png"
        predict_path = predict_path.format(num = str(i))
        img=mpimg.imread(predict_path)
        plt.axis('off')
        imshow(img)
    plt.show()
def print_16s():
    model, stride = resnet50_16s(input_shape=(
        224, 224, 3), model_input='')
    #adadelta = optimizers.Adadelta(lr=0.05, rho=0.9, decay=0.0)
    adam = optimizers.adam(lr=float("1e-4"))
    model.compile(optimizer=adam, loss=euc_joint_dist_loss,
                  metrics= [euc_joint_metrics_dist_keras, metrics_pckh])
    #lrate = LearningRateScheduler(step_decay)
    clr = CyclicLR(base_lr = float("1e-7"), max_lr = float("1e-4"), step_size = 2069, mode = 'triangular')
    model.summary()
    plot_model(model, to_file='model_16s.png')

def show_lsp():
    plt.figure()
    for i in range(1,8):
        plt.subplot(1,8,i)
        predict_path = "C:/Users/Administrator/Desktop/H36/{num}.jpg"
        predict_path = predict_path.format(num = str(i))
        img=mpimg.imread(predict_path)
        plt.axis('off')
        imshow(img)
    plt.show()
if __name__ == '__main__':
    
    #show_lsp()
    #print_16s()
    #show_sequence()
    #plot_3d_Pose_grid()
    model = PoseNet_50(input_shape=(224, 224, 3))
    adadelta = optimizers.Adadelta(lr = 0.05, rho = 0.9, decay = 0.0)
    model.compile(optimizer = adadelta, loss = euc_dist_keras,
                  metrics=['mae'])
    #model.load_weights('D:/dissertation/log/weights-118.9920.hdf5')
    model.summary()
    plot_model(model, to_file='2dmodel.png')
    # model.load_weights('D:/dissertation/log/weights-118.9920.hdf5')
    
    # img = cv2.imread("D:/dissertation/data/ECCV18_Preprocessed/train/00001.jpg")
    # img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)

    # prediction = model.predict(np.array([read_image(1989), read_image(2)]))

    # result = prediction[0]
    # print(np.shape(prediction))
    # #result = result.transpose((2, 0, 1))
    # heatmap = np.clip(1 - np.amax(result, axis=2), 0.0, 1.2)

    # model = make_seq_model(base_model)
    # adam = optimizers.adam(lr=float("1e-4"))
    # model.compile(optimizer=adam, loss=['mae'],metrics= ['mae'])
    # model.load_weights('D:/dissertation/model_data/3d_conv_weights-0.2407.hdf5')
    #model.summary()
    

    # imgs = []
    # for i in range(1,9):
    #     path = "G:/{img}.jpg"
    #     path = path.format(img = str(i))
    #     img=mpimg.imread(path)
    #     imgs.append(img)

    # plt.figure()
    # for i in range(8):
    #     plt.subplot(2,4,i+1)
    #     imshow(imgs[i])
    #     plt.axis('off')
    # plt.show()
    # plt.figure()
    # for i in range(15):
    #     plt.subplot(4,4,i+1)
    #     if i == 14:
    #       imshow(read_image(1989))
    #     else:
    #       imshow(result[:,:,i])
    # plt.show()

    # debug_read_heat_info(1989)

    #pre_processing_lsp("D:/dissertation/data/lsp_dataset/joints.mat", [1], (255,255), debug_flag=True)
    #get_MPII_data(True)
    # train_array = list(range(1, 16545)) 
    # for index in train_array:
    #     img_path = "{root_path}{data_path}{pid}.jpg"
    #     img_path = img_path.format(root_path = os.path.abspath('.'), data_path = "/data/ECCV18_Preprocessed/train/", pid  = str(index).zfill(5))
    #     if not os.path.isfile(img_path):
    #        print(img_path)
    #        f=open('no_bbox.txt','a')
    #        text = str(index).zfill(5) + ","
    #        f.writelines(text)
    #        f.close()
    # val_array = list(range(1, 4740)) 
    # for index in val_array:
    #    img_path = "{root_path}{data_path}{pid}.jpg"
    #    img_path = img_path.format(root_path = os.path.abspath('.'), data_path = "/data/ECCV18_Preprocessed/val/", pid  = str(index).zfill(5))
    #    if not os.path.isfile(img_path):
    #        print(img_path)
    #        f=open('no_val_bbox.txt','a')
    #        text = str(index).zfill(5) + ","
    #        f.writelines(text)
    #        f.close()

    #read_pose_data(1, True)
