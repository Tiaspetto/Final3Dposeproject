import scipy.io
import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import os
from mpl_toolkits.mplot3d import Axes3D
import math
from matplotlib.pyplot import imshow
import csv
body_part = 14
sigma = 32.0

lsp_img_source_path = "/data/lsp_dataset/images/"
heatmap_path = "/data/lsp_dataset/heat/"

ECCV_source_train_path = "/data/ECCV18_Challenge/Train/"
ECCV_source_val_path = "/data/ECCV18_Challenge/Val/"

ECCV_joints = [1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16]

MPII_source_pose_path = "/data/MPII/mpii_human_pose_v1/pose/"
MPII_source_train_path = "/data/MPII/mpii_human_pose_v1/pose/train/"
MPII_source_val_path = "/data/MPII/mpii_human_pose_v1/pose/val/"
MPII_source_img_path = "/data/MPII/mpii_human_pose_v1/images/"
MPII_joints = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 8, 9]

Human36_joints = [1, 2, 3, 6, 7, 8, 13, 14, 16, 18, 19, 24, 26, 27]

def put_heatmap(heatmap, plane_idx, center):
    #print(plane_idx, center, sigma)
    center_x, center_y = center
    _, height, width = heatmap.shape[:3]

    th = 4.6052
    delta = math.sqrt(th * 2)

    x0 = int(max(0, center_x - delta * sigma))
    y0 = int(max(0, center_y - delta * sigma))

    x1 = int(min(width, center_x + delta * sigma))
    y1 = int(min(height, center_y + delta * sigma))

    for y in range(y0, y1):
        for x in range(x0, x1):
            d = (x - center_x) ** 2 + (y - center_y) ** 2
            exp = d / 2.0 / sigma / sigma
            if exp > th:
                continue
            heatmap[plane_idx][y][x] = max(
                heatmap[plane_idx][y][x], math.exp(-exp))
            heatmap[plane_idx][y][x] = min(heatmap[plane_idx][y][x], 1.0)


def get_heatmap(target_size, joint_list, height, width):
    heatmap = np.zeros((body_part, height, width), dtype=np.float32)

    #print(np.shape(heatmap))test_joints
    count = 0
    point = []
    for i in range(body_part):
        put_heatmap(heatmap, i, (joint_list[0][i], joint_list[1][i]))

        heatmap = heatmap.transpose((1, 2, 0))

        # background
        #heatmap[:, :, -1] = np.clip(1 - np.amax(heatmap, axis=2), 0.0, 1.0)

        heatmap = heatmap.transpose((2, 0, 1))

        result = []
        if target_size:
            for i in range(body_part):
                result.append(cv2.resize(
                    heatmap[i], target_size, interpolation=cv2.INTER_LINEAR))

        result = np.asarray(result)
    return result.astype(np.float32)


def re_orgnize(samples):
    x = []
    y = []
    result = []
    count = 0
    for joint in samples:
        if count % 2 == 0:
            x.append(float(joint))
        else:
            y.append(float(joint))
        count += 1
    result.append(x)
    result.append(y)

    return result


def get_picture_info(picid):
    img_path = "{root_path}{data_path}im{frames}.jpg"
    img_path = img_path.format(root_path = os.path.abspath('.'), data_path = lsp_img_source_path, frames=str(picid).zfill(4))

    img = cv2.imread(img_path)
    height, width, _ = img.shape
    return height, width

#img = cv2.imread("D:/dissertation/data/lsp_dataset/images/im0002.jpg")


def pre_processing_lsp(file_name, picture_ids, target_size, debug_flag=False):
    annot = scipy.io.loadmat(file_name)
    joints = annot['joints']

    for picid in picture_ids:
        height, width = get_picture_info(picid)
        sample = joints[:, :, picid-1]
        heat = get_heatmap(target_size, sample, height, width)

        print(np.shape(heat))

        if debug_flag == True:
            plt.figure()
            for i in range(body_part):
                plt.subplot(4, 4, i+1)
                imshow(heat[i]/255.0)
            plt.show()
        else:
            heat_path = "{root_path}{data_path}im{frames}.mat"
            heat_path = heat_path.format(root_path = os.path.abspath('.'), heat_path = heatmap_path, frames=str(picid).zfill(4))
            scipy.io.savemat(heat_path, {'heat': heat})


def read_heat_info(picid):
    heat_path = "{root_path}{data_path}im{frames}.mat"
    heat_path = heat_path.format(root_path = os.path.abspath('.'), data_path = heatmap_path, frames=str(picid).zfill(4))
    data = scipy.io.loadmat(heat_path)
    heat = data['heat']
    heat = heat.transpose((1, 2, 0))
    return heat


def read_image(picid, dataset = "lsp", isTrain = True):
    if dataset == "lsp":
        img_path = "{root_path}{data_path}im{pid}.jpg"
        img_path = img_path.format(root_path = os.path.abspath('.'), data_path = lsp_img_source_path, pid = str(picid).zfill(4))
    elif dataset == "ECCV":
        if isTrain == True:
            img_path = "{root_path}{data_path}{pid}.jpg"
            img_path = img_path.format(root_path = os.path.abspath('.'), data_path = "/data/ECCV18_Preprocessed/train/", pid  = str(picid).zfill(5))
            #print(img_path)
        else:
            img_path = "{root_path}{data_path}{pid}.jpg"
            img_path = img_path.format(root_path = os.path.abspath('.'), data_path = "/data/ECCV18_Preprocessed/val/", pid  = str(picid).zfill(5))
            #print(img_path)
   
    #print(img_path)

    img = cv2.imread(img_path)

    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    
    img = img * (2.0 / 255.0) - 1.0   
    #print(np.shape(img))
    return img


def debug_read_heat_info(picid):
    heat_path = "{root_path}{data_path}im{frames}.mat"
    heat_path = heat_path.format(root_path = os.path.abspath('..'), data_path = "/lsp_dataset/heat/", frames=str(picid).zfill(4))
    print(heat_path)
    data = scipy.io.loadmat(heat_path)
    heat = data['heat']
    plt.figure()
    for i in range(body_part):
        plt.subplot(4, 4, i+1)
        imshow(heat[i]/255.0)
    plt.show()

    return heat

def read_pose_data(picid, isTrain):
    pose_path = "{root_path}{data_path}{folder_type}{pid}{data_type}"
    if isTrain:
        pose_path = pose_path.format(root_path = os.path.abspath('.'), data_path = ECCV_source_train_path, folder_type = "POSE/", pid = str(picid).zfill(5), data_type = ".csv")
    else:
        pose_path = pose_path.format(root_path = os.path.abspath('.'), data_path = ECCV_source_val_path, folder_type = "POSE/", pid = str(picid).zfill(5), data_type = ".csv")

    with open(pose_path,'r') as csvfile:
        reader = csv.reader(csvfile)
        data = []
        index = 0
        for row in reader:
            if index in ECCV_joints:
                row = [float(x) for x in row]
                data.append(row)
            index += 1

        data =np.array(data)
        data = np.reshape(data, (42, ))
        data = data * (1.0/1000.0)

    #print(data)
    return(data)

def get_MPII_data(isTrain):
    data_path = "{root_path}{data_path}{file_name}"
    if isTrain:
        data_path = data_path.format(root_path = os.path.abspath('.'), data_path = MPII_source_pose_path, file_name = "train_joints.csv")

        with open(data_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]        
        return rows
    else:
        data_path = data_path.format(root_path = os.path.abspath('.'), data_path = MPII_source_pose_path, file_name = "test_joints.csv")
        with open(data_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]
        return rows

def MPI_read_img(file_name):
    data_path = os.path.abspath('.') + MPII_source_img_path
    data_path = data_path + file_name
    img = cv2.imread(data_path)

    height, width, _ = img.shape
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    img = img * (2.0 / 255.0) - 1.0
    return (img, height, width)

def MPI_process_heat(data_array, height, width, file_name, isTrain):
    joint_list = re_orgnize(data_array)
    #print(joint_list)
    x = []
    y = []
    result = []
    for index in MPII_joints:
        x.append(joint_list[0][index])
        y.append(joint_list[1][index])

    result.append(x)
    result.append(y)
    data_path = ""
    if isTrain:
        data_path = os.path.abspath('.') + MPII_source_train_path
    else:
        data_path = os.path.abspath('.') + MPII_source_val_path

    data_path = data_path + file_name
    heat = get_heatmap((224, 224), result, height, width)
    scipy.io.savemat(data_path, {'heat': heat})
    #return heat

def MPI_prerpocessing(isTrain):
    data = get_MPII_data(isTrain)
    for row in data:
        img, height, width = MPI_read_img(row[0])
        MPI_process_heat(row[1:33], height, width, row[0], isTrain)

def MPI_read_heat_info(isTrain, file_name):
    data_path = ""
    if isTrain:
        data_path = os.path.abspath('.') + MPII_source_train_path
    else:
        data_path = os.path.abspath('.') + MPII_source_val_path

    data_path = data_path + file_name

    data = scipy.io.loadmat(data_path)
    heat = data['heat']
    
    heat = heat.transpose((2, 1, 0))
    
    #print(np.shape(heat))
    return heat

def print_path():
    print(os.path.abspath('.')+lsp_img_source_path,
          os.path.abspath('.')+heatmap_path)

#==========================================================================


#folder_name = 's_{:02d}_act_{:02d}_subact_{:02d}_ca_{:02d}'.format(subject, action, subaction, camera)

def human36_read_joints(file_path):
    data = scipy.io.loadmat(file_path)
    
    pose = data['data']
    #print(np.max(pose), np.min(pose))
    return pose

def human36_pose_preprocess(data):
    data = np.reshape(data, (32, 3))
    pose_data = []
    for i in range(0, 32):
        if i in Human36_joints:
            #print(i)
            pose_data.append(data[i,:]-data[0,:])

    pose_data = np.array(pose_data)
    pose_data = pose_data * (1.0/1000.0)
    pose_data = np.reshape(pose_data, (42, ))
    return pose_data


def get_3d_train_batch(img_path, pose_path):
    action_s = {2:"Directions", 3:"Discussion", 4:"Eating", 5:"Greeting", 6:"Phoning", 7:"Posing", 8:"Purchases", 9:"Sitting", 10:"SittingDown", 11:"Smoking", 12:"Photo", 13:"Waiting", 14:"Walking", 15:"WalkDog", 16:"WalkTogether"}
    camera_s = [".54138969", ".55011271", ".58860488", ".60457274"]
    sub_s = [" 1", " 2", " 3", ""]
    subject_list = [1, 5, 6, 7, 8, 9, 11]
    action_list = np.arange(2, 17)
    subaction_list = np.arange(1, 3)
    camera_list = np.arange(1, 5)
    for subject in subject_list:
        for action in action_list:
            for subaction in subaction_list:
                for camera in camera_list:
                    folder_name = 's_{:02d}_act_{:02d}_subact_{:02d}_ca_{:02d}'.format(subject, action, subaction, camera)
                    path = img_path + folder_name
                    meta_name = path + '/matlab_meta.mat'
                    train_start_index = 41
                    X_data_quene = []
                    
                    if not os.path.exists(meta_name):
                        print (meta_name, 'not exists!')
                        continue

                    meta = scipy.io.loadmat(meta_name)
                    num_images = meta['num_images']  

                    # query path
                    exist_path = [] 
                    for i in range(0,4):
                        pose_file_name = "S{subject}/{action}{subindex}{camindex}.cdf.mat".format(subject = subject, action = action_s[action], subindex = sub_s[i], camindex = camera_s[camera-1])
                        pose_file_path = pose_path + pose_file_name
                        if os.path.exists(pose_file_path):
                            exist_path.append(pose_file_path)


                    pose_file_path = ""

                    if len(exist_path)!=2:
                        print(exist_path, folder_name,'ecifficient path!!')
                        #assert len(exist_path) < 2, folder_name
                        continue
                    else:
                        pose_file_path = exist_path[subaction - 1]

                    if not os.path.exists(pose_file_path):
                        print(pose_file_path, subaction , 'not exists!')


                    pose_data = human36_read_joints(pose_file_path)
                    pose_data = pose_data[0,0]

                    num_images = min(num_images, np.shape(pose_data)[0])

                    while train_start_index < num_images:
                        for i in range(0,8):
                            if len(X_data_quene) == 8:
                                X_data_quene.pop(0)

                            train_start_index += 5
                            if train_start_index > num_images:
                                break
                            img_name = img_path + folder_name + '/' + '{}_{:06d}.jpg'.format(folder_name, train_start_index)
                            if not os.path.exists(img_name):
                                print(pose_file_path, num_images, img_name, 'not exists!')
                                continue
                            else: 
                                img = cv2.imread(img_name)
                                img = img * (2.0 / 255.0) - 1.0
                                X_data_quene.append(img)
                            



                        if len(X_data_quene) <8:
                            break
                        X_data = np.array(X_data_quene)
                        if train_start_index > np.shape(pose_data)[0]:
                            print(pose_file_path, folder_name, train_start_index, np.shape(pose_data)[0])

                        Y_data = pose_data[train_start_index-1, :]
                        Y_data = human36_pose_preprocess(Y_data)

                        train_start_index += 5
                        X_data = np.reshape(X_data, (1, 8, 224, 224, 3))
                        Y_data = np.reshape(Y_data, (1, 42))

                        yield X_data, Y_data

def get_3d_Val_batch(img_path, pose_path):
    action_s = {2:"Directions", 3:"Discussion", 4:"Eating", 5:"Greeting", 6:"Phoning", 7:"Posing", 8:"Purchases", 9:"Sitting", 10:"SittingDown", 11:"Smoking", 12:"TakingPhoto", 13:"Waiting", 14:"Walking", 15:"WalkingDog", 16:"WalkTogether"}
    camera_s = [".54138969", ".55011271", ".58860488", ".60457274"]
    sub_s = [" 1", " 2", " 3", ""]
    subject_list = [1, 5, 6, 7, 8, 9, 11]
    action_list = np.arange(2, 17)
    subaction_list = np.arange(1, 3)
    camera_list = np.arange(1, 5)
    for subject in subject_list:
        for action in action_list:
            for subaction in subaction_list:
                for camera in camera_list:
                    folder_name = 's_{:02d}_act_{:02d}_subact_{:02d}_ca_{:02d}'.format(subject, action, subaction, camera)
                    path = img_path + folder_name
                    meta_name = path + '/matlab_meta.mat'
                    data_start_index = 36
                    pre_load_index = 1
                    X_data_quene = []
                    # query path

                    exist_path = [] 
                    for i in range(0,4):
                        pose_file_name = "S{subject}/{action}{subindex}{camindex}.cdf.mat".format(subject = subject, action = action_s[action], subindex = sub_s[i], camindex = camera_s[camera-1])
                        pose_file_path = pose_path + pose_file_name
                        if os.path.exists(pose_file_path):
                            exist_path.append(pose_file_path)


                    pose_file_path = ""

                    if len(exist_path)!=2:
                        print(exist_path, folder_name,'ecifficient path!!')
                        #assert len(exist_path) < 2, folder_name
                        continue
                    else:
                        pose_file_path = exist_path[subaction - 1]

                    if not os.path.exists(pose_file_path):
                        print(pose_file_path, subaction , 'not exists!')


                    pose_data = human36_read_joints(pose_file_path)
                    pose_data = pose_data[0,0]

                    while pre_load_index < data_start_index:
                        img_name = img_path + folder_name + '/' + '{}_{:06d}.jpg'.format(folder_name, pre_load_index)
                        img = cv2.imread(img_name)
                        img = img * (2.0 / 255.0) - 1.0
                        X_data_quene.append(img)
                        pre_load_index += 5

                    while data_start_index <= 36:
                        if len(X_data_quene) == 8:
                            X_data_quene.pop(0)
                        img_name = img_path + folder_name + '/' + '{}_{:06d}.jpg'.format(folder_name, data_start_index)
                        img = cv2.imread(img_name)
                        img = img * (2.0 / 255.0) - 1.0
                        X_data_quene.append(img)

                        X_data = np.array(X_data_quene)
                        Y_data = pose_data[data_start_index-1, :]
                        Y_data = human36_pose_preprocess(Y_data)

                        data_start_index += 5
                        
                        X_data = np.reshape(X_data, (1, 8, 224, 224, 3))
                        Y_data = np.reshape(Y_data, (1, 42))
                        yield X_data, Y_data



if __name__ == '__main__':
    #MPI_prerpocessing(True)
    #MPI_prerpocessing(False)
    images_path = "D:\\dissertation\\data\\human3.6\\H36M-images\\images\\"
    pose_path = "D:\\dissertation\\data\\human3.6\\Annot\\"

    data = human36_read_joints("D:\\dissertation\\data\\human3.6\\Annot\\S11\\Directions 1.54138969.cdf.mat")
    data = data[0,0]
    print(np.shape(data))

    Y_data = data[1, :]

    human36_pose_preprocess(Y_data)
