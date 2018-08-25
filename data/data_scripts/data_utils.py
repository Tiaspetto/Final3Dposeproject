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
import os
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
    heat_path = heat_path.format(root_path = os.path.abspath('.'), heat_path = heatmap_path, frames=str(picid).zfill(4))
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
                data.append(row)
            index += 1

        data =np.array(data)

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


if __name__ == '__main__':
    MPI_prerpocessing(True)
    MPI_prerpocessing(False)
