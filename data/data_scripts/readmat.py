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
import cdflib

body_part = 32
width = 1002
height = 1000
sigma = 8.0


target = np.zeros((width, height), dtype=np.float32)
def put_heatmap(heatmap, plane_idx, center, sigma):
    print(plane_idx, center, sigma)
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
            heatmap[plane_idx][y][x] = max(heatmap[plane_idx][y][x], math.exp(-exp))
            heatmap[plane_idx][y][x] = min(heatmap[plane_idx][y][x], 1.0)

def get_heatmap(target_size, joint_list):
        heatmap = np.zeros((body_part+1, height, width), dtype=np.float32)

        print(np.shape(heatmap))

        count = 0
        point = []
        for i in range(body_part):
          put_heatmap(heatmap, i, (joint_list[0][i], joint_list[1][i]), sigma)

        heatmap = heatmap.transpose((1, 2, 0))

        # background
        heatmap[:, :, -1] = np.clip(1 - np.amax(heatmap, axis=2), 0.0, 1.0)

        heatmap = heatmap.transpose((2, 0, 1))
        #if target_size:
            #heatmap = cv2.resize(heatmap, target_size)

        return heatmap.astype(np.float32)

def re_orgnize(samples):
    x = []
    y = []
    result = []
    count = 0
    for joint in samples:
        if count % 2 == 0:
            x.append(joint)
        else:
            y.append(joint)
        count += 1
    result.append(x)
    result.append(y)

    return result


if __name__ == '__main__':

    cdf_file = cdflib.CDF('D:/dissertation/data/human3.6/S1/MyPoseFeatures/D2_Positions/Discussion 1.54138969.cdf')
    data = cdf_file.varget("Pose")

    #cdf_file 
    print(np.shape(data))

    #img = cv2.imread("D:/dissertation/data/lsp_dataset/images/im0002.jpg")

    #height, width, _ = img.shape

    #print(img.shape)

    #annot = scipy.io.loadmat('D:/dissertation/data/lsp_dataset/joints.mat')

    #joints = annot['joints']

    #sample = joints[:, :, 1]

    #print(np.shape(sample[0]))

    samples = data[0]
    sample = samples[155, :]
    sample = re_orgnize(sample)

    print(sample)
    heat = get_heatmap(np.shape(target), sample)

    plt.figure()
    for i in range(body_part+1):
        plt.subplot(5,7,i+1)
        imshow(heat[i]/255.0)

    plt.show()

    #imgpts, __ = cv2.projectPoints(axis, rvec, tvec, rgb_mtx, rgb_dist)

 
