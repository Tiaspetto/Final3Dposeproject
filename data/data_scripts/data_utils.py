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
body_part = 14
sigma = 8.0

lsp_img_source_path = "/data/lsp_dataset/images/"
heatmap_path = "/data/lsp_dataset/heat/"

ECCV_source_path = "/data/ECCV18Challenge/"

def put_heatmap(heatmap, plane_idx, center):
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

def get_heatmap(target_size, joint_list, height, width):
    heatmap = np.zeros((body_part, height, width), dtype=np.float32)

    print(np.shape(heatmap))

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
                result.append(cv2.resize(heatmap[i], target_size, interpolation=cv2.INTER_LINEAR))
        
        result = np.asarray(result)
    return result.astype(np.float32)

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

def get_picture_info(picid):
    ima_path = os.path.abspath('.') + lsp_img_source_path 
    img_path = ima_path+"im{frames}.jpg"
    img_path = img_path.format(frames=str(picid).zfill(4))

    img = cv2.imread(img_path)
    height, width, _ = img.shape
    return height, width

#img = cv2.imread("D:/dissertation/data/lsp_dataset/images/im0002.jpg")

def pre_processing_lsp(file_name, picture_ids, target_size, debug_flag = False):
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
                plt.subplot(4,4,i+1)
                imshow(heat[i]/255.0)
            plt.show()
        else:  
            heat_path = heatmap_path+"im{frames}.mat"
            heat_path = heat_path.format(frames=str(picid).zfill(4))
            scipy.io.savemat(heat_path, {'heat':heat})
  
def read_heat_info(picid):
    heat_path = os.path.abspath('.') + heatmap_path
    heat_path = heat_path+"im{frames}.mat"
    heat_path = heat_path.format(frames=str(picid).zfill(4))
    data = scipy.io.loadmat(heat_path)
    heat = data['heat']
    heat = heat.transpose((1, 2, 0))
    return heat

def read_image(picid, dataset = "lsp", datatype = "Train"):
    ima_path = ""
    if dataset == "lsp":
        ima_path = os.path.abspath('.') + lsp_img_source_path 
        img_path = ima_path+"im{frames}.jpg"
        img_path = img_path.format(frames=str(picid).zfill(4))
    else

    img = cv2.imread(img_path)

    img = cv2.resize(img, (224,224), interpolation=cv2.INTER_LINEAR)

    return img

def debug_read_heat_info(picid):
    heat_path = os.path.abspath('..') + "/lsp_dataset/heat/"
    heat_path = heat_path+"im{frames}.mat"
    heat_path = heat_path.format(frames=str(picid).zfill(4))
    print(heat_path)
    data = scipy.io.loadmat(heat_path)
    heat = data['heat']
    plt.figure()
    for i in range(body_part):
        plt.subplot(4,4,i+1)
        imshow(heat[i]/255.0)
    plt.show()
    
    return heat

def print_path():
    print(os.path.abspath('.')+lsp_img_source_path ,os.path.abspath('.')+heatmap_path)
      
if __name__ == '__main__':
    debug_read_heat_info(1)
