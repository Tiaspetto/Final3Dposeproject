import scipy.io
import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import os
from mpl_toolkits.mplot3d import Axes3D

IMG_SIZE = 512
eps = 1e-4
J = 28
MPII_order = [10, 9, 8, 11, 12, 13, 14, 15, 1, 0, 4, 3, 2, 5, 6, 7,16,17,18,19,20,21,22,23,24,25,26,27]
edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5],
         [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15],
         [6, 8], [8, 9]]


def show_3d(points, c='b'):
  points = points.reshape(J, 3)
  x, y, z = np.zeros((3, J))
  for j in range(J):
    x[j] = points[j, 0]
    y[j] = - points[j, 1]
    z[j] = - points[j, 2]
  ax.scatter(z, x, y, c=c)
  # ax.scatter(z[k], x[k], y[k], c = (0, 0, 0))
  #for e in edges:
    #ax.plot(z[e], x[e], y[e], c=c)


def show_2d(img, points, c=(255, 0, 0)):
  points = points.reshape(J, 2)
  for j in range(J):
    cv2.circle(img, (int(points[j, 0]), int(points[j, 1])), 3, c, -1)
  #for e in edges:
    #cv2.line(img, (int(points[e[0], 0]), int(points[e[0], 1])),
                   #(int(points[e[1], 0]), int(points[e[1], 1])), c, 2)


if __name__ == '__main__':

    image = np.zeros((512,512,3), np.uint8)
    #image = cv2.resize(image, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
    annot = scipy.io.loadmat('G:/mpi_inf_3dhp/S1/Seq1/annot.mat')
    univ_annot3 = annot["univ_annot3"]
    annot3 = annot["annot3"]
    annot2 = annot["annot2"]

    # image = image.transpose(1, 2, 0).copy()
    # for h in range(image.shape[0]):
    #     for w in range(image.shape[1]):
    #         if image[h][w][0] < 0.4 + eps and image[h][w][1] < 0.4 + eps and image[h][w][2] < 0.4 + eps and image[h][w][0] > 0.4 - eps and image[h][w][1] > 0.4 - eps and image[h][w][2] > 0.4 - eps:
    #             image[h][w][0], image[h][w][1], image[h][w][2] = 0, 0, 0
    raw_joints = univ_annot3[0][0][2500]
    joint = []
    for i in MPII_order:
    	jox = raw_joints[3*i]
    	joy = raw_joints[3*i+1]
    	joz = raw_joints[3*i+2]

    	joint.append([jox, joy, joz])
    joint = np.asarray(joint)
    print(joint)
    print(np.shape(joint))

    fig = plt.figure()
    ax = fig.add_subplot((111), projection='3d')
    ax.set_xlabel('z') 
    ax.set_ylabel('x') 
    ax.set_zlabel('y')
    oo = 2
    xmax, ymax, zmax, xmin, ymin, zmin = oo, oo, oo, -oo, -oo, -oo
    joint = (joint - joint[6]) / 5 + IMG_SIZE / 2
    show_3d(joint)
    show_2d(image, joint[:, :2])
    cv2.imshow('img', image)
    plt.show()
    cv2.waitKey()
