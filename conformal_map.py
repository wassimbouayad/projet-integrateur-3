import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
import os
from isoparam import comformal_map
import math

def generate_mapped_plate(theta):

    dat_file='./Calculix_interface/test.dat'

    coord=sio.loadmat('./mesh/mesh_8node_sym.mat')
    points=coord['points'][0][0]
    cells=coord['cells'][0]
    corners = generate_new_map(theta)
    # corners = np.array(([-1-x, 1-x, 1+x, -1+x], [-1-y, -1-y, 1+y, 1+y]))
    mapped = np.zeros_like(points)
    transformed = comformal_map(points, corners)
    mapped[:, 0] = transformed[:, 0]
    mapped[:, 1] = transformed[:, 1]
    mapped[:, 2] = points[:, 2]

    # To compare the original plate with the mapped plate
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(mapped[:, 0], mapped[:, 1], mapped[:, 2],c='b', s=0.05)
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2],c='b', s=0.05)
    # ax.scatter(mapped[0, 0], mapped[0, 1], mapped[0, 2],c='r')
    # ax.scatter(points[0, 0], points[0, 1], points[0, 2],c='r')
    # ax.scatter(mapped[10, 0], mapped[10, 1], mapped[10, 2], c='g')
    # ax.scatter(points[10, 0], points[10, 1], points[10, 2], c='g')
    # ax.scatter(mapped[4032, 0], mapped[4032, 1], mapped[4032, 2],c='g')
    # ax.scatter(points[4032, 0], points[4032, 1], points[4032, 2],c='g')
    # plt.show()

    os.chdir('mesh')
    print(os.getcwd())
    sio.savemat("mesh_8node_sym_mapped.mat", mdict={"points": mapped, "cells": cells})
    print('Mesh succesfully saved')
    os.chdir('..')


# def generate_new_map(x, y):
#     corners = np.array(([-1 - x, 1 - x, 1 + x, -1 + x], [-1 - y, -1 - y, 1 + y, 1 + y]))
#     return corners

def generate_new_map(theta):
    # long cst
    corners = np.array(([-1, 1, 1 + 2 * np.tan(math.radians(theta)), -1 + 2 * np.tan(math.radians(theta))], [-1, -1, 1, 1]))
    # diag cst
    # corners = np.array(([-1, 1 - 2 * np.tan(math.radians(theta)), 1, -1 + 2 * np.tan(math.radians(theta))], [-1, -1, 1, 1]))
    return corners
# generate_mapped_plate(0)