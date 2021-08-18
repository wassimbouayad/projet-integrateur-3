import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
from isoparam import comformal_map
import conformal_map as cm
from os import system as sys
import os
import math as math
import curvature as cv
import time


###### legacy
# def extract_dat(dat_file):
#     with open(dat_file, 'r') as f:
#             lines = f.readlines()
#     keep=lines[3:]
#     # print(keep[0])
#     u=np.empty((len(keep),3))
#     print(dat_file)
#     print(len(keep))
#     for i in range(len(keep)):
#         temp=keep[i].split(' ')
#         # temp=temp.split('-')
#
#
#
#         while("" in temp) :
#             temp.remove("")
#         # print(temp)
#         try:
#             u[i,0]=float(temp[1])
#             u[i,1]=float(temp[2])
#             u[i,2]=float(temp[3])
#         except:
#             # print(temp)
#             continue
#     # fig = plt.figure()
#     # ax = Axes3D(fig)
#     # ax.scatter(u[:,0],u[:,1],u[:,2])
#     # plt.show()
#     return u


def extract_dat(dat_file):
    with open(dat_file, 'r') as f:
        lines = f.readlines()

    lookup = 'displacements'
    token = None
    for i in range(len(lines)):
        if lookup in lines[i]:
            token = i
    keep = lines[token + 2:]  #### keep infor from last increment only (this works for lin & non lin)

    # build displacement matrix
    u = np.empty((len(keep), 3))
    for i in range(len(keep)):
        temp = keep[i].split(' ')
        while ("" in temp):
            temp.remove("")
        u[i, 0] = float(temp[1])
        u[i, 1] = float(temp[2])
        u[i, 2] = float(temp[3])

    return u


def extract_sim_dat(theta):

    dat_file = './Calculix_interface/test.dat'
    # dat_file='./Calculix_interface/mapped.dat'
    coord = sio.loadmat('./mesh/mesh_8node_sym.mat')
    # coord=sio.loadmat('./mesh/mesh_8node_sym_mapped.mat')
    points = coord['points'][0][0]
    # points = coord['points']
    # print(points.shape)
    corners = cm.generate_new_map(theta)
    # corners = np.array(([-1.1, 0.9, 1.1, -0.9], [-1, -1, 1, 1]))
    mapped = np.zeros_like(points)
    transformed = comformal_map(points, corners)
    mapped[:, 0] = transformed[:, 0]
    mapped[:, 1] = transformed[:, 1]
    mapped[:, 2] = points[:, 2]
    # print(mapped)

    u = extract_dat(dat_file)
    coord = mapped + u

    # find curve
    find_x = sio.loadmat('./mid_x.mat')
    find_y = sio.loadmat('./mid_y.mat')
    mid_x = reformat(find_x['points'][0], coord)
    mid_y = reformat(find_y['points'][0], coord)

    zero_index_in_x, one_index_for_x, one_index_for_y, borders_index_x, borders_index_y = load_mats()

    reduced_curve_x = reformat(one_index_for_x, mid_x)
    reduced_curve_y = reformat(one_index_for_y, mid_y)

    borders_x = reformat(borders_index_x, reduced_curve_x)
    borders_y = reformat(borders_index_y, reduced_curve_y)

    zero_coordinates = np.zeros((1, 3))
    zero_coordinates = mid_x[zero_index_in_x, :]
    curve_x, curve_y = curve(borders_x, borders_y, zero_coordinates)

    # SHOW PLOT
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(coord[:, 0], coord[:, 1], coord[:, 2])
    ax.scatter(mid_x[:, 0], mid_x[:, 1], mid_x[:, 2], c='g')
    ax.scatter(mid_y[:, 0], mid_y[:, 1], mid_y[:, 2], c='g')
    ax.scatter(mid_x[zero_index_in_x, 0], mid_x[zero_index_in_x, 1], mid_x[zero_index_in_x, 2], c='red')
    ax.scatter(borders_x[:, 0], borders_x[:, 1], borders_x[:, 2], c='red')
    ax.scatter(borders_y[:, 0], borders_y[:, 1], borders_y[:, 2], c='red')
    plt.show()


    sio.savemat("mapped_sim.mat", mdict={'points': coord})
    return curve_x, curve_y


def run_octave_sim():
    os.chdir('Calculix_interface')
    cmd = 'octave main.m > /dev/null 2>&1'
    sys(cmd)
    os.chdir('..')
    return 1


def extract_mapped_sim_dat(theta):
    cm.generate_mapped_plate(theta)
    flag = 0
    flag = run_octave_sim()
    # print(flag)

    # print('next step')
    # print(os.getcwd())
    dat_file = './Calculix_interface/mapped.dat'

    coord = sio.loadmat('./mesh/mesh_8node_sym_mapped.mat')
    points = coord['points']

    u = extract_dat(dat_file)
    coord = points + u

    find_x = sio.loadmat('./mid_x.mat')
    find_y = sio.loadmat('./mid_y.mat')


    mid_x = reformat(find_x['points'][0], coord)
    mid_y = reformat(find_y['points'][0], coord)
    zero_index_in_x, one_index_for_x, one_index_for_y, borders_index_x, borders_index_y = load_mats()


    reduced_curve_x = reformat(one_index_for_x, mid_x)
    reduced_curve_y = reformat(one_index_for_y, mid_y)

    borders_x = reformat(borders_index_x, reduced_curve_x)
    borders_y = reformat(borders_index_y, reduced_curve_y)

    zero_coordinates = np.zeros((1, 3))
    zero_coordinates = mid_x[zero_index_in_x, :]
    curve_x, curve_y = curve(borders_x, borders_y, zero_coordinates)

    # SHOW PLOT
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(coord[:, 0], coord[:, 1], coord[:, 2])
    # ax.scatter(mid_x[:, 0], mid_x[:, 1], mid_x[:, 2], c='g')
    # ax.scatter(mid_y[:, 0], mid_y[:, 1], mid_y[:, 2], c='g')
    # ax.scatter(mid_x[zero_index_in_x, 0], mid_x[zero_index_in_x, 1], mid_x[zero_index_in_x, 2], c='purple')
    # ax.scatter(borders_x[:, 0], borders_x[:, 1], borders_x[:, 2], c='red')
    # ax.scatter(borders_y[:, 0], borders_y[:, 1], borders_y[:, 2], c='red')
    # plt.show()

    sio.savemat("original_sim.mat", mdict={'points': coord})
    return curve_x, curve_y

def find_principal_curve():
    cm.generate_mapped_plate(0)
    flag = 0
    flag = run_octave_sim()
    # print(flag)
    #
    # print('next step')
    # print(os.getcwd())
    dat_file = './Calculix_interface/mapped.dat'

    coord = sio.loadmat('./mesh/mesh_8node_sym_mapped.mat')
    points = coord['points']

    u = extract_dat(dat_file)
    coord = points + u
    find_x = np.where(np.abs(coord[:, 0]) < 10 ** (-3))
    find_y = np.where(np.abs(coord[:, 1]) < 10 ** (-3))

    sio.savemat("mid_x.mat", mdict={'points': find_x[0]})
    sio.savemat("mid_y.mat", mdict={'points': find_y[0]})

    mid_x = reformat(find_x[0], coord)
    mid_y = reformat(find_y[0], coord)


    zero_y, zero_x, one_y, one_x = find_zero(mid_x, mid_y)


    intersection_x = reformat(zero_y, mid_x)
    borders_x = reformat(one_y[0], mid_x)
    borders_y = reformat(one_x[0], mid_y)


    lower_x, lower_border_x = find_min(1, borders_x)
    upper_x, upper_border_x = find_max(1, borders_x)
    lower_y, lower_border_y = find_min(0, borders_y)
    upper_y, upper_border_y = find_max(0, borders_y)

    # creating array for the index of the borders
    x_coordinates = np.zeros(2, dtype=int)
    x_coordinates[0] = lower_border_x
    x_coordinates[1] = upper_border_x

    y_coordinates = np.zeros(2, dtype=int)
    y_coordinates[0] = lower_border_y
    y_coordinates[1] = upper_border_y

    # saving mats
    save_mats(zero_y, one_y, one_x, x_coordinates, y_coordinates)


    # SHOW PLOT
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(coord[:, 0], coord[:, 1], coord[:, 2])
    # Scatter the two principal curves
    ax.scatter(mid_x[:, 0], mid_x[:, 1], mid_x[:, 2], c='g')
    ax.scatter(mid_y[:, 0], mid_y[:, 1], mid_y[:, 2], c='g')
    # Scatter the points for the curves
    ax.scatter(intersection_x[:, 0], intersection_x[:, 1], intersection_x[:, 2], c='red')
    ax.scatter(lower_x[:, 0], lower_x[:, 1], lower_x[:, 2], c='red')
    ax.scatter(upper_x[:, 0], upper_x[:, 1], upper_x[:, 2], c='red')
    ax.scatter(lower_y[:, 0], lower_y[:, 1], lower_y[:, 2], c='red')
    ax.scatter(upper_y[:, 0], upper_y[:, 1], upper_y[:, 2], c='red')
    plt.show()

def find_zero(x, y):
    zero_y = np.where(np.abs(x[:, 1]) < 10 ** (-3))
    zero_x = np.where(np.abs(y[:, 0]) < 10 ** (-3))
    one_y = np.where(np.abs(x[:, 1]) < 2)
    one_x = np.where(np.abs(y[:, 0]) < 2)
    return zero_y, zero_x, one_y, one_x

def find_max(axis, array):
    max = np.amax(array[:, axis])
    find_max = np.where(array[:, axis] == max)
    upper_border = np.zeros((1, 3))
    upper_border = array[find_max, :]
    return upper_border[0], find_max[0]

def find_min(axis, array):
    min = np.amin(array[:, axis])
    find_min = np.where(array[:, axis] == min)
    lower_border = np.zeros((1, 3))
    lower_border = array[find_min, :]
    return lower_border[0], find_min[0]


def save_mats(zero_index, one_index_for_x, one_index_for_y, borders_x, borders_y):
    sio.savemat("intersection.mat", mdict={'points': zero_index[0][0]})
    sio.savemat("reduced_x.mat", mdict={'points': one_index_for_x[0]})
    sio.savemat("reduced_y.mat", mdict={'points': one_index_for_y[0]})
    sio.savemat("borders_coordinate_x.mat", mdict={'points': borders_x})
    sio.savemat("borders_coordinate_y.mat", mdict={'points': borders_y})

def load_mats():
    zero_index_in_x = sio.loadmat('intersection.mat')['points'][0][0]
    one_index_for_x = sio.loadmat('reduced_x.mat')['points'][0]
    one_index_for_y = sio.loadmat('reduced_y.mat')['points'][0]
    borders_index_x = sio.loadmat('borders_coordinate_x.mat')['points'][0]
    borders_index_y = sio.loadmat('borders_coordinate_y.mat')['points'][0]
    return zero_index_in_x, one_index_for_x, one_index_for_y, borders_index_x, borders_index_y

def reformat(array, unformated_array):
    i = 0
    reformated_array = np.zeros((np.shape(array)[0], 3))
    for a in array:
        reformated_array[i, :] = unformated_array[a, :]
        i += 1
    return reformated_array

def curve(borders_x, borders_y, zero):
    lower_border_x = borders_x[0, :]
    upper_border_x = borders_x[1, :]
    lower_border_y = borders_y[0, :]
    upper_border_y = borders_y[1, :]
    h_x = (upper_border_x[1] - lower_border_x[1]) / 2
    curve_x = (upper_border_x[2] - 2 * zero[2] + lower_border_x[2]) / (h_x ** 2)

    h_y = (upper_border_y[0] - lower_border_y[0]) / 2
    curve_y = (upper_border_y[2] - 2 * zero[2] + lower_border_y[2]) / (h_y ** 2)
    # print('Curve x:')
    # print(curve_x)
    # print('Curve y:')
    # print(curve_y)
    return curve_x, curve_y


# find_principal_curve()
# extract_mapped_sim_dat(10)
extract_sim_dat(0)



# def find_mid():
    # extract_mapped_sim_dat(0)
    # find_x = np.where(np.abs(coord[:, 0]) < 10 ** (-3))
    # find_y = np.where(np.abs(coord[:, 1]) < 10 ** (-3))
    # length_x = np.shape(find_x)
    # length_y = np.shape(find_y)
    # print(find_x[0])
    # print(np.shape(coord))
    # sio.savemat("mid_x.mat", mdict={'points': find_x[0]})
    # sio.savemat("mid_y.mat", mdict={'points': find_y[0]})
    # mid_x = np.zeros((length_x[1], 3))
    # mid_y = np.zeros((length_y[1], 3))
    # i = 0
    # for x in find_x[0]:
    #     mid_x[i, :] = coord[x, :]
    #     i += 1
    # j = 0
    # for y in find_y[0]:
    #     mid_y[j, :] = coord[y, :]
    #     j += 1
    #
    # sio.savemat("mid_x.mat", mdict={'points': mid_x})
    # sio.savemat("mid_y.mat", mdict={'points': mid_y})
    # print(find_x)
    # print(np.shape(find_y))