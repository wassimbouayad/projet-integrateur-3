import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
from isoparam import comformal_map
from os import system as sys
import os
import extract_dat as extract

def find_error(sim1, sim2):
    coord = sio.loadmat(sim1)
    points =coord['points']
    coord2 =sio.loadmat(sim2)
    points2 =coord2['points']
    # fig=plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2],c='b', s=0.05)
    # ax.scatter(points2[:, 0], points2[:, 1], points2[:, 2],c='b', s=0.05)
    # ax.scatter(points[0, 0], points[0, 1], points[0, 2],c='r')
    # ax.scatter(points2[0, 0], points2[0, 1], points2[0, 2],c='r')
    # ax.scatter(points[10, 0], points[10, 1], points[10, 2], c='g')
    # ax.scatter(points2[10, 0], points2[10, 1], points2[10, 2], c='g')
    # ax.scatter(points[4032, 0], points[4032, 1], points[4032, 2],c='g')
    # ax.scatter(points2[4032, 0], points2[4032, 1], points2[4032, 2],c='g')
    # plt.show()
    mse= np.sqrt(np.mean((points-points2)**2))
    return mse

# sim1= './original_sim.mat'
# sim2= './mapped_sim.mat'
# error= find_error(sim1, sim2)
# print(error)

# os.chdir('Calculix_interface')
# cmd = 'octave main.m'
# sys(cmd)


theta_arr = np.array([0, 2, 4, 6, 8, 10, 12, 14, 15])
# theta_arr = np.array([6])=
i = 0
errors = np.zeros_like(theta_arr, dtype=float)
curve_x_sim1 = np.zeros_like(theta_arr, dtype=float)
curve_y_sim1 = np.zeros_like(theta_arr, dtype=float)
curve_x_sim2 = np.zeros_like(theta_arr, dtype=float)
curve_y_sim2 = np.zeros_like(theta_arr, dtype=float)
for theta in theta_arr:
    curve_x_sim2[i], curve_y_sim2[i] = extract.extract_mapped_sim_dat(theta)
    curve_x_sim1[i], curve_y_sim1[i] = extract.extract_sim_dat(theta)
    sim1= './original_sim.mat'
    sim2= './mapped_sim.mat'
    error = find_error(sim1, sim2)
    print('angle: ')
    print(theta)
    print(error)
    errors[i] = error
    i=i+1

np.savetxt("errors.csv", errors, delimiter=",")
np.savetxt("curve_x_sim1.csv", curve_x_sim1, delimiter=",")
np.savetxt("curve_y_sim1.csv", curve_y_sim1, delimiter=",")
np.savetxt("curve_x_sim2.csv", curve_x_sim2, delimiter=",")
np.savetxt("curve_y_sim2.csv", curve_y_sim2, delimiter=",")
print(errors)
# error= find_error(sim1, sim2)
# theta = 40
# extract.extract_mapped_sim_dat(theta)
# extract.extract_sim_dat(theta)
# sim1= './original_sim.mat'
# sim2= './mapped_sim.mat'
# error = find_error(sim1, sim2)
# print(error)