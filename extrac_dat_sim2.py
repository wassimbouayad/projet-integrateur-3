import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio

from isoparam import comformal_map

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

    lookup='displacements'
    token=None
    for i in range(len(lines)):
        if lookup in lines[i]:
            token=i
    keep=lines[token+2:] #### keep infor from last increment only (this works for lin & non lin)


    #build displacement matrix
    u=np.empty((len(keep),3))
    for i in range(len(keep)):
        temp=keep[i].split(' ')
        while("" in temp) :
            temp.remove("")
        u[i,0]=float(temp[1])
        u[i,1]=float(temp[2])
        u[i,2]=float(temp[3])

    return u
def extract_mapped_sim_dat():
    if __name__ == "__main__":
        dat_file='./Calculix_interface/mapped.dat'

        coord=sio.loadmat('./mesh/mesh_8node_sym_mapped.mat')
        points=coord['points']


        u=extract_dat(dat_file)
        coord = points + u
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(coord[:, 0], coord[:, 1], coord[:, 2])
        # ax.scatter(points[:, 0], points[:, 1], points[:, 2])
        plt.show()
        sio.savemat("original_sim.mat", mdict= {'points': coord})
