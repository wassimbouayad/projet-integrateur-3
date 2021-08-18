from os import system as sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pymesh


def get_curv(mesh_file):
    # ##### for DiffGeoOps stopped using because of bug
    # cmd= 'python3 DiffGeoOps.py --mode 0 --op 2 %s.off'%(mesh_file)
    # sys(cmd)
    # curv_file='%s_KG.npy'%(mesh_file)
    # nodal_curv=np.load(curv_file)
    tri_mesh = pymesh.load_mesh(mesh_file)
    tri_mesh.add_attribute("vertex_gaussian_curvature")
    nodal_curv=tri_mesh.get_attribute("vertex_gaussian_curvature")


    mesh=sio.loadmat('./mesh/mesh_8node_sym.mat')
    quad=mesh['cells'][0][0]


    elem_curv=np.empty((quad.shape[0],))

    for i in range(quad.shape[0]):
        elem_curv[i]=(nodal_curv[quad[i,0]]+nodal_curv[quad[i,1]]+nodal_curv[quad[i,2]]+nodal_curv[quad[i,3]])/4

    return elem_curv,nodal_curv





# print(curv.shape)
#
# # cropped=curv #[1:33,1:33]
# print(cropped.shape)
# plt.imshow(cropped, origin='lower')
# plt.colorbar()
# plt.show()
