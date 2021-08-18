# -*- coding: utf-8 -*-
import pygmsh
from pygmsh.helpers import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import meshio
import numpy as np
from scipy.spatial import distance


def func_mesh(geo, lcar):
    geom = pygmsh.built_in.Geometry()



    poly = geom.add_polygon(geo, lcar)
    try:
        surf=geom.set_transfinite_surface(poly.surface)
    except:
        print('Transfinite impossible, unstructured mesh used \n')


    #recombine triangles into quads
    geom.add_raw_code("Recombine Surface {%s};" % poly.surface.id)
    # geom.add_raw_code('Mesh.Algorithm = 8;')
    geom.add_raw_code('Mesh.SecondOrderIncomplete=1;')
    geom.add_raw_code('Mesh.ElementOrder = 2;')

    #generate mesh
    points, cells, _, _, _ = pygmsh.generate_mesh(geom, dim=2, verbose=False)
    #
    # print(points.shape)
    # print(cells['quad8'].shape)
    ### Save quad mesh
    # meshio.write_points_cells("quads.vtu", points, cells)

    # print(cells)
    # Extract quad elements from mesh
    quad=cells['quad8']


    # Define triangle element array to build mesh with 4 triangles per quad (top to bottom)
    tri=np.zeros((2*quad.shape[0],3), dtype=int)
    k=0
    # get quad centroid
    # centroid=np.empty((quad.shape[0],3))
    # for i in range(quad.shape[0]):
    #     centroid[i,:]=(points[quad[i,0]]+points[quad[i,1]]+points[quad[i,2]]+points[quad[i,3]])/4
    #
    # ind=np.lexsort((centroid[:,0],centroid[:,1],centroid[:,2]))
    # quad=quad[ind]

    # split quad elements in top to bottom triangles
    for e in range(quad.shape[0]):
        # print(e+k)
        tri[e+k]=quad[e][[0,1,3]]
        k=k+1
        tri[e+k]=quad[e][[1,2,3]]
        # k=k+1
        # tri[e+k]=quad[e][[0,1,2]]
        # k=k+1
        # tri[e+k]=quad[e][[0,2,3]]
        # print(k)

    #### clear and redefine Geometry (without this step,mesh will be supperimposed)
    # geom = pygmsh.built_in.Geometry()
    # poly = geom.add_polygon(geo, lcar)
    # geom.add_point([ 0.0,  0.0, 0.0], lcar)
    #mesh Geometry with triangles
    # tpoints, cells, _, _, _ = pygmsh.generate_mesh(geom, dim=2, verbose=False)
    #force mesh elements to follow tri element structure computed earlier
    cells['triangle']=tri
    elements=tri
    cells={}
    cells['triangle']=tri

    #save mesh
    meshio.write_points_cells("./mesh/tri.off", points, cells)
    # the mesh is now [[UT_top_element],
    #                  [LT_top_element]
    #                  [LT_bottom_element],
    #                  [UT_bottom_element],
    #                   .
    #                   .
    #                   .]
    with open("./mesh/tri.off", 'r') as f:
            lines = f.readlines()
            # remove spaces
    lines = [line.replace('  ', ' ') for line in lines]
    # print(lines)
    temp=lines[5:]
    lines=[lines[0]]+[lines[3]]+temp
    # finally, write lines in the file
    with open("./mesh/tri.off", 'w') as f:
        f.writelines(lines)
    return points, cells, elements, quad


if __name__ == "__main__":
    import meshio
    import numpy as np
    import scipy.io as sio
    # meshio.write_points_cells("quads.vtu", *func_mesh(geo, lcar))

    Lx=18
    Ly=18
    geo=[
        [ 0.0, 0.0, 0.0],
        [ Lx, 0.0, 0.0],
        [  Lx, Ly, 0.0],
        [ 0.0, Ly, 0.0]
        ]
    lcar=1.0

    # Lx=18
    # Ly=18
    # geo=[
    #     [ -Lx, -Ly, 0.0],
    #     [-Lx, Ly, 0.0],
    #     [  Lx, Ly, 0.0],
    #     [ Lx, -Ly, 0.0]
    #     ]
    # lcar=1.0


    points, cells, elements, quad= func_mesh(geo, lcar)
    # fig, ax=plt.subplots(1)
    pts = np.zeros((1,), dtype=np.object)
    quads = np.zeros((1,), dtype=np.object)
    # plt.figure()
    # for j in range(points.shape[0]):
    #     plt.scatter(points[j,0],points[j,1])
    #     plt.show(block=False)
    #     plt.pause(0.005)
    # for i in range(quad.shape[0]):
    #     plt.scatter(points[quad[i],0],points[quad[i],1])
    #     plt.show(block=False)
    #     plt.pause(0.005)
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()
    pts[0]=points
    quads[0]=quad
    sio.savemat('mesh_8node.mat',mdict={'points': pts, 'cells':quads})
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(points[:,0],points[:,1],points[:,2], s=0.5)
    # ax.scatter(np.mean(points[:,0]),np.mean(points[:,1]),np.mean(points[:,2]))
    # print(points)
    plt.show()
    # print(np.where(~points.any(axis=1))[0])
