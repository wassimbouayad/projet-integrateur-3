from func_mesh import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import meshio
from os import system as sys
import os
# from get_curv import *
from extract_dat import *
import numpy as np
from sklearn.neighbors import NearestNeighbors
from plot_mesh import *
import matplotlib.image as mpimg
import multiprocessing as mp
import time


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()
def unique_rows(A, return_index=False, return_inverse=False):
    """
    Similar to MATLAB's unique(A, 'rows'), this returns B, I, J
    where B is the unique rows of A and I and J satisfy
    A = B[J,:] and B = A[I,:]

    Returns I if return_index is True
    Returns J if return_inverse is True
    """
    A = np.require(A, requirements='C')
    assert A.ndim == 2, "array must be 2-dim'l"

    B = np.unique(A.view([('', A.dtype)]*A.shape[1]),
               return_index=return_index,
               return_inverse=return_inverse)

    if return_index or return_inverse:
        return (B[0].view(A.dtype).reshape((-1, A.shape[1]), order='C'),) \
            + B[1:]
    else:
        return B.view(A.dtype).reshape((-1, A.shape[1]), order='C')

def simulate(patterns,wid,points, cells, elements, quad,points_sym, cells_sym, elements_sym, quad_sym):
    start_time = time.time()
    pats=np.zeros((1,), dtype=np.object)
    X=np.zeros((patterns.shape[0],), dtype=np.object)
    Y=np.zeros((patterns.shape[0],), dtype=np.object)
    cloud=np.zeros((patterns.shape[0],), dtype=np.object)

    # print(patterns.shape[0])
    for i in range(patterns.shape[0]):
        # print(i)
        # patterns[i][:,:]=1

        message='********** starting job '+str(i)+'/'+str(patterns.shape[0])+' for worker '+str(wid)+' **********\n'
        print(message)

        # # ##### for complementary pattern uncomment 3 lines bellow
        # comp_pat=np.ones(patterns[i].shape)
        # comp_pat=comp_pat-patterns[i]
        # patterns[i]=comp_pat

        # patterns[i]=np.flip(patterns[i],axis=1)
        patterns[i]=np.pad(patterns[i],((2,0),(0,2)),'constant',constant_values=(0,0))
        # patterns[i]=patterns[i].T
        #format patterns as object arrays to save in dict to interface with matlab/octave

        pats[0]=patterns[i]

        # plt.figure()
        # plt.imshow(patterns[i])
        jobname='test'
        pat_name='./patterns/pattern_%s.mat'%jobname
        sio.savemat(pat_name,mdict={'pattern': pats})

        # load=20

        cmd='octave ./Calculix_interface/main.m'

        # cmd='octave ./Calculix_interface/main.m %s '%jobname
        sys(cmd)


        ''' Moved Calculix interface code to ./Calculix_interface
            So Calculix is now called from the Ocatave code.
            Calculix cannot be called with a .inp file in another Dir
            That's inconvinient.'''

        # os.environ['OMP_NUM_THREADS']='8' ###run on 8 CPU cores


        # cmd='ccx ./Calculix_interface/%s > /dev/null 2>&1'%jobname
        # sys(cmd)


        dat_file="./Calculix_interface/%s.dat"%jobname
        u=extract_dat(dat_file)
        deformed=points+u
        deformed_before_sym=points+u


        #### Apply SYM On nodes
        symx=np.copy((-1)*deformed)
        symx[:,0]=np.copy(deformed[:,0])
        symx[:,2]=np.copy(deformed[:,2])
        deformed=np.concatenate((np.copy(deformed),symx),axis=0)
        symy=np.copy((-1)*deformed)
        symy[:,1]=np.copy(deformed[:,1])
        symy[:,2]=np.copy(deformed[:,2])
        deformed=np.concatenate((np.copy(deformed),symy),axis=0)

        _, idx = np.unique(deformed, return_index=True, axis=0)
        deformed=deformed[np.sort(idx)]




        ref=points
        symx_p=np.copy((-1)*ref)
        symx_p[:,0]=np.copy(ref[:,0])
        symx_p[:,2]=np.copy(ref[:,2])
        ref=np.concatenate((np.copy(ref),symx_p),axis=0)
        symy_p=np.copy((-1)*ref)
        symy_p[:,1]=np.copy(ref[:,1])
        symy_p[:,2]=np.copy(ref[:,2])
        ref=np.concatenate((np.copy(ref),symy_p),axis=0)


        _, idx = np.unique(ref, return_index=True, axis=0)
        ref=ref[np.sort(idx)]


        '''######################### IMPORTANT NOTE ##############################
        The following line is to neglect x & y dispacements for curvtaure computation
        IRL, 3D scan data does not give info on inplane growth (we do not use DIC)
        Therefore the NN should not learn from such info'''

        # deformed[:,:2]=ref[:,:2]



        _,id=nearest_neighbor(ref, points_sym)
        # _,id=nearest_neighbor(deformed, points_sym)
        copy=np.empty((deformed.shape[0],3))

        for j in range(deformed.shape[0]):
            copy[id[j],:]=deformed[j,:]
        deformed=copy


        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.scatter(deformed[:,0],deformed[:,1],deformed[:,2],s=0.5)
        # ax.scatter(points_sym[:,0],points_sym[:,1],points_sym[:,2],s=0.5, c='r')
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # plt.show()

        quadz = np.zeros((1,), dtype=np.object)
        quadz[0]=quad_sym
        sym_mesh='mesh_8node_sym_worker%d.mat'%wid
        # sio.savemat(sym_mesh,mdict={'points': deformed, 'cells':quadz})

        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.scatter(deformed[:,0],deformed[:,1],deformed[:,2])
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # # for i in range(quad_sym.shape[0]):
        # #         ax.scatter(deformed[quad_sym[i],0],deformed[quad_sym[i],1],deformed[quad_sym[i],2])
        # #         plt.show(block=False)
        # #         plt.pause(0.005)
        #
        # plt.show()

        mesh_file="./mesh/deformed_worker%d.off"%wid

        meshio.write_points_cells(mesh_file, deformed, cells_sym)  #add worker number to file name or pid
        # #Fix OFF formatting to work with DiffGeoOps
        # with open(mesh_file_off, 'r') as f:    #add worker number to file name or pid
        #         lines = f.readlines()
        #         # remove spaces
        # lines = [line.replace('  ', ' ') for line in lines]
        # # print(lines)
        # temp=lines[5:]
        # lines=[lines[0]]+[lines[3]]+temp
        # # finally, write lines in the file
        # with open(mesh_file_off, 'w') as f:     #add worker number to file name or pid
        #     f.writelines(lines)



        # elem_curv,nodal_curv=get_curv(mesh_file)    #add worker number to file name or pid
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.scatter(deformed[:,0],deformed[:,1],deformed[:,2],c=nodal_curv)
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # # for i in range(quad_sym.shape[0]):
        # #         ax.scatter(deformed[quad_sym[i],0],deformed[quad_sym[i],1],deformed[quad_sym[i],2])
        # #         plt.show(block=False)
        # #         plt.pause(0.005)
        #
        # plt.show()
        # print(elem_curv)
        # curv=np.reshape(elem_curv,(36,36))
        # print(curv[17,17])
        # curv=np.flip(curv,axis=0)
        # cropped=curv[2:34,2:34]
        # patterns[i]=mpimg.imread('Maze2_0.png')
        # patterns[i]=np.dot(patterns[i][...,:3], [0.2989, 0.5870, 0.1140])
        # patterns[i]=np.pad(patterns[i],((2,0),(0,2)),'constant',constant_values=(0,0))
        # print(patterns[i].shape)
        plt.figure()
        plt.imshow(patterns[i],cmap='Blues')
        plt.show()
        p=np.flip(patterns[i],axis=0)
        p2=np.flip(p,axis=1)
        mirr=np.pad(p2,((18,0),(0,18)),'symmetric')
        # print(patterns[i].shape)
        # plt.figure()
        # plt.imshow(mirr)
        # plt.show()
        # print(mirr.shape)
        mirr=mirr[2:34,2:34]
        # print(mirr.shape)
        # mirr=np.pad(mirr,(18,0),'symmetric')
        # plt.figure(2)
        # plt.imshow(mirr, origin='lower',cmap='Blues')
        # plt.colorbar()
        # # plt.show()
        # #
        # plt.figure(3)
        # plt.imshow(cropped, origin='lower',cmap='Blues')
        # plt.colorbar()
        # plt.show()

        # X[i]=cropped
        Y[i]=mirr
        cloud[i]=np.copy(deformed)
        #
        # ################# show Gaussian curvature on deformed mesh
        # pad_curv=np.pad(cropped, ((2, 2), (2, 2)), 'constant',constant_values=(0,0))#np.min(cropped)), np.min(np.min(cropped))))
        # pad_curv=pad_curv.reshape((pad_curv.shape[0]*pad_curv.shape[0],))
        # plot_mesh(deformed,quad_sym,pad_curv,projection=True,proj_lines=False)

        # # print(curv[:16,20:].shape)
        # curv_ns=curv[1:17,18:-1]
        # plt.figure()
        # plt.imshow(curv_ns)
        # # plt.show()


        ######## show Gaussian cruvature on deformed mesh without symmetry
        # pad_curv_ns=np.pad(curv_ns, ((2, 0), (0, 2)), 'constant',constant_values=(np.min(np.min(curv_ns)), np.min(np.min(curv_ns))))
        # pad_curv_ns=pad_curv_ns.reshape((pad_curv_ns.shape[0]*pad_curv_ns.shape[1],))
        # plot_mesh(deformed_before_sym,quad,pad_curv_ns)


        ###### show pattern on deformed mesh
        pad_curv=np.pad(mirr, ((2, 2), (2, 2)), 'constant',constant_values=(0,0))#np.min(cropped)), np.min(np.min(cropped))))
        pad_curv=pad_curv.reshape((pad_curv.shape[0]*pad_curv.shape[0],))
        plot_mesh(deformed,quad_sym,pad_curv)


        # deformed[:,2]=0
        # plot_mesh(deformed,quad_sym,pad_curv,projection=True,proj_lines=True)
    # save_name='DATA_32_32_worker%d.mat'%wid
    save_name='./data/DATA_32_32_L20_com_worker%d.mat'%wid
    sio.savemat(save_name,mdict={'X': X, 'Y':Y,'coord':cloud})
    end_time=time.time()
    print("--- Worker %s done Compute time : %s seconds --- \n" % (wid,end_time - start_time))

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out



###### quarter plate mesh
Lx=18
Ly=18
geo=[
    [ 0.0, 0.0, 0.0],
    [ 0.0, Ly, 0.0],
    [  Lx, Ly, 0.0],
    [ Lx, 0.0, 0.0]
    ]
lcar=1.0

points, cells, elements, quad=func_mesh(geo, lcar)
pts = np.zeros((1,), dtype=np.object)
quads = np.zeros((1,), dtype=np.object)
pts[0]=points
quads[0]=quad
sio.savemat('./mesh/mesh_8node.mat',mdict={'points': pts, 'cells':quads})

#####MESH for SYM
geo=[
    [ -Lx, -Ly, 0.0],
    [-Lx, Ly, 0.0],
    [  Lx, Ly, 0.0],
    [ Lx, -Ly, 0.0]
    ]
lcar=1.0
points_sym, cells_sym, elements_sym, quad_sym=func_mesh(geo, lcar)

pts_sym = np.zeros((1,), dtype=np.object)
quads_sym= np.zeros((1,), dtype=np.object)
pts_sym[0]=points_sym
quads_sym[0]=quad_sym

sio.savemat('./mesh/mesh_8node_sym.mat',mdict={'points': pts_sym, 'cells':quads_sym})




#### load patterns
pat=sio.loadmat('./patterns/pat_gen_16_16.mat')
patterns=pat['pattern'][0]

n_workers=1

out=chunkIt(patterns,n_workers)

for wid in range(n_workers):

    p=mp.Process(target=simulate, args=(out[wid],wid,points, cells, elements, quad,points_sym, cells_sym, elements_sym, quad_sym))
    p.start()
