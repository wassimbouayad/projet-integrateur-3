#<editor-fold PLOT DEFORMED PLATE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
def plot_mesh(coord,quad,elem_curv,projection=False,proj_lines=False):

    def getColors(m,a):
        b=m.to_rgba(a)
        return [(i[0],i[1],i[2]) for i in b]

    elements=quad

    fig = plt.figure()
    ax = Axes3D(fig)
    graph=ax.scatter(coord[:,0],coord[:,1],coord[:,2], s=0.0005, c='k') #c=coord[:,2] ,cmap='jet'


    np.save('./mesh/quad', quad)
    m = cm.ScalarMappable(cmap=cm.Blues)
    m.set_array(elem_curv)
    m.set_clim(vmin=np.min(elem_curv),vmax=np.max(elem_curv))
    curv_color=getColors(m,elem_curv)

    if projection==True:
        offset=np.min(coord[:,2])*3.0

        plane=np.ones((coord[:,0].shape[0],))*offset

        graph=ax.scatter(coord[:,0],coord[:,1],plane, s=0.5, c='#A9A9A9')

    for i in range(int(elements.shape[0])):
        x=[coord[elements[i,0],0],coord[elements[i,1],0],coord[elements[i,2],0],coord[elements[i,3],0],coord[elements[i,0],0]]
        y=[coord[elements[i,0],1],coord[elements[i,1],1],coord[elements[i,2],1],coord[elements[i,3],1],coord[elements[i,0],1]]
        z=[coord[elements[i,0],2],coord[elements[i,1],2],coord[elements[i,2],2],coord[elements[i,3],2],coord[elements[i,0],2]]





        verts=[list(zip(x,y,z))]

        ## SET element color to Gaussian Curvature
        cb=1
        q=Poly3DCollection(verts, linewidths=1)


        q.set_facecolor(curv_color[i])


        ax.add_collection3d(q)

        ax.add_collection3d(Poly3DCollection(verts, facecolors=curv_color[i], linewidths=1))  #'#A9A9A9'

        ax.add_collection3d(Line3DCollection(verts, colors='k', linewidths=1, linestyles='-'))



        if projection==True:
            #####project flat shape on offset plane
            x=[coord[elements[i,0],0],coord[elements[i,1],0],coord[elements[i,2],0],coord[elements[i,3],0],coord[elements[i,0],0]]
            y=[coord[elements[i,0],1],coord[elements[i,1],1],coord[elements[i,2],1],coord[elements[i,3],1],coord[elements[i,0],1]]
            z=[coord[elements[i,0],2]*0+offset,coord[elements[i,1],2]*0+offset,coord[elements[i,2],2]*0+offset,coord[elements[i,3],2]*0+offset,coord[elements[i,0],2]*0+offset]
            verts=[list(zip(x,y,z))]
            ax.add_collection3d(Poly3DCollection(verts, facecolors='#A9A9A9', linewidths=1, alpha=0.5))
            ax.add_collection3d(Line3DCollection(verts, colors='#A9A9A9', linewidths=1, linestyles='-'))


    ax.view_init(elev=50, azim=55)
    #<editor-fold  Projection lines ************************************************

    if proj_lines==True:

        x=[coord[0,0],coord[0,0]]
        y=[coord[0,1],coord[0,1]]
        z=[coord[0,2],coord[0,2]*0+offset]

        verts=[list(zip(x,y,z))]
        ax.add_collection3d(Line3DCollection(verts, colors='k', linewidths=1, linestyles='--'))
        x=[coord[1,0],coord[1,0]]
        y=[coord[1,1],coord[1,1]]
        z=[coord[1,2],coord[1,2]*0+offset]

        verts=[list(zip(x,y,z))]
        ax.add_collection3d(Line3DCollection(verts, colors='k', linewidths=1, linestyles='--'))

        x=[coord[2,0],coord[2,0]]
        y=[coord[2,1],coord[2,1]]
        z=[coord[2,2],coord[2,2]*0+offset]

        verts=[list(zip(x,y,z))]
        ax.add_collection3d(Line3DCollection(verts, colors='k', linewidths=1, linestyles='--'))

        x=[coord[3,0],coord[3,0]]
        y=[coord[3,1],coord[3,1]]
        z=[coord[3,2],coord[3,2]*0+offset]

        verts=[list(zip(x,y,z))]
        ax.add_collection3d(Line3DCollection(verts, colors='k', linewidths=1, linestyles='--'))

    #</editor-fold>
    if cb==1:
        fig.colorbar(m,shrink=0.8)

    ax.grid(False)
    plt.axis('off')
    plt.show()
    # plt.show(block=False)
    # plt.pause(10)
    # plt.close('all')
    #</editor-fold>
