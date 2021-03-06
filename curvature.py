import numpy as np

def surfature(X,Y,Z):
    # where X, Y, Z matrices have a shape (lr+1,lb+1)
    lr = X.shape[0] -1
    lb = X.shape[1] -1
    #First Derivatives
    Xv,Xu=np.gradient(X)
    Yv,Yu=np.gradient(Y)
    Zv,Zu=np.gradient(Z)
    #Second Derivatives
    Xuv,Xuu=np.gradient(Xu)
    Yuv,Yuu=np.gradient(Yu)
    Zuv,Zuu=np.gradient(Zu)
    Xvv,Xuv=np.gradient(Xv)
    Yvv,Yuv=np.gradient(Yv)
    Zvv,Zuv=np.gradient(Zv)
    #Reshape to 1D vectors
    nrow=(lr+1)*(lb+1) #total number of rows after reshaping
    Xu=Xu.reshape(nrow,1)
    Yu=Yu.reshape(nrow,1)
    Zu=Zu.reshape(nrow,1)
    Xv=Xv.reshape(nrow,1)
    Yv=Yv.reshape(nrow,1)
    Zv=Zv.reshape(nrow,1)
    Xuu=Xuu.reshape(nrow,1)
    Yuu=Yuu.reshape(nrow,1)
    Zuu=Zuu.reshape(nrow,1)
    Xuv=Xuv.reshape(nrow,1)
    Yuv=Yuv.reshape(nrow,1)
    Zuv=Zuv.reshape(nrow,1)
    Xvv=Xvv.reshape(nrow,1)
    Yvv=Yvv.reshape(nrow,1)
    Zvv=Zvv.reshape(nrow,1)
    Xu=np.c_[Xu, Yu, Zu]
    Xv=np.c_[Xv, Yv, Zv]
    Xuu=np.c_[Xuu, Yuu, Zuu]
    Xuv=np.c_[Xuv, Yuv, Zuv]
    Xvv=np.c_[Xvv, Yvv, Zvv]
    #% First fundamental Coeffecients of the surface (E,F,G)
    E=np.einsum('ij,ij->i', Xu, Xu)
    F=np.einsum('ij,ij->i', Xu, Xv)
    G=np.einsum('ij,ij->i', Xv, Xv)
    m=np.cross(Xu,Xv,axisa=1, axisb=1)
    p=np.sqrt(np.einsum('ij,ij->i', m, m))
    n=m/np.c_[p,p,p]
    #% Second fundamental Coeffecients of the surface (L,M,N)
    L= np.einsum('ij,ij->i', Xuu, n)
    M= np.einsum('ij,ij->i', Xuv, n)
    N= np.einsum('ij,ij->i', Xvv, n)
    #% Gaussian Curvature
    K=(L*N-M**2)/(E*G-L**2)
    K=K.reshape(lr+1,lb+1)
    #% Mean Curvature
    H = (E*N + G*L - 2*F*M)/(2*(E*G - F**2))
    H = H.reshape(lr+1,lb+1)
    #% Principle Curvatures
    Pmax = H + np.sqrt(H**2 - K)
    Pmin = H - np.sqrt(H**2 - K)
    return Pmax,Pmin


def re_organize(M):
   dim = np.shape(M)
   n = np.sqrt(dim[0])
   X = np.reshape(M[:, 0], (n, n))
   Y = np.reshape(M[:, 1], (n, n))
   Z = np.reshape(M[:, 2], (n, n))
   return X, Y, Z
