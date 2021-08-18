The data is too large to be added to git.
It can be found at :

https://drive.google.com/open?id=1hmDmzM7fTM73-WvFX9qrhLI_Nvxs8wUW

to acces the data the following piece of code can be useful:


import scipy.io as sio

fname='data/DATA_32_32_worker0.mat'
indat=sio.loadmat(fname)
gcurv=indat['X']
e11=indat['Y']
coord=indat['coord']
