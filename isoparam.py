import numpy as np
import matplotlib.pyplot as plt

def comformal_map(points, corners):
    # xy represente la matrice des coordonnee du quadrilatere
    xy = corners

    ksiVect = np.zeros((points.shape[0], 1))
    etaVect = np.zeros((points.shape[0], 1))
    ksiVect[:,0] = points[:, 0]
    etaVect[:,0] = points[:, 1]
    # print(ksiVect.shape)
    # print('points',points.shape)

    # Creation d'une matrice de 0 de la meme dimension que la matrice KSI et ETA pour les coordonnee x et y
    x, y =np.zeros_like(ksiVect), np.zeros_like(etaVect)

    # Plotting the desired shape
    # plt.figure()
    # plt.scatter(xy[0, :], xy[1, :])
    # plt.show()


    for i in range(ksiVect.shape[0]):

        N1 = 1 / 4 * (1 - ksiVect[i]) * (1 - etaVect[i])
        N2 = 1 / 4 * (1 + ksiVect[i]) * (1 - etaVect[i])
        N3 = 1 / 4 * (1 + ksiVect[i]) * (1 + etaVect[i])
        N4 = 1 / 4 * (1 - ksiVect[i]) * (1 + etaVect[i])


        x1 = xy[0, 0]
        x2 = xy[0, 1]
        x3 = xy[0, 2]
        x4 = xy[0, 3]

        y1 = xy[1, 0]
        y2 = xy[1, 1]
        y3 = xy[1, 2]
        y4 = xy[1, 3]

        x[i] = N1 * x1 + N2 * x2 + N3 * x3 + N4 * x4
        y[i] = N1 * y1 + N2 * y2 + N3 * y3 + N4 * y4


    # plotting the new mapped plate
    # plt.figure()
    # plt.scatter(x, y)
    # # print(x.shape)
    # plt.show()

    result = np.concatenate((x, y), axis=1)
    # print(result.shape)

    return result

