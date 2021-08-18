import numpy as np
import matplotlib.pyplot as plt

# xy represente la matrice des coordonnee du quadrilatere
xy = np.array(([0, 2, 1, -1, 1, 1.5, 0, -0.5, 0], [1, 1, -1, -1, 1.5, 0, -0.5, 0, 0]))

# nbksi = nombre element meme chose pour nbeta
nbksi = 40
nbeta = 40
# coordonnee ksi et eta
ksiVect = np.linspace(-1, 1, nbksi)
etaVect = np.linspace(-1, 1, nbeta)
# creation de la matrice de coordonnee ksi et eta
ksiMat, etaMat = np.meshgrid(ksiVect, etaVect)

# Creation d'une matrice de 0 de la meme dimension que la matrice KSI et ETA pour les coordonnee x et y
x, y = np.zeros_like(ksiMat), np.zeros_like(etaMat)

plt.figure()
plt.plot(xy[0, :], xy[1, :])
plt.show()

#  lagrange bi quadratique
def q1(param):
    Q = 1 / 2 * param * (param - 1)
    return Q


def q2(param):
    Q = (1 - param) * (1 + param)
    return Q


def q3(param):
    Q = 1 / 2 * param * (param + 1)
    return Q


for i in range(ksiMat.shape[0]):
    for j in range(ksiMat.shape[1]):
        N1 = q1(ksiMat[i, j]) * q1(etaMat[i, j])
        N2 = q3(ksiMat[i, j]) * q1(etaMat[i, j])
        N3 = q3(ksiMat[i, j]) * q3(etaMat[i, j])
        N4 = q1(ksiMat[i, j]) * q3(etaMat[i, j])
        N5 = q2(ksiMat[i, j]) * q1(etaMat[i, j])
        N6 = q3(ksiMat[i, j]) * q2(etaMat[i, j])
        N7 = q2(ksiMat[i, j]) * q3(etaMat[i, j])
        N8 = q1(ksiMat[i, j]) * q2(etaMat[i, j])
        N9 = q2(ksiMat[i, j]) * q2(etaMat[i, j])

        x1 = xy[0, 0]
        x2 = xy[0, 1]
        x3 = xy[0, 2]
        x4 = xy[0, 3]
        x5 = xy[0, 4]
        x6 = xy[0, 5]
        x7 = xy[0, 6]
        x8 = xy[0, 7]
        x9 = xy[0, 8]

        y1 = xy[1, 0]
        y2 = xy[1, 1]
        y3 = xy[1, 2]
        y4 = xy[1, 3]
        y5 = xy[1, 4]
        y6 = xy[1, 5]
        y7 = xy[1, 6]
        y8 = xy[1, 7]
        y9 = xy[1, 8]


        x[i, j] = N1 * x1 + N2 * x2 + N3 * x3 + N4 * x4 + N5 * x5 + N6 * x6 + N7 * x7 + N8 * x8 + N9 * x9
        y[i, j] = N1 * y1 + N2 * y2 + N3 * y3 + N4 * y4 + N5 * y5 + N6 * y6 + N7 * y7 + N8 * y8 + N9 * y9

plt.figure()
plt.scatter(x, y)
plt.show()


