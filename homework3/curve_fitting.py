import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter


def leastSquare(X, Y, return_error=False):
    args = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y).reshape(-1)
    if return_error:
        return args, np.sum(np.abs(Y - X[:, [0]] * args[0] - X[:, [1]] * args[1]))
    return args


def ransac(X, Y, iter=100, threshold=0.1):
    n, d, loss = Y.shape[0] // 5, Y.shape[0] // 2, 1e4
    simpleIndex, bestArgs = range(Y.shape[0]), np.zeros((X.shape[1], 1))
    np.random.seed(2019)
    for i in range(iter):
        index = np.random.choice(simpleIndex, size=n, replace=False)
        x, y = X[index, :], Y[index]
        currentArgs, currentLoss = leastSquare(x, y, return_error=True)
        inliers = []
        for j in simpleIndex:
            if np.abs(Y[j] - X[j][0] * currentArgs[0] - X[j][1] * currentArgs[1]) <= threshold:
                inliers.append(j)
        if len(inliers) >= d:
            x, y = X[inliers, :], Y[inliers]
            currentArgs, currentLoss = leastSquare(x, y, return_error=True)
            if currentLoss < loss:
                bestArgs, loss = currentArgs, currentLoss
    return bestArgs


def houghTransform(X, Y):
    size = Y.shape[0]
    H = []
    for i in range(size):
        for j in range(i + 1, size):
            x = X[i][0] - X[j][0]
            y = Y[j] - Y[i]
            xy = x / y
            sinTheta = xy / np.sqrt(1 + xy ** 2)
            rho = X[i][0] * np.sqrt(1 - sinTheta ** 2) + Y[i] * sinTheta
            key = "%f_%f" % (sinTheta, rho)
            H.append(key)
    H = list(Counter(H).keys())[0].split("_")
    return H


def funcTest():
    def dataGenerator(size=20, outlierSize=5, a=2, b=3, draw=True):
        np.random.seed(2019)
        Y = np.random.randn(size)
        X = (Y - b) / a
        outliers = np.random.randn(2, outlierSize)
        data = np.hstack([np.vstack([X, Y]), outliers]).T
        X, Y = np.hstack([data[:, [0]], np.ones((data.shape[0], 1))]), data[:, [1]]
        dataRange = np.asarray([min(data[:, 0]), max(data[:, 0])])
        if draw:
            plt.scatter(data[:, 0], data[:, 1], c="r")
        return X, Y, dataRange

    def curveDraw(dataRange, args, mode="rectangular"):
        args = np.asarray(args, dtype="float")
        X = dataRange
        Y = None
        if mode == "rectangular":
            Y = X * args[0] + args[1]
        elif mode == "polar":
            Y = -X * (np.sqrt(1 - args[0] ** 2) / args[0]) + args[1] / args[0]
        plt.plot(X, Y, c="g")
        plt.show()

    X, Y, dataRange = dataGenerator()
    args = houghTransform(X, Y)
    curveDraw(dataRange, args, mode="polar")


def imageTest():
    image = Image.open("./resource/myPainting.png").convert("L")
    image = np.array(image)
    newImage = image.copy()
    LoG = np.array([
        [0, 0, -1, 0, 0],
        [0, -1, -2, -1, 0],
        [-1, -2, 16, -2, -1],
        [0, -1, -2, -1, 0],
        [0, 0, -1, 0, 0],
    ])
    for i in range(2,518):
        for j in range(2, 518):
            newImage[i][j] = np.sum(LoG*image[i-2:i+3,j-2:j+3])
    newImage = Image.fromarray(newImage)
    newImage.show()


if __name__ == "__main__":
    imageTest()
