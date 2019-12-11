import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from skimage import io, transform
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


def houghTransform(X, Y, shape=(520, 520), thre=100):
    def hough_line(img):
        thetas = np.deg2rad(np.arange(-90.0, 90.0))
        width, height = img.shape
        diag_len = int(np.ceil(np.sqrt(width * width + height * height)))
        rhos = np.linspace(-diag_len, diag_len, diag_len * 2)
        cos_t, sin_t = np.cos(thetas), np.sin(thetas)
        num_thetas = len(thetas)
        accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
        y_idxs, x_idxs = np.nonzero(img)
        for i in range(len(x_idxs)):
            x = x_idxs[i]
            y = y_idxs[i]
            for t_idx in range(num_thetas):
                rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len)
                accumulator[rho, t_idx] += 1
        return accumulator, thetas, rhos

    image = np.zeros(shape)
    for i in range(Y.shape[0]):
        image[int(X[i][0])][int(Y[i][0])] = 1
    return hough_line(image)


def simpleHoughTransform(X, Y):
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
    args = simpleHoughTransform(X, Y)
    curveDraw(dataRange, args, mode="polar")


def imageTest():
    def dataGenerator():
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
        data = [[], []]
        for i in range(2, 518):
            for j in range(2, 518):
                log = np.sum(LoG * image[i - 2:i + 3, j - 2:j + 3])
                newImage[i][j] = 0 if log < 127 else 255
                if newImage[i][j] == 255:
                    data[0].append(i)
                    data[1].append(j)
        # newImage = Image.fromarray(newImage)
        # newImage.save("./result/myPainting1.png")
        data = np.array(data).T
        X, Y = np.hstack([data[:, [0]], np.ones((data.shape[0], 1))]), data[:, [1]]
        return X, Y, newImage

    def curveDraw(image, accumulator, thetas, rhos, threshold, path="./result/houghTransform2.png"):
        io.imshow(image)
        row, col = image.shape
        for _, angle, dist in zip(*transform.hough_line_peaks(accumulator, thetas, rhos, threshold=threshold)):
            if angle == 0:
                x = dist
                plt.plot((x, x), (0, row), '-c')
                continue
            y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
            y1 = (dist - col * np.cos(angle)) / np.sin(angle)
            plt.plot((0, col), (y0, y1), '-c')
        plt.axis((0, col, row, 0))
        plt.savefig(path)
        plt.close()

    X, Y, newImage = dataGenerator()
    accumulator, thetas, rhos = houghTransform(X, Y)
    curveDraw(newImage, accumulator, thetas, rhos, threshold=100)


if __name__ == "__main__":
    imageTest()
