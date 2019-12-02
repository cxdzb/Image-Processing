from PIL import Image
import numpy as np


def paddingFilling(image, m=3, n=3):
    M, N = m // 2, n // 2
    up, down = image[0], image[-1]
    for i in range(M):
        image = np.vstack([up, image, down])
    left, right = image[:, [0]], image[:, [-1]]
    for i in range(N):
        image = np.hstack([left, image, right])
    return image


def arithmeticMeanFilter(image, m=3, n=3):
    # height, width = image.shape
    # oldImage = paddingFilling(image, m, n)
    # oldImages = []
    # for i in range(m):
    #     for j in range(n):
    #         oldImages.append(oldImage[i:i + height, j:j + width])
    # oldImages = np.asarray(oldImages)
    # newImage = np.mean(oldImages, axis=0)
    # return newImage
    return inverseHarmonicMeanFilter(image, m=m, n=n, Q=0)


def geometricMeanFilter(image, m=3, n=3):
    height, width = image.shape
    oldImage = paddingFilling(image, m, n) + 0.0001
    oldImages = []
    for i in range(m):
        for j in range(n):
            oldImages.append(np.log(oldImage[i:i + height, j:j + width]))
    oldImages = np.asarray(oldImages)
    newImage = np.exp(np.mean(oldImages, axis=0)) - 0.0001
    return newImage


def harmonicMeanFilter(image, m=3, n=3):
    # height, width = image.shape
    # oldImage = paddingFilling(image, m, n) + 0.0001
    # oldImages = []
    # for i in range(m):
    #     for j in range(n):
    #         oldImages.append(1 / oldImage[i:i + height, j:j + width])
    # oldImages = np.asarray(oldImages)
    # newImage = (1 / np.mean(oldImages, axis=0)) - 0.0001
    # return newImage
    return inverseHarmonicMeanFilter(image, m=m, n=n, Q=-1)


def inverseHarmonicMeanFilter(image, m=3, n=3, Q=0):
    height, width = image.shape
    oldImage = paddingFilling(image, m, n)+0.0001
    oldImages = []
    for i in range(m):
        for j in range(n):
            oldImages.append(oldImage[i:i + height, j:j + width])
    oldImages = np.asarray(oldImages)
    if Q < 0:
        return np.sum(oldImages ** (Q + 1), axis=0) / np.sum(1 / (oldImages ** np.abs(Q)), axis=0)
    else:
        return np.sum(oldImages ** (Q + 1), axis=0) / np.sum(oldImages ** Q, axis=0)


if __name__ == "__main__":
    oldImage = Image.open("./test.png")
    oldImage.show()
    oldImage = np.asarray(oldImage)
    newImage = harmonicMeanFilter(oldImage, 5, 5)
    newImage = Image.fromarray(newImage.astype("int"))
    newImage.show()