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


def imageSpliting(image, m=3, n=3):
    height, width = image.shape
    oldImage = paddingFilling(image, m, n)
    oldImages = []
    for i in range(m):
        for j in range(n):
            oldImages.append(oldImage[i:i + height, j:j + width])
    oldImages = np.asarray(oldImages)
    return oldImages


def adaptiveFilter(image, m=3, n=3):
    globalVar = np.var(image)
    oldImages = imageSpliting(image, m=m, n=n)
    localMean = np.mean(oldImages, axis=0)
    localVar = np.var(oldImages, axis=0) + 0.0001
    newImage = image - (globalVar / localVar) * (image - localMean)
    return newImage


def adaptiveMedianFilter(image, m=3, n=3):
    oldImages = imageSpliting(image, m=m, n=n)
    MEDIAN, MIN, MAX = np.median(oldImages, axis=0), np.min(oldImages, axis=0), np.max(oldImages, axis=0)


if __name__ == "__main__":
    oldImage = Image.open("./test.png")
    oldImage.show()
    oldImage = np.asarray(oldImage)
    newImage = adaptiveFilter(oldImage, 3, 3)
    newImage = Image.fromarray(newImage.astype("int"))
    newImage.show()
