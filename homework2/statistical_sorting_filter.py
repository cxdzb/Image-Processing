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


def medianFilter(image, m=3, n=3):
    oldImages = imageSpliting(image, m=m, n=n)
    newImage = np.median(oldImages, axis=0)
    return newImage


def maximumFilter(image, m=3, n=3):
    oldImages = imageSpliting(image, m=m, n=n)
    newImage = np.max(oldImages, axis=0)
    return newImage


def minimumFilter(image, m=3, n=3):
    oldImages = imageSpliting(image, m=m, n=n)
    newImage = np.min(oldImages, axis=0)
    return newImage


def medianRangeFilter(image, m=3, n=3):
    oldImages = imageSpliting(image, m=m, n=n)
    newImage = (np.max(oldImages, axis=0) + np.min(oldImages, axis=0)) / 2
    return newImage


def modifiedAlphaMeanFilter(image, m=3, n=3, d=2):
    d = d // 2
    oldImages = imageSpliting(image, m=m, n=n)
    oldImages = np.sort(oldImages, axis=0)
    newImage = np.mean(oldImages[d:m * n - d], axis=0)
    return newImage


if __name__ == "__main__":
    oldImage = Image.open("./test.png")
    oldImage.show()
    oldImage = np.asarray(oldImage)
    newImage = modifiedAlphaMeanFilter(oldImage, 5, 5)
    newImage = Image.fromarray(newImage.astype("int"))
    newImage.show()
