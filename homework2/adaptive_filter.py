from PIL import Image
import numpy as np
import time


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
    localVar = np.var(oldImages, axis=0) + 0.000001
    newImage = image - (globalVar / localVar) * (image - localMean)
    return newImage


def adaptiveMedianFilter(image, Smax=7):
    height, width = image.shape
    newImage = image.copy()
    for i in range(height):
        for j in range(width):
            size = 3
            z = image[i][j]
            while (size <= Smax):
                s = size // 2
                tmp = image[max(0, i - s):i + s + 1, max(0, j - s):j + s + 1].reshape(-1)
                tmp.sort()
                zmin, zmax, zmed = tmp[0], tmp[-1], tmp[tmp.shape[0] // 2]
                if zmin < zmed < zmax:
                    if z == zmin or z == zmax:
                        newImage[i][j] = zmed
                    break
                else:
                    size += 2
    return newImage


if __name__ == "__main__":
    oldImage = Image.open("./lena1.png")
    oldImage.show()
    oldImage = np.asarray(oldImage)
    begin = time.perf_counter()
    newImage = adaptiveFilter(oldImage)
    print(time.perf_counter() - begin)
    newImage = Image.fromarray(newImage.astype("int"))
    newImage.show()
