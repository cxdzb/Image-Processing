from PIL import Image
import numpy as np
import math, random

LoG = np.array([
    [0, 0, -1, 0, 0],
    [0, -1, -2, -1, 0],
    [-1, -2, 16, -2, -1],
    [0, -1, -2, -1, 0],
    [0, 0, -1, 0, 0],
])
num, inf = 10, np.inf
linker, slack = np.zeros((num), dtype=np.int32), np.zeros((num))
lx, ly = np.zeros((num)), np.zeros((num))
visx, visy = np.zeros((num), dtype=np.int8), np.zeros((num), dtype=np.int8)


def KM():
    def DFS(x):
        visx[x] = 1
        for y in range(num):
            if visy[y]: continue
            tmp = lx[x] + ly[y] - costs[x][y]
            if np.abs(tmp) < 1e-6:
                visy[y] = 1
                if linker[y] == -1 or DFS(linker[y]):
                    linker[y] = x
                    return 1
            elif slack[y] > tmp:
                slack[y] = tmp
        return 0

    linker[:], ly[:] = -1, 0
    lx = np.max(costs, axis=1)
    for x in range(num):
        slack[:] = inf
        while True:
            visx[:], visy[:] = 0, 0
            if DFS(x): break
            d = np.min(slack[visy == 0])
            lx[visx != 0] -= d
            ly[visy != 0] += d
            slack[visy == 0] -= d
    index = linker != -1
    res = np.sum(costs[linker[index], index])
    return res


def getEdge(image):
    image = image.astype("int")
    up, down = image[0], image[-1]
    image = np.vstack([up, up, image, down, down])
    left, right = image[:, [0]], image[:, [-1]]
    image = np.hstack([left, left, image, right, right])
    height, width = image.shape

    logs = []
    for i in range(5):
        for j in range(5):
            logs.append(image[i:i + height - 4, j:j + width - 4] * LoG[i][j])
    logs = np.asarray(logs)
    newImage = np.sum(logs, axis=0)
    newImage = np.abs(newImage)

    allPixels = np.sort(newImage.reshape(-1))
    threshold = (allPixels[allPixels.shape[0] // 2] * 2 + allPixels[0] + allPixels[-1]) / 4
    newImage[newImage < threshold], newImage[newImage >= threshold] = 0, 255
    newImage = newImage.astype(np.uint8)

    edge = np.argwhere(newImage == 255)

    if num >= edge.shape[0]:
        return edge
    np.random.seed(2019)
    index = np.random.choice(np.arange(0, edge.shape[0], 1), size=num, replace=False)
    edge = edge[index]
    return edge


def getShapeContext(points, N=5, M=12, interval=5):
    minDegree = math.pi * 2 / M
    shapeContexts = []
    for i in range(num):
        shapeContext = np.zeros((N, M), dtype="int")
        for j in range(num):
            if i == j: continue
            x, y = points[j][0] - points[i][0], points[j][1] - points[i][1]
            dist = math.hypot(x, y)
            theta = math.atan2(x, y)
            x, y = int(math.log(dist)), int((theta - 1e-6 + math.pi) / minDegree)
            if x <= N - 1:
                shapeContext[x][y] += 1
        shapeContexts.append(shapeContext)
    return np.asarray(shapeContexts, dtype="int")


def getCost(context1, context2):
    costs = np.zeros((num, num), dtype="float")
    for i in range(num):
        for j in range(num):
            ctx1, ctx2 = context1[i], context2[j]
            sub, add = ctx1 - ctx2, ctx1 + ctx2 + 1e-6
            costs[i][j] = np.sum((sub ** 2) / add) / 2
    return costs


if __name__ == "__main__":
    image1 = np.asarray(Image.open("./resource/image1.png").convert("L"))
    image2 = np.asarray(Image.open("./resource/image2.png").convert("L"))
    edge1 = getEdge(image1)
    edge2 = getEdge(image2)
    shapeContext1 = getShapeContext(edge1)
    shapeContext2 = getShapeContext(edge2)
    costs = -getCost(shapeContext1, shapeContext2)
    similarity = -KM()
    print(similarity)
