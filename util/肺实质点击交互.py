import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

img = cv.imread('image.jpg')
img1=img.copy()

# 计算种子点和其领域的像素值之差
def getGrayDiff(gray, current_seed, tmp_seed):
    return abs(int(gray[current_seed[0], current_seed[1]]) - int(gray[tmp_seed[0], tmp_seed[1]]))


# 区域生长算法
def regional_growth(gray, seeds):
    # 八领域
    connects = [(-1, -1), (0, -1), (1, -1), (1, 0), \
                (1, 1), (0, 1), (-1, 1), (-1, 0)]
    seedMark = np.zeros((gray.shape))
    height, width = gray.shape
    threshold = 6
    seedque = deque()
    label = 255
    seedque.extend(seeds)

    while seedque:
        # 队列具有先进先出的性质。所以要左删
        current_seed = seedque.popleft()
        seedMark[current_seed[0], current_seed[1]] = label
        for i in range(8):
            tmpX = current_seed[0] + connects[i][0]
            tmpY = current_seed[1] + connects[i][1]
            # 处理边界情况
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= width:
                continue

            grayDiff = getGrayDiff(gray, current_seed, (tmpX, tmpY))
            if grayDiff < threshold and seedMark[tmpX, tmpY] != label:
                seedque.append((tmpX, tmpY))
                seedMark[tmpX, tmpY] = label
    return seedMark


# 交互函数
def Event_Mouse(event, x, y, flags, param):
    # 左击鼠标
    if event == cv.EVENT_LBUTTONDOWN:
        # 添加种子
        seeds.append((y, x))
        # 画实心点
        cv.circle(img, center=(x, y), radius=2,color=(0, 0, 255), thickness=-1)


def Region_Grow(img):
    cv.namedWindow('img')
    cv.setMouseCallback('img', Event_Mouse)
    cv.imshow('img', img)

    while True:
        cv.imshow('img', img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            #看这里，输入q 进行运行
            break
    cv.destroyAllWindows()


    seedMark = np.uint8(regional_growth(cv.cvtColor(img1, cv.COLOR_BGR2GRAY), seeds))

    cv.imshow('seedMark', seedMark)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    fill_image = cv.morphologyEx(seedMark, cv.MORPH_CLOSE, kernel=kernel)
    cv.imshow('fill', fill_image)
    cv.waitKey(0)



if __name__ == '__main__':

    seeds = []
    Region_Grow(img)

