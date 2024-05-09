import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import pandas as pd

# 绘制直方图函数
'''def grayHist(img):
    h, w = img.shape[:2]
    pixelSequence = img.reshape([h * w, ])
    numberBins = 256
    histogram, bins, patch = plt.hist(pixelSequence, numberBins,facecolor='black', histtype='bar')
    plt.xlabel("gray label")
    plt.ylabel("number of pixels")
    plt.axis([0, 255, 0, np.max(histogram)])
    plt.show()
image = cv2.imread("1.jpg",0)
grayHist(image)'''

color_dict = {"green": [0, 0, 0],
              "blue": [0, 0, 255],
              "pink": [0, 0, 255]}

# img = cv2.imread('1.jpg')
img = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)


# img_color = gray2rgb(gray , color_dict)

def gray2rgb(gray, color_dict):
    # 1：创建新图像容器
    rgb_image = np.zeros(shape=(*gray.shape, 3))
    # 2： 遍历每个像素点
    for i in range(rgb_image.shape[0]):
        for j in range(rgb_image.shape[1]):
            # 3：对不同的灰度值选择不同的颜色
            if gray[i, j] < 31:
                rgb_image[i, j, :] = color_dict["green"]
            elif gray[i, j] >= 31:
                rgb_image[i, j, :] = color_dict["blue"]

    return rgb_image.astype(np.uint8)


img_color = gray2rgb(img.astype(np.uint8), color_dict)
cv2.imshow("colored", img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

