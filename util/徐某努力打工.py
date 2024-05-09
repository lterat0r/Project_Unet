import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import filedialog

import cv2
from skimage import data
from skimage.filters import threshold_otsu  #这一部分谁写的？怎么跑不动?

from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

root = tk.Tk()
root.withdraw()
# FolderPath=filedialog.askdirectory() #选择文件夹用的
FilePath = filedialog.askopenfilename()  # 选择文件用的
# print('FolderPath:',FolderPath)#验证文件夹
print('FilePath:', FilePath)  # 验证文件
img = cv.imread(FilePath)

cv.imshow("img", img)
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
cv.imshow("gray", gray)
ret, dst = cv.threshold(gray, 15, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
cv.imshow("dst", dst)

constant = cv.copyMakeBorder(dst, 1, 1, 1, 1, cv.BORDER_CONSTANT)


def fillHole(image):
    h, w = image.shape[:2]
    mask = np.zeros(((h + 2, w + 2)), np.uint8)
    cv.floodFill(image, mask, (0, 0), 255)

    return image


constant = fillHole(constant)
cv.imshow("constant2", constant)
constant = cv.bitwise_not(constant)
cv.imshow("constant", constant)


def fill_small_region(image, max_area, num):
    fill_contour = []

    contours, _ = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv.contourArea(contour)
        if area <= max_area:
            fill_contour.append(contour)
    cv.fillPoly(image, fill_contour, num)

    return image


fill_image = fill_small_region(constant, 1000, 0)

cv.imshow("fill_image", fill_image)

kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
fill_image = cv.morphologyEx(fill_image, cv.MORPH_CLOSE, kernel=kernel)
cv.imshow("fill_image2", fill_image)
print(fill_image.shape)

gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
img1 = gray.copy()
cv.imshow("01", img1)
# 全局二值化
ret, binary = cv.threshold(
    img1, 0, 255, cv.THRESH_BINARY
    | cv.THRESH_TRIANGLE)  # THRESH_OTSU自动阈值  方法不一样，阈值不一样（很有用，自己查！！
imgbai = binary.copy()
img2 = binary.copy()
cv.imshow("02", img2)

# 图像反色
img3 = cv.bitwise_not(img1)
cv.imshow("03", img3)

# 图像相乘
img4 = cv.multiply(img3, img2)
cv.imshow("04", img4)
# 中值模糊  对椒盐噪声有很好的去燥效果
img5 = cv.medianBlur(img4, 5)
cv.imshow("05", img5)
# 开闭运算去除不必要部分
kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
#mb = cv.morphologyEx(img5, cv.MORPH_OPEN, kernel, iterations=3)
img6 = cv.morphologyEx(img5, cv.MORPH_CLOSE, kernel, iterations=5)
cv.imshow("06", img6)
# 取反色
img7 = cv.bitwise_not(img6)
cv.imshow("07", img7)
# 图像相加
img8 = cv.add(img1, img7)
cv.imshow("08", img8)
# 取反色
img9 = cv.bitwise_not(img8)
cv.imshow("09", img9)
# 相除
img10 = cv.subtract(img6, img9)
cv.imshow("10", img10)

img11 = cv.subtract(img1, img10)
cv.imshow("11", img11)
constant = cv.copyMakeBorder(img11, 1, 1, 1, 1, cv.BORDER_CONSTANT)
cv.imshow("111", constant)
print(constant.shape)
# rgb格式转化
img12 = cv.cvtColor(constant, cv.COLOR_RGB2RGBA)

cv2.imwrite("1.jpg", img12)  #将图片保存为1.jpg
print(img12.shape)

img13 = cv.cvtColor(img12, cv.COLOR_BGR2GRAY)
ret, binary = cv.threshold(img13, 255, 63, cv.THRESH_BINARY | cv.THRESH_OTSU)
print("threshold value %s" % ret)
cv.namedWindow('global_threshold_binary222', 0)  #可调节窗口
cv.imshow("global_threshold_binary222", binary)

img14 = cv.cvtColor(img12, cv.COLOR_BGR2GRAY)
ret, binary = cv.threshold(img14, 255, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
print("threshold value %s" % ret)
cv.namedWindow('global_threshold_binary112', 0)  #可调节窗口
cv.imshow("global_threshold_binary112", binary)

color_dict = {"green": [0, 0, 0], "blue": [0, 0, 255], "pink": [0, 0, 255]}

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

cv.waitKey(0)
cv.destroyAllWindows()
