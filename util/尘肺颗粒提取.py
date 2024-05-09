import cv2 as cv

import tkinter as tk
from tkinter import filedialog




root = tk.Tk()
root.withdraw()
# FolderPath=filedialog.askdirectory() #选择文件夹用的
FilePath = filedialog.askopenfilename()  # 选择文件用的
# print('FolderPath:',FolderPath)#验证文件夹
print('FilePath:', FilePath)  # 验证文件
img = cv.imread(FilePath)

gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
img1 = gray.copy()
cv.imshow("01",img1)
# 全局二值化
ret, binary = cv.threshold(img1, 0, 255,cv.THRESH_BINARY | cv.THRESH_TRIANGLE)  # THRESH_OTSU自动阈值  方法不一样，阈值不一样（很有用，自己查！！
imgbai=binary.copy()
img2=binary.copy()
cv.imshow("02",img2)

# 图像反色
img3 = cv.bitwise_not(img1)
cv.imshow("03",img3)

# 图像相乘
img4 = cv.multiply(img3, img2)
cv.imshow("04",img4)
# 中值模糊  对椒盐噪声有很好的去燥效果
img5 = cv.medianBlur(img4, 5)
cv.imshow("05",img5)
# 开闭运算去除不必要部分
kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
#mb = cv.morphologyEx(img5, cv.MORPH_OPEN, kernel, iterations=3)
img6 = cv.morphologyEx(img5, cv.MORPH_CLOSE, kernel, iterations=5)
cv.imshow("06",img6)
# 取反色
img7 = cv.bitwise_not(img6)
cv.imshow("07",img7)
# 图像相加
img8 = cv.add(img1, img7)
cv.imshow("08",img8)
# 取反色
img9 = cv.bitwise_not(img8)
cv.imshow("09",img9)
# 相除
img10 = cv.subtract(img6, img9)
cv.imshow("10",img10)

img11= cv.subtract(img1, img10)
cv.imshow("11",img11)
constant = cv.copyMakeBorder(img11, 1, 1, 1, 1, cv.BORDER_CONSTANT)
cv.imshow("111",constant)
print(constant.shape)
# rgb格式转化
img12= cv.cvtColor(constant, cv.COLOR_RGB2RGBA)
print(img12.shape)



cv.waitKey(0)
cv.destroyAllWindows()

