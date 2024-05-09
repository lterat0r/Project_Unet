import os
import cv2
import numpy as np
from PIL import Image
import glob

def getFileList(dir, Filelist, ext=None):
    """
    获取文件夹及其子文件夹中文件列表
    输入 dir：文件夹根目录
    输入 ext: 扩展名
    返回： 文件路径列表
    """
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)

    return Filelist

org_img_folder = r'D:\shiyantupian\yuantu'
save_img_folder = r'D:\shiyantupian\kl/'

# 检索文件
imglist = getFileList(org_img_folder, [], 'jpg' and 'png')
print('本次执行检索到 ' + str(len(imglist)) + ' 张图像\n')
i = 1
for imgpath in imglist:
    imgname = os.path.splitext(os.path.basename(imgpath))[0]
    img0 = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT)
    imgfsl=img.copy()
    imgsz = img.copy()
    imgmask=img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img0 = cv2.copyMakeBorder(img0, 1, 1, 1, 1, cv2.BORDER_CONSTANT)
    img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    cv2.imshow("img1", img1)
    print(img1.shape)

   #分水岭算法提取骨架
    blurredfsl = cv2.pyrMeanShiftFiltering(imgfsl, 5, 10)  # 去除噪点
    cv2.imshow("fsl", blurredfsl)
    # =========确定前景对象==========
    # gray\binary image
    grayfsl = cv2.cvtColor(blurredfsl, cv2.COLOR_BGR2GRAY)  # 转灰度图
    ret, binaryfsl = cv2.threshold(grayfsl, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 二值化
    cv2.imshow("Bibarization", binaryfsl)
    binaryfsl = cv2.bitwise_not(binaryfsl)
    cv2.imshow("binary_f", binaryfsl)

    # morphology operation
    kernelfsl = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mb = cv2.morphologyEx(binaryfsl, cv2.MORPH_OPEN, kernelfsl, iterations=1)  # 形态学开操作 先腐蚀后膨胀
    sure_bgfsl = cv2.dilate(mb, kernelfsl, iterations=3)  # 膨胀
    cv2.imshow("Operation_Morphological", sure_bgfsl)
    # =============================
    # distance transform
    distfsl = cv2.distanceTransform(sure_bgfsl, cv2.DIST_L2, 3)  # 提取前景
    cv2.imshow("distfsl", distfsl)
    # dist = cv22.distanceTransform(src=gaussian_hsv, distanceType=cv22.DIST_L2, maskSize=5) 距离变换函数
    # dist – 具有计算距离的输出图像。它是一个与 src 大小相同的 32 位浮点单通道图像。
    # src – 8 位、单通道（二进制）源图像。
    # distanceType – 距离类型。它可以是 cv2_DIST_L1、cv2_DIST_L2 或 cv2_DIST_C。
    # maskSize – 距离变换掩码的大小。它可以是 3、5 或 cv2_DIST_MASK_PRECISE（后一个选项仅由第一个函数支持）。
    #     在 cv2_DIST_L1 或 cv2_DIST_C 距离类型的情况下，参数被强制为 3，因为 3\times 3 掩码给出与 5\times 5 或任何更大孔径相同的结果。
    dist_output = cv2.normalize(distfsl, 0, 1, cv2.NORM_MINMAX)  # 归一化在0~1之间
    cv2.imshow("Distance_Transformation", dist_output * 255)

    ret, surfacefsl = cv2.threshold(distfsl, dist_output.max() * 0.015, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("Find_Seed", surfacefsl)
    surface_fg = np.uint8(surfacefsl)

    color_dict = {"black": [0, 0, 0],
                  "bai": [255, 255, 255],
                  "pink": [255, 255, 255]}

    def gray2rgb(gray, color_dict):
        # 1：创建新图像容器
        rgb_image = np.zeros(shape=(*gray.shape, 3))
        # 2： 遍历每个像素点
        for i in range(rgb_image.shape[0]):
            for j in range(rgb_image.shape[1]):
                # 3：对不同的灰度值选择不同的颜色
                if gray[i, j] < 5:                                                            #5 和下面的 1可以进行修改，根据需求进行修改 属于调整灰度值
                    rgb_image[i, j, :] = color_dict["black"]
                elif gray[i, j] >= 1:
                    rgb_image[i, j, :] = color_dict["bai"]

        return rgb_image.astype(np.uint8)


    img_color = gray2rgb(img0.astype(np.uint8), color_dict)
    cv2.imshow("colored", img_color)
    print(img_color.shape)
    #下面进行二值化的处理，将3通道图像转为2通道，便于后续操作
    imgkl = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    cv2.imshow("imgkl",imgkl)
    print(imgkl.shape)

    imgkl_n = cv2.bitwise_not(imgkl)

    def fillHoletcgj(im_in):
        im_floodfill = im_in.copy()

        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = im_in.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (255, 255), 255);

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # Combine the two images to get the foreground.
        im_out = im_in | im_floodfill_inv

        return im_out
        #
    imgkl1 = fillHoletcgj(imgkl_n)
    cv2.imshow("imgkl1", imgkl1)

    imgkl2 = cv2.bitwise_not(imgkl1)
    cv2.imshow("imgkl2", imgkl2)

    fbgj1=cv2.add(imgkl2,surface_fg)
    cv2.imshow("fbgj1",fbgj1)

    def custom_blur_img(img):
        # kernel = np.ones([9, 9], np.float32)/25
        kernel = np.array([[3, 1, 0], [5, 5, 0], [3, 1, 0]], np.float32)   #此处为模糊操作，数值可以进行修改，但是每一个数都会和其他的相关
        custom_blur = cv2.filter2D(img, -1, kernel=kernel)
        return custom_blur

    gray = custom_blur_img(gray)
    cv2.imshow("gray", gray)

    ret, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("dst", dst)
    gray_n = cv2.bitwise_not(dst)

    def fillHolegray(im_in):
        im_floodfill = im_in.copy()

        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = im_in.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (255, 255), 255);

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # Combine the two images to get the foreground.
        im_out = im_in | im_floodfill_inv

        return im_out
        #
    gray1 = fillHolegray(gray_n)
    cv2.imshow("gray1", gray1)

    #骨架
    gray2 = cv2.bitwise_not(gray1)
    cv2.imshow("gray2", gray2)
    fbgj=cv2.add(fbgj1,gray2)
    cv2.imshow("fbgj",fbgj)

    imgkl3 = cv2.subtract(imgkl, fbgj)
    cv2.imshow("imgkl3", imgkl3)

    def fillHole(image):
        h, w = image.shape[:2]
        mask = np.zeros(((h + 2, w + 2)), np.uint8)
        cv2.floodFill(image, mask, (0, 0), 255)
        return image

    constant = fillHole(imgkl)
    cv2.imshow("constant2", constant)
    constant = cv2.bitwise_not(constant)
    cv2.imshow("constant", constant)

    def fill_small_region(image, max_area, num):
        fill_contour = []

        contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area <= max_area:
                fill_contour.append(contour)
        cv2.fillPoly(image, fill_contour, num)

        return image

    fill_image = fill_small_region(constant, 1500, 0)  # 1500可改 1000-2500均可以
    fill_image = cv2.medianBlur(fill_image, 7)  # 7可改 9 11 13 15均可以
    cv2.imshow("fill_image", fill_image)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 务必3
    # 腐蚀图像
    fill_image = cv2.erode(fill_image, kernel1)
    # 显示腐蚀后的图像
    cv2.imshow("fill_image2", fill_image)

    def fillHoletc(im_in):
        im_floodfill = im_in.copy()

        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = im_in.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255);

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # Combine the two images to get the foreground.
        im_out = im_in | im_floodfill_inv

        return im_out


    fill_image = fillHoletc(fill_image)
    szsz = fill_image.copy()
    cv2.imshow("fill_image3", fill_image)

    imgjhkl = cv2.multiply(fill_image, imgkl3)
    cv2.imshow("imgjhkl", imgjhkl)
    imgjh = imgjhkl.copy()

    xueguan = fill_small_region(imgjhkl, 5, 0)  # ()内的数值可以自由替换哈
    cv2.imshow("xueguan", xueguan)

    chenfei = cv2.subtract(imgjh, xueguan)
    cv2.imshow("cf", chenfei)

    graysz = cv2.cvtColor(imgsz, cv2.COLOR_RGB2GRAY)
    cv2.imshow("graysz", graysz)
    # 读入灰度图
    gray_imgsz = graysz.copy()


    # 大津二值化算法
    def otsu(gray_img):
        h = gray_img.shape[0]
        w = gray_img.shape[1]
        N = h * w
        threshold_t = 0
        max_g = 0

        # 遍历每一个灰度级
        for t in range(256):
            # 使用numpy直接对数组进行运算
            n0 = gray_img[np.where(gray_img < t)]
            n1 = gray_img[np.where(gray_img >= t)]
            w0 = len(n0) / N
            w1 = len(n1) / N
            u0 = np.mean(n0) if len(n0) > 0 else 0.
            u1 = np.mean(n1) if len(n1) > 0 else 0.

            g = w0 * w1 * (u0 - u1) ** 2
            if g > max_g:
                max_g = g
                threshold_t = t
        print('类间方差最大阈值：', threshold_t)
        gray_img[gray_img < threshold_t] = 0
        gray_img[gray_img >= threshold_t] = 255
        return gray_img


    otsu_imgsz = otsu(gray_imgsz)
    cv2.imshow('otsu_imgsz ', otsu_imgsz)

    imgotsu=cv2.subtract(otsu_imgsz,fbgj)
    cv2.imshow("imgotsu",imgotsu)


    gray3 = cv2.subtract(dst, fbgj)
    cv2.imshow("gray3", gray3)


    imgdj=cv2.add(imgotsu,gray3)

    imgdj=cv2.multiply(szsz,imgdj)
    cv2.imshow("imgdj", imgdj)

    xg = fill_small_region(imgdj, 13, 1)                                                     #(13,1)可以进行修改
    cv2.imshow("xg",xg)

    cff = cv2.bitwise_not(xg)
    cv2.imshow("cff", cff)

    cfzz = cv2.multiply(imgdj, cff)
    cv2.imshow("cfzz", cfzz)

    zhjhkl=cv2.add(cfzz,chenfei)
    cv2.imshow("zhjhkl",zhjhkl)

    def yanse(image):
        imgkl = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(imgkl.shape)
        rows, cols, channels = imgkl.shape
        # 将在两个阈值内的像素值设置为白色（255），
        # 而不在阈值区间内的像素值设置为黑色（0）
        lower_red = np.array([30, 30, 30])
        upper_red = np.array([255, 255, 255])
        mask = cv2.inRange(imgkl, lower_red, upper_red)

        for i in range(rows):
            for j in range(cols):
                if mask[i, j] == 255:  # 像素点255表示白色,180为灰度
                    imgkl[i, j] = (0, 0, 255)  # 此处替换颜色，为BGR通道，不是RGB通道

        ###cv2.imshow("cfklr", imgkl)
        print(imgkl.shape)
        return  imgkl


    cftqcf=yanse(zhjhkl)
    cfjh = cv2.add(cftqcf, imgmask)
    cv2.imshow("mask", cfjh)

    def caijian(img):
        img1 = img[1:513, 1:513]
        return img1


    cftq=caijian(cfjh)

    # 保存
    i = i + 1
    Img_Name = save_img_folder + str(i - 1) + ".jpg" or ".png"
    cv2.imwrite(Img_Name,cftq)

    cv2.waitKey(0)
    cv2.destroyAllWindows()