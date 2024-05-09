
# 导入必要的库
import os
import cv2
import numpy as np
from PIL import Image
import glob

# 定义函数：获取文件夹及其子文件夹中文件列表
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

# 设置原始图像文件夹和保存处理后图像的文件夹路径
org_img_folder = r'D:\shiyantupian\yuantu'
save_img_folder = r'D:\shiyantupian\kl/'

# 执行文件检索
imglist = getFileList(org_img_folder, [], 'jpg' and 'png')
print('本次执行检索到 ' + str(len(imglist)) + ' 张图像\n')

# 初始化计数器
i = 1

# 循环处理每张图像
for imgpath in imglist:
    # 获取图像文件名
    imgname = os.path.splitext(os.path.basename(imgpath))[0]

    # 读取原始图像和彩色图像，进行边界处理
    img0 = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT)
    imgfsl = img.copy()
    imgsz = img.copy()
    imgmask = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img0 = cv2.copyMakeBorder(img0, 1, 1, 1, 1, cv2.BORDER_CONSTANT)
    img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

    # 展示原始图像
    cv2.imshow("img1", img1)
    print(img1.shape)

   # 分水岭算法提取骨架
    blurredfsl = cv2.pyrMeanShiftFiltering(imgfsl, 5, 10)  # 去除噪点
    cv2.imshow("fsl", blurredfsl)
    # =========确定前景对象==========
    # 灰度图与二值图处理
    grayfsl = cv2.cvtColor(blurredfsl, cv2.COLOR_BGR2GRAY)  # 转灰度图
    ret, binaryfsl = cv2.threshold(grayfsl, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 二值化
    cv2.imshow("Bibarization", binaryfsl)
    binaryfsl = cv2.bitwise_not(binaryfsl)
    cv2.imshow("binary_f", binaryfsl)

    # 形态学操作
    kernelfsl = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mb = cv2.morphologyEx(binaryfsl, cv2.MORPH_OPEN, kernelfsl, iterations=1)  # 形态学开操作 先腐蚀后膨胀
    sure_bgfsl = cv2.dilate(mb, kernelfsl, iterations=3)  # 膨胀
    cv2.imshow("Operation_Morphological", sure_bgfsl)
    # =============================
    # 距离变换
    distfsl = cv2.distanceTransform(sure_bgfsl, cv2.DIST_L2, 3)  # 提取前景
    cv2.imshow("distfsl", distfsl)

    # 归一化
    dist_output = cv2.normalize(distfsl, 0, 1, cv2.NORM_MINMAX)
    cv2.imshow("Distance_Transformation", dist_output * 255)

    # 阈值处理
    ret, surfacefsl = cv2.threshold(distfsl, dist_output.max() * 0.015, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("Find_Seed", surfacefsl)
    surface_fg = np.uint8(surfacefsl)

    # 颜色映射函数
    color_dict = {"black": [0, 0, 0], "bai": [255, 255, 255], "pink": [255, 255, 255]}

    def gray2rgb(gray, color_dict):
        # 创建新图像容器
        rgb_image = np.zeros(shape=(*gray.shape, 3))
        # 遍历每个像素点
        for i in range(rgb_image.shape[0]):
            for j in range(rgb_image.shape[1]):
                # 对不同的灰度值选择不同的颜色
                if gray[i, j] < 5:   # 调整灰度值阈值
                    rgb_image[i, j, :] = color_dict["black"]
                elif gray[i, j] >= 1:
                    rgb_image[i, j, :] = color_dict["bai"]
        return rgb_image.astype(np.uint8)

    # 将灰度图映射为伪彩色图
    img_color = gray2rgb(img0.astype(np.uint8), color_dict)
    cv2.imshow("colored", img_color)
    print(img_color.shape)

    # 二值化处理
    imgkl = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    cv2.imshow("imgkl", imgkl)
    print(imgkl.shape)

    imgkl_n = cv2.bitwise_not(imgkl)

    # 洞填充操作
    def fillHoletcgj(im_in):
        im_floodfill = im_in.copy()

        # Mask used to flood filling.
        h, w = im_in.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (255, 255), 255);

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # Combine the two images to get the foreground.
        im_out = im_in | im_floodfill_inv

        return im_out

    # 执行洞填充


    imgkl1 = fillHoletcgj(imgkl_n)
    cv2.imshow("imgkl1", imgkl1)

    imgkl2 = cv2.bitwise_not(imgkl1)
    cv2.imshow("imgkl2", imgkl2)

    fbgj1 = cv2.add(imgkl2, surface_fg)
    cv2.imshow("fbgj1", fbgj1)

    # 自定义模糊操作
    def custom_blur_img(img):
        kernel = np.array([[3, 1, 0], [5, 5, 0], [3, 1, 0]])
        custom_blur = cv2.filter2D(img, -1, kernel=kernel)
        return custom_blur

    gray = custom_blur_img(gray)
    cv2.imshow("gray", gray)

    # 大津法二值化
    ret, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("dst", dst)
    gray_n = cv2.bitwise_not(dst)

    # 填充孔洞
    def fillHolegray(im_in):
        im_floodfill = im_in.copy()

        # Mask used to flood filling.
        h, w = im_in.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (255, 255), 255);

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # Combine the two images to get the foreground.
        im_out = im_in | im_floodfill_inv

        return im_out

    # 执行灰度图孔洞填充
    gray1 = fillHolegray(gray_n)
    cv2.imshow("gray1", gray1)

    # 骨架图像处理
    gray2 = cv2.bitwise_not(gray1)
    cv2.imshow("gray2", gray2)
    fbgj = cv2.add(fbgj1, gray2)
    cv2.imshow("fbgj", fbgj)

    # 图像相减
    imgkl3 = cv2.subtract(imgkl, fbgj)
    cv2.imshow("imgkl3", imgkl3)

    # 图像洞填充
    def fillHole(image):
        h, w = image.shape[:2]
        mask = np.zeros(((h + 2, w + 2)), np.uint8)
        cv2.floodFill(image, mask, (0, 0), 255)
        return image

    # 洞填充
    constant = fillHole(imgkl)
    cv2.imshow("constant2", constant)
    constant = cv2.bitwise_not(constant)
    cv2.imshow("constant", constant)

    # 区域填充
    def fill_small_region(image, max_area, num):
        fill_contour = []

        contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area <= max_area:
                fill_contour.append(contour)
        cv2.fillPoly(image, fill_contour, num)

        return image

    fill_image = fill_small_region(constant, 1500, 0)  # 参数可调整
    fill_image = cv2.medianBlur(fill_image, 7)  # 参数可调整
    cv2.imshow("fill_image", fill_image)

    # 形态学操作
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    fill_image = cv2.erode(fill_image, kernel1)
    cv2.imshow("fill_image2", fill_image)

    # 洞填充
    def fillHoletc(im_in):
        im_floodfill = im_in.copy()

        # Mask used to flood filling.
        h, w = im_in.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255);

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # Combine the two images to get the foreground.
        im_out = im_in | im_floodfill_inv

        return im_out

    # 执行洞填充
    fill_image = fillHoletc(fill_image)
    szsz = fill_image.copy()
    cv2.imshow("fill_image3", fill_image)







    # 颗粒图像处理
    imgjhkl = cv2.multiply(fill_image, imgkl3)
    cv2.imshow("imgjhkl", imgjhkl)
    imgjh = imgjhkl.copy()

    # 区域填充
    xueguan = fill_small_region(imgjhkl, 5, 0)  # 参数可调整
    cv2.imshow("xueguan", xueguan)

    # 颗粒图像减去血管图像
    chenfei = cv2.subtract(imgjh, xueguan)
    cv2.imshow("cf", chenfei)

    # 灰度图处理
    graysz = cv2.cvtColor(imgsz, cv2.COLOR_RGB2GRAY)
    cv2.imshow("graysz", graysz)
    gray_imgsz = graysz.copy()

    # 大津法二值化
    def otsu(gray_img):
        h = gray_img.shape[0]
        w = gray_img.shape[1]
        N = h * w
        threshold_t = 0
        max_g = 0

        # 遍历每一个灰度级
        for t         in range(256):
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

        # 使用阈值进行二值化
        gray_img[gray_img < threshold_t] = 0
        gray_img[gray_img >= threshold_t] = 255
        return gray_img

    # 应用大津法进行二值化
    otsu_imgsz = otsu(gray_imgsz)
    cv2.imshow('otsu_imgsz ', otsu_imgsz)

    # 血管图像减去分水岭前景，得到血管图像
    imgotsu = cv2.subtract(otsu_imgsz, fbgj)
    cv2.imshow("imgotsu", imgotsu)

    # 灰度图与分水岭前景图像相减
    gray3 = cv2.subtract(dst, fbgj)
    cv2.imshow("gray3", gray3)

    # 血管图像与分水岭前景相加
    imgdj = cv2.add(imgotsu, gray3)
    imgdj = cv2.multiply(szsz, imgdj)
    cv2.imshow("imgdj", imgdj)

    # 区域填充
    xg = fill_small_region(imgdj, 13, 1)  # 参数可调整
    cv2.imshow("xg", xg)

    # 取反操作
    cff = cv2.bitwise_not(xg)
    cv2.imshow("cff", cff)

    # 血管图像与取反结果相乘
    cfzz = cv2.multiply(imgdj, cff)
    cv2.imshow("cfzz", cfzz)

    # 最终结果，取反结果与血管图像相加
    zhjhkl = cv2.add(cfzz, chenfei)
    cv2.imshow("zhjhkl", zhjhkl)

    # 颜色处理函数
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

        print(imgkl.shape)
        return imgkl

    # 应用颜色处理函数
    cftqcf = yanse(zhjhkl)
    cfjh = cv2.add(cftqcf, imgmask)
    cv2.imshow("mask", cfjh)
    cv2.imshow("hongsekeli",cftqcf)

    # 图像裁剪
    def caijian(img):
        img1 = img[1:513, 1:513]
        return img1

    cftq = caijian(cfjh)

    # 保存图像
    i = i + 1
    Img_Name = save_img_folder + str(i - 1) + ".jpg" or ".png"
    cv2.imwrite(Img_Name, cftq)

    # 等待用户按键，关闭图像窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()
