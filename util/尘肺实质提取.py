import cv2
import numpy as np

img=cv2.imread("image.jpg")
cv2.imshow("img",img)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2.imshow("gray",gray)
ret, dst = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY| cv2.THRESH_OTSU)
cv2.imshow("dst",dst)


constant = cv2.copyMakeBorder(dst, 1, 1, 1, 1, cv2.BORDER_CONSTANT)

def fillHole(image):
    h, w = image.shape[:2]
    mask = np.zeros(((h+2, w+2)), np.uint8)
    cv2.floodFill(image, mask, (0, 0), 255)

    return image
constant=fillHole(constant)
cv2.imshow("constant2",constant)
constant = cv2.bitwise_not(constant)
cv2.imshow("constant",constant)
def fill_small_region(image, max_area, num):
    fill_contour = []

    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= max_area:
            fill_contour.append(contour)
    cv2.fillPoly(image, fill_contour, num)

    return image
fill_image = fill_small_region(constant, 1000, 0)

cv2.imshow("fill_image",fill_image)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
fill_image = cv2.morphologyEx(fill_image, cv2.MORPH_CLOSE, kernel=kernel)
cv2.imshow("fill_image2",fill_image)
print(fill_image.shape)









cv2.waitKey(0)
cv2.destroyAllWindows()
