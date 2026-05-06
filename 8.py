import cv2
import numpy as np

img = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)
kernel = np.ones((5, 5), np.uint8)

erosion = cv2.erode(img, kernel, iterations=1)

dilation = cv2.dilate(img, kernel, iterations=1)

cv2.imwrite('grayscale_erosion.jpg', erosion)
cv2.imwrite('grayscale_dilation.jpg', dilation)