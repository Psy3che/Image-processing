import cv2

img = cv2.imread('input.jpg', 0)
ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
cv2.imwrite('threshold.jpg', thresh)