import cv2
img = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)
equ = cv2.equalizeHist(img)
cv2.imwrite('equalized.jpg', equ)
