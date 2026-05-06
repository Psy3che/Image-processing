import cv2

img = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)
k_size = 3 # 必须是奇数
dst = cv2.medianBlur(img, k_size)
cv2.imwrite('median.jpg', dst)