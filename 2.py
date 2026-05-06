import cv2
img = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)
k_size = 3  
dst = cv2.blur(img, (k_size, k_size), borderType=cv2.BORDER_CONSTANT)
cv2.imwrite('averaging.jpg', dst)