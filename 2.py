import cv2

img = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)
k_size = 3 # 可选 3, 5, 7

# borderType: 
# cv2.BORDER_ISOLATED (仅完全重叠), cv2.BORDER_CONSTANT (补零), cv2.BORDER_REFLECT (忽略/反射)
# dst = cv2.blur(img, (k_size, k_size), borderType=cv2.BORDER_CONSTANT)
dst = cv2.blur(img, (k_size, k_size))
cv2.imwrite('averaging.jpg', dst)