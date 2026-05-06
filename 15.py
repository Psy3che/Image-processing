import cv2

# 读取灰度图
img = cv2.imread('input.jpg', 0)

canny = cv2.Canny(img, 100, 200)

sobel = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)  # xy双向边缘
sobel = cv2.convertScaleAbs(sobel)

laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

diff1 = cv2.absdiff(canny, sobel)
cv2.imwrite('compare_Canny_Sobel.jpg', diff1)

diff2 = cv2.absdiff(canny, laplacian)
cv2.imwrite('compare_Canny_Laplacian.jpg', diff2)

diff3 = cv2.absdiff(sobel, laplacian)
cv2.imwrite('compare_Sobel_Laplacian.jpg', diff3)