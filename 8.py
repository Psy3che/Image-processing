import cv2
import numpy as np

# 1. 以灰度模式读取图像
img = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)

# 2. 定义结构元素 (Kernel)
# 这是一个 5x5 的全 1 矩阵，尺寸越大，效果越明显
kernel = np.ones((5, 5), np.uint8)

# 3. 执行灰度腐蚀 (Erosion)
# 逻辑：输出像素是内核覆盖范围内的最小值
erosion = cv2.erode(img, kernel, iterations=1)

# 4. 执行灰度膨胀 (Dilation)
# 逻辑：输出像素是内核覆盖范围内的最大值
dilation = cv2.dilate(img, kernel, iterations=1)

# 5. 保存结果
cv2.imwrite('grayscale_erosion.jpg', erosion)
cv2.imwrite('grayscale_dilation.jpg', dilation)