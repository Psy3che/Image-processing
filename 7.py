# 导入图像处理库
import cv2
import numpy as np

# 以灰度模式读取图片
img = cv2.imread('input.jpg', 0)

# 定义 3×3 击中击不中核（核心！）
kernel = np.array([[0, 1, 0],
                   [1, -1, 1],
                   [0, 1, 0]], dtype=np.int8)

# 执行击中击不中变换
hm = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel)

# 保存结果
cv2.imwrite('hit_miss.jpg', hm)