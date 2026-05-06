import cv2
import numpy as np

# 1. 快速读取所有图片并转为 3D 矩阵 (List Comprehension)
imgs = [cv2.imread(f'img{i}.jpg', 0) for i in range(3)]

# 2. 计算平均值 (需转为 float 避免溢出，最后转回 uint8)
avg_img = np.mean(imgs, axis=0).astype(np.uint8)

# 3. 计算中值 (中值对噪点更鲁棒)
med_img = np.median(imgs, axis=0).astype(np.uint8)

# 4. 保存结果
cv2.imwrite('avg.jpg', avg_img)
cv2.imwrite('med.jpg', med_img)