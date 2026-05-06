import cv2
import numpy as np

# ===================== 输入参数 =====================
img = cv2.imread("input.jpg", 0)   # OpenCV 读取图像
D0 = 20                            # 低截止频率
D1 = 50                            # 高截止频率
# ====================================================

# 傅里叶变换（仅滤波部分用numpy）
dft = np.fft.fftshift(np.fft.fft2(img))
h, w = img.shape
y, x = np.ogrid[:h, :w]
dist = np.sqrt((y - h//2)**2 + (x - w//2)**2)

# ===================== 滤波器选择 =====================
# 1. IBPF 理想带通滤波（保留 D0 ~ D1 之间的频率）
mask = (dist > D0) & (dist <= D1)

# 2. IBSF 理想带阻滤波（去掉 D0 ~ D1 之间的频率）
# mask = (dist <= D0) | (dist > D1)
# ======================================================

# 滤波 + 逆傅里叶变换
filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(dft * mask)))

# ===================== OpenCV 操作 =====================
# 归一化到 0~255（必须用OpenCV）
filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# 保存图像（必须用OpenCV）
cv2.imwrite("ibpf_result.jpg", filtered)