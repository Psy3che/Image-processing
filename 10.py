import cv2
import numpy as np

# ===================== 输入参数 =====================
img = cv2.imread("input.jpg", 0)    # OpenCV 读取图像
D0 = 30                             # 截止频率（可自由修改）
# ====================================================

# FFT 频域滤波（必须用 numpy，无法用 OpenCV 替代）
dft = np.fft.fftshift(np.fft.fft2(img))
h, w = img.shape

# 生成低通掩码
y, x = np.ogrid[:h, :w]
mask = np.sqrt((y - h//2)**2 + (x - w//2)**2) <= D0

# 滤波 + 逆变换
filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(dft * mask)))

# ===================== OpenCV 完成所有其他操作 =====================
filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
cv2.imwrite("filter_result.jpg", filtered)