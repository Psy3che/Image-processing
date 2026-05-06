import cv2
import numpy as np

img = cv2.imread("input.jpg", 0)   
D0 = 30                 

dft = np.fft.fftshift(np.fft.fft2(img))
h, w = img.shape

y, x = np.ogrid[:h, :w]
# ILPF
mask = np.sqrt((y - h//2)**2 + (x - w//2)**2) <= D0


















# IHPF
# mask = np.sqrt((y - h//2)**2 + (x - w//2)**2) > D0
# # ===================== 滤波器选择 =====================
# # 1. IBPF 理想带通滤波（保留 D0 ~ D1 之间的频率）
# mask = (dist > D0) & (dist <= D1)

# # 2. IBSF 理想带阻滤波（去掉 D0 ~ D1 之间的频率）
# # mask = (dist <= D0) | (dist > D1)
# # ============================

filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(dft * mask)))

filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
cv2.imwrite("filter_result.jpg", filtered)
