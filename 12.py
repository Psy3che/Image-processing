import cv2
import numpy as np

# 1. 预处理：读取图像并变换到频域
img = cv2.imread('input.jpg', 0)
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
dft = np.fft.fftshift(np.fft.fft2(img))

# 2. 生成距离矩阵 D (所有滤波器的基础)
y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
D = np.sqrt(x*x + y*y)
D0, n = 30, 2  # 截止频率和巴特沃斯阶数

# 3. 定义三种低通 Mask
mask_i = (D <= D0).astype(float)                # ILPF (理想)
mask_b = 1 / (1 + (D / D0)**(2 * n))            # BLPF (巴特沃斯)
mask_g = np.exp(-(D**2) / (2 * (D0**2)))        # GLPF (高斯)

# 4. 逆变换回图像的快捷函数
def to_img(m):
    res = np.abs(np.fft.ifft2(np.fft.ifftshift(dft * m)))
    return cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# 5. 获取结果
lp_i, lp_b, lp_g = to_img(mask_i), to_img(mask_b), to_img(mask_g)

# 6. 成对对比（两两相减看差异）
diff_i_b = cv2.absdiff(lp_i, lp_b) # 理想 vs 巴特沃斯
diff_b_g = cv2.absdiff(lp_b, lp_g) # 巴特沃斯 vs 高斯

cv2.imwrite('diff_ideal_butterworth.jpg', diff_i_b)
cv2.imwrite('diff_butterworth_gaussian.jpg', diff_b_g)
# import cv2
# import numpy as np

# # ===================== 输入参数 =====================
# img = cv2.imread("input.jpg", 0)   # OpenCV 读取图像
# D0 = 20                            # 低截止频率
# D1 = 50                            # 高截止频率
# # ====================================================

# # 傅里叶变换（仅滤波部分用numpy）
# dft = np.fft.fftshift(np.fft.fft2(img))
# h, w = img.shape
# y, x = np.ogrid[:h, :w]
# dist = np.sqrt((y - h//2)**2 + (x - w//2)**2)

# # ===================== 滤波器选择 =====================
# # 1. IBPF 理想带通滤波（保留 D0 ~ D1 之间的频率）
# mask = (dist > D0) & (dist <= D1)

# # 2. IBSF 理想带阻滤波（去掉 D0 ~ D1 之间的频率）
# # mask = (dist <= D0) | (dist > D1)
# # ======================================================

# # 滤波 + 逆傅里叶变换
# filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(dft * mask)))

# # ===================== OpenCV 操作 =====================
# # 归一化到 0~255（必须用OpenCV）
# filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# # 保存图像（必须用OpenCV）
# cv2.imwrite("ibpf_result.jpg", filtered)
