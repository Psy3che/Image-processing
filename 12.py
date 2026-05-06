import cv2
import numpy as np


img = cv2.imread('input.jpg', 0)

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

dft = np.fft.fftshift(np.fft.fft2(img))


y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
D = np.sqrt(x*x + y*y)
D0, n = 30, 2 


mask_ilpf = (D <= D0).astype(float)
mask_blpf = 1 / (1 + (D / D0)**(2 * n))
mask_glpf = np.exp(-(D**2) / (2 * (D0**2)))


mask_ihpf = 1 - mask_ilpf
mask_bhpf = 1 - mask_blpf
mask_ghpf = 1 - mask_glpf

def to_img(mask):
    f_shift = dft * mask
    img_back = np.fft.ifft2(np.fft.ifftshift(f_shift))
    img_back = np.abs(img_back)
    return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

lp_i, lp_b, lp_g = to_img(mask_ilpf), to_img(mask_blpf), to_img(mask_glpf)
hp_i, hp_b, hp_g = to_img(mask_ihpf), to_img(mask_bhpf), to_img(mask_ghpf)

def get_enhanced_diff(img1, img2):
    
    diff = cv2.absdiff(img1, img2)
    return cv2.equalizeHist(diff)

diff_lp_ib = get_enhanced_diff(lp_i, lp_b)   
diff_lp_bg = get_enhanced_diff(lp_b, lp_g)  
diff_lp_ig = get_enhanced_diff(lp_i, lp_g)   
diff_hp_ib = get_enhanced_diff(hp_i, hp_b)
diff_hp_bg = get_enhanced_diff(hp_b, hp_g)
diff_hp_ig = get_enhanced_diff(hp_i, hp_g)

cv2.imwrite('diff_lp_ideal_butter.jpg', diff_lp_ib)
cv2.imwrite('diff_lp_butter_gauss.jpg', diff_lp_bg)
cv2.imwrite('diff_hp_ideal_gauss.jpg', diff_hp_ig)
