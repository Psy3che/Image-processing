import cv2
import numpy as np


imgs = [cv2.imread(f'img{i}.jpg', 0) for i in range(3)]

avg_img = np.mean(imgs, axis=0).astype(np.uint8)

med_img = np.median(imgs, axis=0).astype(np.uint8)


cv2.imwrite('avg.jpg', avg_img)
cv2.imwrite('med.jpg', med_img)