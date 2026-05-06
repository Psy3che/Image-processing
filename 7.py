
import cv2
import numpy as np

img = cv2.imread('input.jpg', 0)


kernel = np.array([[0, 1, 0],
                   [1, -1, 1],
                   [0, 1, 0]], dtype=np.int8)


hm = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel)

cv2.imwrite('hit_miss.jpg', hm)