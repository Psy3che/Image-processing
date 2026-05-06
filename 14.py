import cv2
import numpy as np

img = cv2.imread('input.jpg', 0)
kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) # Prewitt
grad_x = cv2.filter2D(img, -1, kernel_x)
cv2.imwrite('edge_template.jpg', grad_x)