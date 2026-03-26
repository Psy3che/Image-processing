# Image-processing
Histogram equalization on grayscale images (reading and writing the images is based on OpenCV module).

Averaging filter on grayscale images (reading and writing the images is based on OpenCV module).
Program parameters: kernel size (3×3, 5×5, or 7×7) and handling of missing positions: only full overlap allowed, missing elements treated as zeros, or missing elements ignored.

Median filter on grayscale images (reading and writing the images is based on OpenCV module).
Program parameters: kernel size (3×3, 5×5, or 7×7) and handling of missing positions: only full overlap allowed, missing elements treated as zeros, or missing elements ignored.

Image averaging and median computation (reading and writing the images is based on OpenCV module).

Thresholding on grayscale images (reading and writing the images is based on OpenCV module).

Morphological operator (erosion, dilation, opening, or closing). Only one needs to be implemented.
OpenCV functions may be used for reading and saving the images.

Hit-or-Miss transformation.
OpenCV functions may be used for reading and saving the images.

Grayscale dilation and erosion.
OpenCV functions may be used for reading and saving the images.

# Command Format

程序运行格式：

```bash
python image_processing_full_updated.py <operation> <input> <output> [parameters]
```

---

# 1 Histogram Equalization

```bash
python image_processing_full_updated.py equalize gradient.png out1_equalize.png
```

---

# 2 Averaging Filter

参数：

* `--kernel` : 3 / 5 / 7
* `--mode` :

| 模式     | 说明      |
| ------ | ------- |
| valid  | 只允许完全重叠 |
| zero   | 缺失元素视为0 |
| ignore | 忽略缺失元素  |

示例：

```bash
python image_processing_full_updated.py average noise.png out2_avg.png --kernel 3 --mode zero
```

---

# 3 Median Filter

```bash
python image_processing_full_updated.py median noise.png out3_median.png --kernel 3 --mode ignore
```

---

# 4 Image Averaging

```bash
python image_processing_full_updated.py img_avg img1.png img2.png img3.png out4_imgavg.png
```

---

# 5 Image Median

```bash
python image_processing_full_updated.py img_med img1.png img2.png img3.png out5_imgmed.png
```

---

# 6 Thresholding

```bash
python image_processing_full_updated.py threshold gradient.png out6_threshold.png --th 120
```

---

# 7 Morphological Operator (Erosion)

```bash
python image_processing_full_updated.py erosion shapes.png out7_erosion.png --kernel 3
```

---

# 8 Hit-or-Miss Transformation

```bash
python image_processing_full_updated.py hitmiss shapes.png out8_hitmiss.png
```

---

# 9 Grayscale Dilation

```bash
python image_processing_full_updated.py gray_dilate gradient.png out9_dilate.png --kernel 3
```

Grayscale erosion：

```bash
python image_processing_full_updated.py gray_erode gradient.png out9_erode.png --kernel 3
```

---

# 10 Fourier Transform

作业要求：输入矩阵 **≤15×15**

```bash
python image_processing_full_updated.py dft matrix15.png out10_dft.png
```

---

# 11 Inverse Fourier Transform

```bash
python image_processing_full_updated.py idft matrix15.png out11_idft.png
```

---

# 12 Frequency Domain Filtering (ILPF / IHPF)

Ideal Low Pass Filter：

```bash
python image_processing_full_updated.py ilpf noise.png out12_ilpf.png --cutoff 30
```

Ideal High Pass Filter：

```bash
python image_processing_full_updated.py ihpf noise.png out12_ihpf.png --cutoff 30
```

---

# 13 Band Filters (IBPF / IBSF)

Ideal Band Pass Filter：

```bash
python image_processing_full_updated.py ibpf noise.png out13_ibpf.png --cutoff 20 --cutoff2 60
```

Ideal Band Stop Filter：

```bash
python image_processing_full_updated.py ibsf noise.png out13_ibsf.png --cutoff 20 --cutoff2 60
```

**17. Sobel Edge Detector（Sobel 边缘检测器）17. Sobel 边缘检测器**
```bash
python image_processing_full_updated.py edge_sobel input.JPG output_edge_sobel.pgm
```
**18. Prewitt Edge Detector（普雷维特边缘检测器）**
```bash
python image_processing_full_updated.py edge_prewitt input.JPG output_edge_prewitt.pgm
```
**19. Roberts Edge Detector（罗伯茨边缘检测器）**
```bash
python image_processing_full_updated.py edge_roberts input.JPG output_edge_roberts.pgm
```
**20. Edge Detector Comparison（边缘检测器比较）**
```bash
python image_processing_full_updated.py edge_compare input.JPG output_edge_compare.pgm
```
**21. Hough Transform (Rectangle Orientation)（霍夫变换：矩形方向检测）**
```bash
python image_processing_full_updated.py hough_rect input.JPG output_hough_rect.JPG
```
终端输出示例
Estimated rectangle orientation: 89.73 degrees
**22. Skeletonization（骨骼化）**
```bash
python image_processing_full_updated.py skeleton input.JPG output_skeleton.pgm
```
# Recommended Full Test Sequence

建议完整测试顺序：

```
1  equalize
2  average
3  median
4  img_avg
5  img_med
6  threshold
7  erosion
8  hitmiss
9  gray_dilate
10 gray_erode
11 dft
12 idft
13 ilpf
14 ihpf
15 ibpf
16 ibsf
```

