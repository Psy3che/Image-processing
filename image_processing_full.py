import cv2
import numpy as np
import argparse
import sys


# =====================================
# Utility
# =====================================

def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Cannot read image:", path)
        sys.exit(1)
    return img


def save_image(path, img):
    cv2.imwrite(path, img)
import cv2

images = ["img1.JPG", "img2.JPG", "img3.JPG"]
target_size = (256, 256)

for p in images:
    img = cv2.imread(p)
    img_resized = cv2.resize(img, target_size)
    cv2.imwrite(p, img_resized)

# =====================================
# 1. Histogram Equalization
# =====================================

def histogram_equalization(img):
    hist = np.bincount(img.flatten(), minlength=256)
    cdf = hist.cumsum()
    cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf = cdf.astype(np.uint8)
    return cdf[img]


# =====================================
# 2. Averaging Filter
# mode: valid | zero | ignore
# =====================================

def averaging_filter(img, kernel_size, mode):
    h, w = img.shape
    pad = kernel_size // 2

    if mode == "valid":
        padded = img
    else:
        padded = np.pad(img, pad, mode='constant')

    result = np.zeros_like(img, dtype=np.uint8)

    for i in range(h):
        for j in range(w):

            if mode == "valid":
                if i - pad < 0 or i + pad >= h or j - pad < 0 or j + pad >= w:
                    continue
                region = img[i-pad:i+pad+1, j-pad:j+pad+1]
                result[i, j] = np.mean(region)

            elif mode == "ignore":
                region = padded[i:i+kernel_size, j:j+kernel_size]
                valid_pixels = region.flatten()
                valid_pixels = valid_pixels[valid_pixels != 0]
                if len(valid_pixels) > 0:
                    result[i, j] = np.mean(valid_pixels)

            else:  # zero
                region = padded[i:i+kernel_size, j:j+kernel_size]
                result[i, j] = np.mean(region)

    return result


# =====================================
# 3. Median Filter
# =====================================

def median_filter(img, kernel_size, mode):
    h, w = img.shape
    pad = kernel_size // 2

    if mode == "valid":
        padded = img
    else:
        padded = np.pad(img, pad, mode='constant')

    result = np.zeros_like(img, dtype=np.uint8)

    for i in range(h):
        for j in range(w):

            if mode == "valid":
                if i - pad < 0 or i + pad >= h or j - pad < 0 or j + pad >= w:
                    continue
                region = img[i-pad:i+pad+1, j-pad:j+pad+1]
                result[i, j] = np.median(region)

            elif mode == "ignore":
                region = padded[i:i+kernel_size, j:j+kernel_size]
                valid_pixels = region.flatten()
                valid_pixels = valid_pixels[valid_pixels != 0]
                if len(valid_pixels) > 0:
                    result[i, j] = np.median(valid_pixels)

            else:  # zero
                region = padded[i:i+kernel_size, j:j+kernel_size]
                result[i, j] = np.median(region)

    return result


# =====================================
# 4. Image Averaging
# =====================================

def image_average(paths):
    images = [read_image(p).astype(np.float32) for p in paths]
    avg = np.mean(images, axis=0)
    return avg.astype(np.uint8)


# =====================================
# 5. Image Median
# =====================================

def image_median(paths):
    images = [read_image(p) for p in paths]
    med = np.median(images, axis=0)
    return med.astype(np.uint8)


# =====================================
# 6. Thresholding
# =====================================

def thresholding(img, threshold):
    return np.where(img >= threshold, 255, 0).astype(np.uint8)


# =====================================
# 7. Binary Erosion (Morphological Operator)
# =====================================

def binary_erosion(img, kernel_size):
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(img, kernel)


# =====================================
# 8. Hit-or-Miss
# =====================================

def hit_or_miss(img):
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    kernel1 = np.array([[0,1,0],
                        [1,1,1],
                        [0,1,0]], dtype=np.uint8)

    kernel2 = np.array([[1,0,1],
                        [0,0,0],
                        [1,0,1]], dtype=np.uint8)

    erode1 = cv2.erode(img, kernel1)
    erode2 = cv2.erode(255 - img, kernel2)

    return cv2.bitwise_and(erode1, erode2)


# =====================================
# 9. Grayscale Dilation & Erosion
# =====================================

def grayscale_dilation(img, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(img, kernel)


def grayscale_erosion(img, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(img, kernel)


# =====================================
# Command Line Interface
# =====================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation")
    parser.add_argument("input", nargs="+")
    parser.add_argument("output")
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--mode", type=str, default="zero",
                        choices=["valid", "zero", "ignore"])
    parser.add_argument("--th", type=int, default=128)

    args = parser.parse_args()

    if args.operation in ["img_avg", "img_med"]:
        if args.operation == "img_avg":
            result = image_average(args.input)
        else:
            result = image_median(args.input)
    else:
        img = read_image(args.input[0])

        if args.operation == "equalize":
            result = histogram_equalization(img)

        elif args.operation == "average":
            result = averaging_filter(img, args.kernel, args.mode)

        elif args.operation == "median":
            result = median_filter(img, args.kernel, args.mode)

        elif args.operation == "threshold":
            result = thresholding(img, args.th)

        elif args.operation == "erosion":
            result = binary_erosion(img, args.kernel)

        elif args.operation == "hitmiss":
            result = hit_or_miss(img)

        elif args.operation == "gray_dilate":
            result = grayscale_dilation(img, args.kernel)

        elif args.operation == "gray_erode":
            result = grayscale_erosion(img, args.kernel)

        else:
            print("Unknown operation")
            sys.exit(1)

    save_image(args.output, result)


if __name__ == "__main__":
    main()