import cv2
import numpy as np
import argparse
import sys

# =============================================================
# Utility
# =============================================================

def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Cannot read image:", path)
        sys.exit(1)
    return img


def save_image(path, img):
    cv2.imwrite(path, img)


# =============================================================
# 1. Histogram Equalization on Grayscale Images
# =============================================================

def histogram_equalization(img):
    hist = np.bincount(img.flatten(), minlength=256)
    cdf = hist.cumsum()

    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_masked = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
    cdf_final = np.ma.filled(cdf_masked, 0).astype(np.uint8)

    return cdf_final[img]


# =============================================================
# 2. Averaging Filter on Grayscale Images
# kernel: 3 / 5 / 7
# mode: valid | zero | ignore
# =============================================================

def averaging_filter(img, kernel_size, mode):
    h, w = img.shape
    pad = kernel_size // 2

    if mode == "valid":
        padded = img
    else:
        padded = np.pad(img, pad, mode="constant")

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
                vals = region.flatten()
                vals = vals[vals != 0]
                if len(vals) > 0:
                    result[i, j] = np.mean(vals)

            else:  # zero
                region = padded[i:i+kernel_size, j:j+kernel_size]
                result[i, j] = np.mean(region)

    return result


# =============================================================
# 3. Median Filter on Grayscale Images
# kernel: 3 / 5 / 7
# mode: valid | zero | ignore
# =============================================================

def median_filter(img, kernel_size, mode):
    h, w = img.shape
    pad = kernel_size // 2

    if mode == "valid":
        padded = img
    else:
        padded = np.pad(img, pad, mode="constant")

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
                vals = region.flatten()
                vals = vals[vals != 0]
                if len(vals) > 0:
                    result[i, j] = np.median(vals)

            else:  # zero
                region = padded[i:i+kernel_size, j:j+kernel_size]
                result[i, j] = np.median(region)

    return result


# =============================================================
# 4. Image Averaging and Median of Multiple Images
# =============================================================

def image_average(paths):
    imgs = [read_image(p).astype(np.float32) for p in paths]

    first_shape = imgs[0].shape
    for img in imgs:
        if img.shape != first_shape:
            print("Error: All images must have the same size for img_avg.")
            sys.exit(1)

    avg = np.mean(imgs, axis=0)
    return avg.astype(np.uint8)


def image_median(paths):
    imgs = [read_image(p) for p in paths]

    first_shape = imgs[0].shape
    for img in imgs:
        if img.shape != first_shape:
            print("Error: All images must have the same size for img_med.")
            sys.exit(1)

    med = np.median(imgs, axis=0)
    return med.astype(np.uint8)


# =============================================================
# 5. Thresholding on Grayscale Images
# =============================================================

def thresholding(img, th):
    return np.where(img >= th, 255, 0).astype(np.uint8)


# =============================================================
# 6. Morphological Operator (Binary Erosion implemented)
# =============================================================

def binary_erosion(img, kernel_size):
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(img, kernel)


# =============================================================
# 7. Hit-or-Miss Transformation
# =============================================================

def hit_or_miss(img):
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    kernel1 = np.array([[0,1,0],
                        [1,1,1],
                        [0,1,0]], dtype=np.uint8)

    kernel2 = np.array([[1,0,1],
                        [0,0,0],
                        [1,0,1]], dtype=np.uint8)

    er1 = cv2.erode(img, kernel1)
    er2 = cv2.erode(255 - img, kernel2)

    return cv2.bitwise_and(er1, er2)


# =============================================================
# 8. Grayscale Dilation and Erosion
# =============================================================

def grayscale_dilation(img, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(img, kernel)


def grayscale_erosion(img, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(img, kernel)


# =============================================================
# 9. Fourier Transform and Inverse (manual, <=15x15 matrix)
# =============================================================

def fourier_transform(img):
    img = img.astype(np.float32)
    M, N = img.shape
    F = np.zeros((M, N), dtype=np.complex64)

    for u in range(M):
        for v in range(N):
            s = 0
            for x in range(M):
                for y in range(N):
                    angle = -2j * np.pi * ((u * x / M) + (v * y / N))
                    s += img[x, y] * np.exp(angle)
            F[u, v] = s

    return F


def inverse_fourier_transform(F):
    M, N = F.shape
    img = np.zeros((M, N), dtype=np.float32)

    for x in range(M):
        for y in range(N):
            s = 0
            for u in range(M):
                for v in range(N):
                    angle = 2j * np.pi * ((u * x / M) + (v * y / N))
                    s += F[u, v] * np.exp(angle)
            img[x, y] = np.real(s) / (M * N)

    return np.clip(img, 0, 255).astype(np.uint8)


# =============================================================
# Helper Functions for Frequency Filters
# =============================================================

def distance(u, v, M, N):
    return np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)


# =============================================================
# 10. Frequency Domain Filtering: ILPF and IHPF
# =============================================================

def ideal_low_pass_filter(img, D0):
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    M, N = img.shape
    H = np.zeros((M, N))

    for u in range(M):
        for v in range(N):
            if distance(u, v, M, N) <= D0:
                H[u, v] = 1

    G = dft_shift * H
    img_back = np.fft.ifft2(np.fft.ifftshift(G))
    img_back = np.abs(img_back)

    return np.clip(img_back, 0, 255).astype(np.uint8)


def ideal_high_pass_filter(img, D0):
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    M, N = img.shape
    H = np.ones((M, N))

    for u in range(M):
        for v in range(N):
            if distance(u, v, M, N) <= D0:
                H[u, v] = 0

    G = dft_shift * H
    img_back = np.fft.ifft2(np.fft.ifftshift(G))
    img_back = np.abs(img_back)

    return np.clip(img_back, 0, 255).astype(np.uint8)


# =============================================================
# 11. Frequency Domain Filtering: IBPF and IBSF
# =============================================================

def ideal_band_pass_filter(img, D1, D2):
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    M, N = img.shape
    H = np.zeros((M, N))

    for u in range(M):
        for v in range(N):
            d = distance(u, v, M, N)
            if D1 <= d <= D2:
                H[u, v] = 1

    G = dft_shift * H
    img_back = np.fft.ifft2(np.fft.ifftshift(G))
    img_back = np.abs(img_back)

    return np.clip(img_back, 0, 255).astype(np.uint8)


def ideal_band_stop_filter(img, D1, D2):
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    M, N = img.shape
    H = np.ones((M, N))

    for u in range(M):
        for v in range(N):
            d = distance(u, v, M, N)
            if D1 <= d <= D2:
                H[u, v] = 0

    G = dft_shift * H
    img_back = np.fft.ifft2(np.fft.ifftshift(G))
    img_back = np.abs(img_back)

    return np.clip(img_back, 0, 255).astype(np.uint8)


# =============================================================
# 12. Comparison Helper
# =============================================================

def difference_image(img1, img2):
    return cv2.absdiff(img1, img2)


# =============================================================
# 13. Edge Detector (template-based)
# Sobel / Prewitt / Roberts
# =============================================================

def convolve2d_manual(img, kernel, mode="zero"):
    h, w = img.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    result = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            result[i, j] = np.sum(region * kernel)

    return result


def edge_detector(img, method="sobel"):
    img = img.astype(np.float32)

    if method == "sobel":
        gx_kernel = np.array([[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]], dtype=np.float32)

        gy_kernel = np.array([[-1, -2, -1],
                              [ 0,  0,  0],
                              [ 1,  2,  1]], dtype=np.float32)

    elif method == "prewitt":
        gx_kernel = np.array([[-1, 0, 1],
                              [-1, 0, 1],
                              [-1, 0, 1]], dtype=np.float32)

        gy_kernel = np.array([[-1, -1, -1],
                              [ 0,  0,  0],
                              [ 1,  1,  1]], dtype=np.float32)

    elif method == "roberts":
        gx_kernel = np.array([[1, 0],
                              [0, -1]], dtype=np.float32)

        gy_kernel = np.array([[0, 1],
                              [-1, 0]], dtype=np.float32)

    else:
        print("Error: Unknown edge detector method.")
        sys.exit(1)

    gx = convolve2d_manual(img, gx_kernel)
    gy = convolve2d_manual(img, gy_kernel)

    magnitude = np.sqrt(gx**2 + gy**2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    return magnitude.astype(np.uint8)


# =============================================================
# 14. Comparison of Edge Detectors
# =============================================================

def compare_edge_detectors(img):
    sobel = edge_detector(img, "sobel")
    prewitt = edge_detector(img, "prewitt")
    roberts = edge_detector(img, "roberts")

    diff_sp = cv2.absdiff(sobel, prewitt)
    diff_sr = cv2.absdiff(sobel, roberts)
    diff_pr = cv2.absdiff(prewitt, roberts)

    top = np.hstack([sobel, prewitt, roberts])
    bottom = np.hstack([diff_sp, diff_sr, diff_pr])

    result = np.vstack([top, bottom])

    return result


# =============================================================
# 15. Hough Transform for Rectangle Orientation
# =============================================================

def rectangle_orientation_hough(img):
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    edges = edge_detector(binary, "sobel")
    _, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                            minLineLength=30, maxLineGap=10)

    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    angles = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            if angle < 0:
                angle += 180

            angles.append(angle)
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if len(angles) == 0:
        orientation = 0.0
    else:
        orientation = np.median(angles)

    cv2.putText(output, f"Angle: {orientation:.2f} deg",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    return output, orientation


# =============================================================
# 16. Skeletonization (Morphological Skeleton)
# =============================================================

def skeletonization(img):
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    skel = np.zeros(img.shape, np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    temp_img = img.copy()

    while True:
        eroded = cv2.erode(temp_img, kernel)
        opened = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(temp_img, opened)
        skel = cv2.bitwise_or(skel, temp)
        temp_img = eroded.copy()

        if cv2.countNonZero(temp_img) == 0:
            break

    return skel


# =============================================================
# Command Line Interface
# =============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Image Processing Full Assignment Tool"
    )

    parser.add_argument(
        "operation",
        help="""
Operations:
equalize, average, median, img_avg, img_med, threshold,
erosion, hitmiss, gray_dilate, gray_erode,
dft, idft, ilpf, ihpf, ibpf, ibsf,
edge_sobel, edge_prewitt, edge_roberts, edge_compare,
hough_rect, skeleton
"""
    )

    parser.add_argument("input", nargs="+", help="Input image(s)")
    parser.add_argument("output", help="Output image")

    parser.add_argument("--kernel", type=int, default=3, choices=[3,5,7],
                        help="Kernel size (3,5,7)")
    parser.add_argument("--mode", default="zero", choices=["valid", "zero", "ignore"],
                        help="Boundary handling mode")
    parser.add_argument("--th", type=int, default=128,
                        help="Threshold value")
    parser.add_argument("--cutoff", type=int, default=30,
                        help="Frequency cutoff")
    parser.add_argument("--cutoff2", type=int, default=60,
                        help="Second frequency cutoff for band filters")

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

        elif args.operation == "dft":
            F = fourier_transform(img)
            result = np.log(1 + np.abs(F))
            result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
            result = result.astype(np.uint8)

        elif args.operation == "idft":
            F = fourier_transform(img)
            result = inverse_fourier_transform(F)

        elif args.operation == "ilpf":
            result = ideal_low_pass_filter(img, args.cutoff)

        elif args.operation == "ihpf":
            result = ideal_high_pass_filter(img, args.cutoff)

        elif args.operation == "ibpf":
            result = ideal_band_pass_filter(img, args.cutoff, args.cutoff2)

        elif args.operation == "ibsf":
            result = ideal_band_stop_filter(img, args.cutoff, args.cutoff2)

        elif args.operation == "edge_sobel":
            result = edge_detector(img, "sobel")

        elif args.operation == "edge_prewitt":
            result = edge_detector(img, "prewitt")

        elif args.operation == "edge_roberts":
            result = edge_detector(img, "roberts")

        elif args.operation == "edge_compare":
            result = compare_edge_detectors(img)

        elif args.operation == "hough_rect":
            result, angle = rectangle_orientation_hough(img)
            print(f"Estimated rectangle orientation: {angle:.2f} degrees")

        elif args.operation == "skeleton":
            result = skeletonization(img)

        else:
            print("Unknown operation:", args.operation)
            sys.exit(1)

    save_image(args.output, result)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()