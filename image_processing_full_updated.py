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
    cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf = cdf.astype(np.uint8)
    return cdf[img]


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

            else:
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

            else:
                region = padded[i:i+kernel_size, j:j+kernel_size]
                result[i, j] = np.median(region)

    return result


# =============================================================
# 4. Image Averaging and Median of Multiple Images
# =============================================================

def image_average(paths):

    imgs = [read_image(p).astype(np.float32) for p in paths]

    avg = np.mean(imgs, axis=0)

    return avg.astype(np.uint8)


def image_median(paths):

    imgs = [read_image(p) for p in paths]

    med = np.median(imgs, axis=0)

    return med.astype(np.uint8)


# =============================================================
# 5. Thresholding on Grayscale Images
# =============================================================

def thresholding(img, th):

    return np.where(img >= th, 255, 0).astype(np.uint8)


# =============================================================
# 6. Morphological Operator (Erosion implemented)
# =============================================================

def binary_erosion(img, kernel_size):

    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    return cv2.erode(img, kernel)


# =============================================================
# 7. Hit‑or‑Miss Transformation
# =============================================================

def hit_or_miss(img):

    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    kernel1 = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)

    kernel2 = np.array([[1,0,1],[0,0,0],[1,0,1]], dtype=np.uint8)

    er1 = cv2.erode(img, kernel1)

    er2 = cv2.erode(255-img, kernel2)

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

    M,N = img.shape

    F = np.zeros((M,N), dtype=np.complex64)

    for u in range(M):
        for v in range(N):

            s = 0

            for x in range(M):
                for y in range(N):

                    angle = -2j*np.pi*((u*x/M)+(v*y/N))

                    s += img[x,y]*np.exp(angle)

            F[u,v] = s

    return F


def inverse_fourier_transform(F):

    M,N = F.shape

    img = np.zeros((M,N), dtype=np.float32)

    for x in range(M):
        for y in range(N):

            s = 0

            for u in range(M):
                for v in range(N):

                    angle = 2j*np.pi*((u*x/M)+(v*y/N))

                    s += F[u,v]*np.exp(angle)

            img[x,y] = np.real(s)/(M*N)

    return np.clip(img,0,255).astype(np.uint8)


# =============================================================
# Helper Functions for Frequency Filters
# =============================================================

def distance(u,v,M,N):

    return np.sqrt((u-M/2)**2 + (v-N/2)**2)


# =============================================================
# 10. Frequency Domain Filtering: ILPF and IHPF
# =============================================================

def ideal_low_pass_filter(img, D0):

    dft = np.fft.fft2(img)

    dft_shift = np.fft.fftshift(dft)

    M,N = img.shape

    H = np.zeros((M,N))

    for u in range(M):
        for v in range(N):

            if distance(u,v,M,N) <= D0:

                H[u,v] = 1

    G = dft_shift * H

    img_back = np.fft.ifft2(np.fft.ifftshift(G))

    img_back = np.abs(img_back)

    return np.clip(img_back,0,255).astype(np.uint8)


def ideal_high_pass_filter(img, D0):

    dft = np.fft.fft2(img)

    dft_shift = np.fft.fftshift(dft)

    M,N = img.shape

    H = np.ones((M,N))

    for u in range(M):
        for v in range(N):

            if distance(u,v,M,N) <= D0:

                H[u,v] = 0

    G = dft_shift * H

    img_back = np.fft.ifft2(np.fft.ifftshift(G))

    img_back = np.abs(img_back)

    return np.clip(img_back,0,255).astype(np.uint8)


# =============================================================
# 11. Frequency Domain Filtering: IBPF and IBSF
# =============================================================

def ideal_band_pass_filter(img, D1, D2):

    dft = np.fft.fft2(img)

    dft_shift = np.fft.fftshift(dft)

    M,N = img.shape

    H = np.zeros((M,N))

    for u in range(M):
        for v in range(N):

            d = distance(u,v,M,N)

            if D1 <= d <= D2:

                H[u,v] = 1

    G = dft_shift * H

    img_back = np.fft.ifft2(np.fft.ifftshift(G))

    img_back = np.abs(img_back)

    return np.clip(img_back,0,255).astype(np.uint8)


def ideal_band_stop_filter(img, D1, D2):

    dft = np.fft.fft2(img)

    dft_shift = np.fft.fftshift(dft)

    M,N = img.shape

    H = np.ones((M,N))

    for u in range(M):
        for v in range(N):

            d = distance(u,v,M,N)

            if D1 <= d <= D2:

                H[u,v] = 0

    G = dft_shift * H

    img_back = np.fft.ifft2(np.fft.ifftshift(G))

    img_back = np.abs(img_back)

    return np.clip(img_back,0,255).astype(np.uint8)


# =============================================================
# 12. Comparison of LPF and HPF Filters
# =============================================================

def difference_image(img1,img2):

    return cv2.absdiff(img1,img2)


# =============================================================
# 13. Comparison of Band Filters
# =============================================================

# For assignment comparison images can be generated
# using OpenCV operations


# =============================================================
# Command Line Interface
# =============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("operation")

    parser.add_argument("input", nargs="+")

    parser.add_argument("output")

    parser.add_argument("--kernel", type=int, default=3)

    parser.add_argument("--mode", default="zero", choices=["valid","zero","ignore"])

    parser.add_argument("--th", type=int, default=128)

    parser.add_argument("--cutoff", type=int, default=30)

    parser.add_argument("--cutoff2", type=int, default=60)


    args = parser.parse_args()


    if args.operation in ["img_avg","img_med"]:

        if args.operation == "img_avg":

            result = image_average(args.input)

        else:

            result = image_median(args.input)


    else:

        img = read_image(args.input[0])


        if args.operation == "equalize":

            result = histogram_equalization(img)


        elif args.operation == "average":

            result = averaging_filter(img,args.kernel,args.mode)


        elif args.operation == "median":

            result = median_filter(img,args.kernel,args.mode)


        elif args.operation == "threshold":

            result = thresholding(img,args.th)


        elif args.operation == "erosion":

            result = binary_erosion(img,args.kernel)


        elif args.operation == "hitmiss":

            result = hit_or_miss(img)


        elif args.operation == "gray_dilate":

            result = grayscale_dilation(img,args.kernel)


        elif args.operation == "gray_erode":

            result = grayscale_erosion(img,args.kernel)


        elif args.operation == "dft":

            F = fourier_transform(img)

            result = np.log(1+np.abs(F))

            result = cv2.normalize(result,None,0,255,cv2.NORM_MINMAX)

            result = result.astype(np.uint8)


        elif args.operation == "idft":

            F = fourier_transform(img)

            result = inverse_fourier_transform(F)


        elif args.operation == "ilpf":

            result = ideal_low_pass_filter(img,args.cutoff)


        elif args.operation == "ihpf":

            result = ideal_high_pass_filter(img,args.cutoff)


        elif args.operation == "ibpf":

            result = ideal_band_pass_filter(img,args.cutoff,args.cutoff2)


        elif args.operation == "ibsf":

            result = ideal_band_stop_filter(img,args.cutoff,args.cutoff2)


        else:

            print("Unknown operation")

            sys.exit(1)


    save_image(args.output,result)


if __name__ == "__main__":

    main()
