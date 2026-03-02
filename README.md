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

```bash
python image_processing_full.py equalize input.JPG out1.pgm
python image_processing_full.py average input.JPG out2.pgm --kernel 5 --mode ignore
python image_processing_full.py median input.JPG out3.pgm --kernel 3 --mode zero
python image_processing_full.py threshold input.JPG out4.pgm --th 128
python image_processing_full.py erosion input.JPG out5.pgm --kernel 3
python image_processing_full.py dilation input.JPG out6.pgm --kernel 5
python image_processing_full.py hitmiss input.JPG out7.pgm
python image_processing_full.py gray_dilate input.JPG out8.pgm --kernel 5
python image_processing_full.py gray_erode input.JPG out9.pgm --kernel 5```
