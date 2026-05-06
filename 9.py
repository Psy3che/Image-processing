import numpy as np

matrix = np.random.randint(0, 256, (15, 15), dtype=np.uint8)
dft = np.fft.fft2(matrix)
idft = np.fft.ifft2(dft).real.astype(np.uint8)

print("Original:\n", matrix)
print("IDFT Result:\n", idft)