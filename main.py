import numpy as np
import cmath
import cv2
import sys

# CONTROL PLANE
IMAGE_FILENAME = 'froggohhh.jpg'
OUTPUT_FILENAME = 'out.txt'
COEFF_COUNT = 200
SCALE_FACTOR = 2
SHOW_OUTPUT = False


class FourierCoefficient:
    def __init__(self, frequency, phase, magnitude):
        self.frequency = frequency
        self.phase = phase
        self.magnitude = magnitude


def image_binary(img, threshold=127, cross_filtering=True):
    shape_img = img.shape[0:2]
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = np.empty(shape=shape_img, dtype=np.uint8)
    for i in range(shape_img[0]):
        for j in range(shape_img[1]):
            gray_img[i, j] = sum(img[i, j]) / 3
    filtered = np.empty(shape=shape_img, dtype=np.uint8)
    for i in range(shape_img[0]):
        for j in range(shape_img[1]):
            val = int(gray_img[i, j])
            if cross_filtering:
                count = 1
                for k in range(1, 9, 2):
                    ii, jj = i - 1 + int(k / 3), j - 1 + k % 3
                    if 0 <= ii < shape_img[0] and 0 <= jj < shape_img[1]:
                        val += gray_img[ii, jj]
                        count += 1
                val /= count
            filtered[i, j] = 0 if val <= threshold else 255
    return filtered


def ftransform(x, y, n_coeffs: int, scale_factor: float = 1):
    assert len(x) == len(y)
    n_elems = len(x)
    t = np.linspace(0, 1, n_coeffs)
    fc = [scale_factor * complex(x[i], y[i]) for i in range(n_elems)]
    out = np.empty(shape=(2 * n_coeffs + 1,), dtype=FourierCoefficient)
    for n in range(-n_coeffs, n_coeffs + 1):
        e = [cmath.exp(-2j * cmath.pi * n * t_i) for t_i in t]
        Cn = sum(fc[i] * e[i] for i in range(n_elems))
        i = 2 * abs(n) - (1 if n > 0 else 0)
        out[i] = FourierCoefficient(n, np.angle(Cn), abs(Cn))
    return out


def image_to_curve(img):
    x = []
    y = []

    # finding the starting point
    starting_point = np.where(img == 0)[0]
    x.append(starting_point[1])
    y.append(starting_point[0])

    return x, y


if __name__ == '__main__':
    # working on the image in order to make it binary
    image = cv2.imread(IMAGE_FILENAME)
    binimg = image_binary(image, cross_filtering=False, threshold=60)

    # getting a one-line closed curve
    x, y = image_to_curve(binimg)

    # getting fourier coefficients
    coeffs = ftransform(x, y, n_coeffs=COEFF_COUNT, scale_factor=SCALE_FACTOR)

    # printing coefficients on the output file
    stdout_bak = sys.stdout
    with open(OUTPUT_FILENAME, 'w') as file:
        sys.stdout = file
        print(f'{COEFF_COUNT * 2}')
        for cn in coeffs:
            print(f'{cn.frequency} {cn.phase} {cn.magnitude}')
    sys.stdout = stdout_bak
