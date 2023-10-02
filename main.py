import math

import numpy as np
import cmath
import cv2
import sys

# CONTROL PLANE
IMAGE_FILENAME = 'froggohhh.jpg'
OUTPUT_FILENAME = 'out.txt'
COEFF_COUNT = 100
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
    t = np.linspace(0, 1, n_elems)
    fc = [scale_factor * complex(x[i], y[i]) for i in range(n_elems)]
    out = np.empty(shape=(2 * n_coeffs + 1,), dtype=FourierCoefficient)
    for n in range(-n_coeffs, n_coeffs + 1):
        e = [cmath.exp(-2j * cmath.pi * n * t_i) for t_i in t]
        Cn = sum(fc[i] * e[i] for i in range(n_elems))
        i = 2 * abs(n) - (1 if n > 0 else 0)
        out[i] = FourierCoefficient(n, np.angle(Cn), abs(Cn))
    return out


def image_to_curve(img, plot_curve=True):
    curve_x = []
    curve_y = []

    # extending img matrix to avoid indexes overflow
    w_img = img.shape[1]
    h_img = img.shape[0]
    tmp_img = np.ones(shape=(h_img + 2, w_img + 2))
    for i in range(h_img):
        for j in range(w_img):
            tmp_img[i + 1, j + 1] = img[i, j]
    img = tmp_img

    # finding the starting point
    starting_point = np.argwhere(img == 0)[0]
    x = starting_point[1]
    y = starting_point[0]

    # here the algorith starts
    vx = 1
    vy = 0
    while True:
        # saving the current point
        curve_x.append(x)
        curve_y.append(y)
        # looking for best next point near the current, using as metric dot product between velocity vectors
        # v0 dot v1. the best value will tell which point will be the next
        score = -1
        next_x = x
        next_y = y
        next_vx = 0
        next_vy = 0
        for i in range(y - 1, y + 2):  # -1 0 +1 (+1 to include the end)
            for j in range(x - 1, x + 2):
                if img[i, j] != 0:
                    continue
                if i == y and j == x:
                    continue
                dx = j - x
                dy = i - y
                norm = np.linalg.norm([dx, dy])
                vx2 = dx / norm
                vy2 = dy / norm
                dot = vx * vx2 + vy * vy2
                if dot > score:
                    score = dot
                    next_x = j
                    next_y = i
                    next_vx = vx2
                    next_vy = vy2
        # if the best point is the starting point of the curve, then the curve is built
        if next_x == starting_point[1] and next_y == starting_point[0]:
            break
        # if the best point is the current one, it means that the curve is not closed, and so it returns
        if x == next_x and y == next_y:
            break

        # sets the new current values
        x = next_x
        y = next_y
        vx = next_vx
        vy = next_vy

    return curve_x, curve_y


if __name__ == '__main__':
    # tests

    # working on the image in order to make it binary
    image = cv2.imread(IMAGE_FILENAME)
    binimg = image_binary(image, cross_filtering=False, threshold=150)
    cv2.imwrite("binimg.bmp", binimg)

    # getting a one-line closed curve
    curve_x, curve_y = image_to_curve(binimg)

    # show a test output
    out_curve = np.ones(shape=binimg.shape)
    print(len(curve_x))
    for c in range(len(curve_x)):
        out_curve[curve_y[c] - 1, curve_x[c] - 1] = 0

    # img = np.stack((out_curve,) * 3, -1)
    # img = img.astype(np.uint8)
    # grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.threshold(grayed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # cv2.imwrite("imgout.bmp", thresh)

    cv2.imshow("Test output", out_curve)
    cv2.waitKey(0)

    # getting fourier coefficients
    coeffs = ftransform(curve_x, curve_y, n_coeffs=COEFF_COUNT, scale_factor=SCALE_FACTOR)

    # printing coefficients on the output file
    stdout_bak = sys.stdout
    with open(OUTPUT_FILENAME, 'w') as file:
        sys.stdout = file
        print(f'{COEFF_COUNT * 2}')
        for i in range(1, len(coeffs)):
            print(f'{coeffs[i].frequency} {coeffs[i].phase} {coeffs[i].magnitude}')
    sys.stdout = stdout_bak
