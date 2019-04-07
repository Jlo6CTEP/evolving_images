import os
import time
from copy import deepcopy

import numpy

from numpy import argwhere, zeros, uint8, full, arange, concatenate, empty, ndarray, array, clip, int32, int8, int16, \
    save, load
from numpy.random import exponential, randint, normal, choice
from scipy.spatial import Delaunay
from scipy.spatial import distance
from multiprocessing import Pool
import cv2

from evaluator import *

PICTURE = "./picture/Tower2.jpg"


if __name__ == '__main__':
    img = cv2.resize(cv2.imread(PICTURE), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    # gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (KERNEL_SIZE, KERNEL_SIZE), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, laplacian = cv2.threshold(cv2.Laplacian(gray, cv2.CV_8U), BLACK_THRESHOLD, MAX_BLACK, cv2.THRESH_BINARY)
    sobely = cv2.Sobel(gray, cv2.CV_8U, 1, 1, ksize=5)

    t1 = time.time()
    img2 = empty([SIZE, SIZE, 3], dtype=uint8)

    with Pool(8, maxtasksperchild=1) as p:
        time.sleep(1)
        for x in range(0, 4):
            for y in range(int(SIZE / F_SIZE)):
                a = p.apply_async(approximate_square,
                                  (img[x * F_SIZE: (x + 1) * F_SIZE, y * F_SIZE: (y + 1) * F_SIZE],
                                   laplacian[x * F_SIZE: (x + 1) * F_SIZE, y * F_SIZE: (y + 1) * F_SIZE], x, y))
        p.close()
        p.join()

    for x in range(int(SIZE / F_SIZE)):
        for y in range(int(SIZE / F_SIZE)):
            try:
                img2[x * F_SIZE: (x + 1) * F_SIZE, y * F_SIZE: (y + 1) * F_SIZE] = load(
                    "./pic/square_{}_{}.npy".format(x, y))
            except FileNotFoundError:
                pass

    cv2.imshow("first_img", img)
    cv2.imshow("second_img", img2)
    print(time.time() - t1)
    cv2.imshow('laplacian', laplacian)
    cv2.imshow('sobely', sobely)

    cv2.waitKey()
