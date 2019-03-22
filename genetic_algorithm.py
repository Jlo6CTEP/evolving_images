import time

import numpy
import cv2
import random

from numba import jit

SIZE = 1024


@jit(nopython=True)
def random_picture(dim_x, dim_y):
    pic = numpy.empty((dim_x, dim_y, 3), dtype=numpy.uint8)
    for x in range(dim_x):
        for y in range(dim_y):
            for z in range(3):
                pic[x, y, z] = random.randint(0, 255)
    return pic


img = cv2.imread("./picture/Tower1.png")
img2 = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)

f = time.time()
for x in range(1):
    cv2.imshow("Shit{}".format(x), cv2.blur(random_picture(SIZE, SIZE), (2, 2)))
print(time.time() - f)


cv2.waitKey()
