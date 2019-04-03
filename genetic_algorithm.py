import time
from scipy.spatial import Delaunay
from scipy.spatial import distance
import numpy as np
import cv2

SIZE = 512
POPULATION_SIZE = 200

KERNEL_SIZE = 3
THRESHOLD = 8
MAX_BLACK = 255

EDGE_COUNT_PENALTY = 8

MAX_POINT_FACTOR = 1 / 8
MIN_POINT_COUNT = 4
COLOR_DELTA = 10
RANDOM_POINTS = 3
FITNESS_STEP = 2

BORDER_POINT_FACTOR = 1

F_SIZE = 64

PICTURE = "./picture/Tower2.jpg"


def generate_population(source, mask):
    points = np.argwhere(mask == MAX_BLACK)

    inner_count = round(min(max(len(points) * MAX_POINT_FACTOR, MIN_POINT_COUNT), len(points)))
    border_count = inner_count * BORDER_POINT_FACTOR

    # points on the borders of the square
    border_points_top_left = np.zeros((F_SIZE * 2, 2), dtype=np.uint8)
    border_points_bottom_right = np.full((F_SIZE * 2, 2), F_SIZE - 1, dtype=np.uint8)
    border_points_top_left[0:F_SIZE, 1] = np.arange(0, F_SIZE)
    border_points_top_left[F_SIZE:2 * F_SIZE, 0] = np.arange(0, F_SIZE)
    border_points_bottom_right[0:1 * F_SIZE, 1] = np.arange(0, F_SIZE)
    border_points_bottom_right[F_SIZE:2 * F_SIZE, 0] = np.arange(0, F_SIZE)
    border_points = np.concatenate((border_points_top_left, border_points_bottom_right))

    population = np.empty(POPULATION_SIZE, np.ndarray)

    for n in range(POPULATION_SIZE):
        inner_count += np.random.randint(-RANDOM_POINTS, RANDOM_POINTS)
        inner_count = max(0, inner_count)
        if inner_count == 0:
            inner = np.array([[int(F_SIZE / 2), int(F_SIZE / 2)]])
        elif len(points) < inner_count:
            random_points = np.empty([inner_count - len(points), 2], dtype=np.uint8)
            for x in range(inner_count - len(points)):
                random_points[x][0] = np.random.randint(0, F_SIZE - 1)
                random_points[x][1] = np.random.randint(0, F_SIZE - 1)
            inner = np.concatenate([points, random_points])
        else:
            inner = points[np.random.choice(points.shape[0], inner_count, replace=False), :]
        borders = border_points[np.random.choice(border_points.shape[0], border_count, replace=False), :]
        borders = np.concatenate(
            (borders, np.array([[0, 0], [F_SIZE - 1, F_SIZE - 1], [F_SIZE - 1, 0], [0, F_SIZE - 1]])))

        mesh = np.concatenate((inner, borders))

        tri = Delaunay(mesh).simplices

        individual = np.empty([4], np.ndarray)
        individual[0] = mesh

        square = np.empty([F_SIZE, F_SIZE, 3], dtype=np.uint8)
        colors = np.empty([tri.shape[0], 3], dtype=np.uint8)
        for i in range(tri.shape[0]):
            cm = np.array(np.sum(mesh[tri[i]], axis=0) / 3, dtype=np.int32)
            color = source[cm[1], cm[0]].tolist() + np.random.randint(low=-COLOR_DELTA, high=COLOR_DELTA, size=(1, 3))
            color = np.clip(color, 0, 255)[0].tolist()
            colors[i] = color
        individual[1] = tri
        individual[2] = colors
        individual[3] = evaluate_fitness(individual, source)
        population[n] = individual
    return population


def crossover(ind_one, ind_two):
    pass


def mutation(ind):
    pass


def approximate_square(source, mask):
    population = np.array(generate_population(source, mask))
    print()


def evaluate_fitness(tgt, ref):
    fit = 0
    square = np.empty([F_SIZE, F_SIZE, 3], dtype=np.uint8)
    for x in range(len(tgt[1])):
        arr = tgt[0][tgt[1][x]]
        cv2.drawContours(square, [arr], -1, tgt[2][x].tolist(), thickness=cv2.FILLED)
    for x in range(0, F_SIZE, FITNESS_STEP):
        for y in range(0, F_SIZE, FITNESS_STEP):
            fit += abs(distance.euclidean(square[x, y], ref[x, y]))
    return fit / (F_SIZE ** 2)


img = cv2.resize(cv2.imread(PICTURE), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
# gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (KERNEL_SIZE, KERNEL_SIZE), 0)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, laplacian = cv2.threshold(cv2.Laplacian(gray, cv2.CV_8U), THRESHOLD, MAX_BLACK, cv2.THRESH_BINARY)
sobely = cv2.Sobel(gray, cv2.CV_8U, 1, 1, ksize=5)

cv2.imshow('init', img)

t1 = time.time()
approximate_square(img[0:F_SIZE, 0:F_SIZE], laplacian[0:F_SIZE, 0:F_SIZE])
print(time.time() - t1)
cv2.imshow('laplacian', laplacian)
cv2.imshow('sobely', sobely)

cv2.waitKey()
