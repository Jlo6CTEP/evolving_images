import time
from copy import deepcopy

import numpy

from numpy import argwhere, zeros, uint8, full, arange, concatenate, empty, ndarray, array, clip, int32, int8
from numpy.random import exponential, randint, normal, choice
from scipy.spatial import Delaunay
from scipy.spatial import distance
import cv2

SIZE = 512
POPULATION_SIZE = 200
F_SIZE = 64

KERNEL_SIZE = 3
THRESHOLD = 8
MAX_BLACK = 255

EDGE_COUNT_PENALTY = 8

MAX_POINT_FACTOR = 1 / 8
MIN_POINT_COUNT = 4
COLOR_DRIFT = 5
POINT_DRIFT = F_SIZE / 16
RANDOM_POINTS = 3
FITNESS_STEP = 2

MODE_LAMBDA = 0.8
INTENSITY_LAMBDA = 1.5

BORDER_POINT_FACTOR = 1

PICTURE = "./picture/Tower2.jpg"


def generate_population(source, mask):
    points = argwhere(mask == MAX_BLACK)

    inner_count = round(min(max(len(points) * MAX_POINT_FACTOR, MIN_POINT_COUNT), len(points)))
    border_count = inner_count * BORDER_POINT_FACTOR

    # points on the borders of the square
    border_points_top_left = zeros((F_SIZE * 2, 2), dtype=uint8)
    border_points_bottom_right = full((F_SIZE * 2, 2), F_SIZE - 1, dtype=uint8)
    border_points_top_left[0:F_SIZE, 1] = arange(0, F_SIZE)
    border_points_top_left[F_SIZE:2 * F_SIZE, 0] = arange(0, F_SIZE)
    border_points_bottom_right[0:1 * F_SIZE, 1] = arange(0, F_SIZE)
    border_points_bottom_right[F_SIZE:2 * F_SIZE, 0] = arange(0, F_SIZE)
    border_points = concatenate((border_points_top_left, border_points_bottom_right))

    population = empty(POPULATION_SIZE, ndarray)

    for n in range(POPULATION_SIZE):
        inner_count += randint(-RANDOM_POINTS, RANDOM_POINTS)
        inner_count = max(0, inner_count)
        if inner_count == 0:
            inner = array([[int(F_SIZE / 2), int(F_SIZE / 2)]])
        elif len(points) < inner_count:
            random_points = empty([inner_count - len(points), 2], dtype=uint8)
            for x in range(inner_count - len(points)):
                random_points[x][0] = randint(0, F_SIZE)
                random_points[x][1] = randint(0, F_SIZE)
            inner = concatenate([points, random_points])
        else:
            inner = points[choice(points.shape[0], inner_count, replace=False), :]
        borders = border_points[choice(border_points.shape[0], border_count, replace=False), :]
        borders = concatenate(
            (borders, array([[0, 0], [F_SIZE - 1, F_SIZE - 1], [F_SIZE - 1, 0], [0, F_SIZE - 1]])))

        mesh = concatenate((inner, borders))

        tri = triangulate(mesh)

        individual = empty([4], ndarray)
        individual[0] = mesh

        colors = calculate_colors(mesh, tri, source, randint(low=-COLOR_DRIFT, high=COLOR_DRIFT, size=3))

        individual[1] = tri
        individual[2] = colors
        individual[3] = evaluate_fitness(individual, source)
        population[n] = individual
    return population


def triangulate(mesh):
    return Delaunay(mesh).simplices


def calculate_colors(mesh, triangles, src, drift):
    colors = empty([triangles.shape[0], 3], dtype=uint8)
    for x in range(triangles.shape[0]):
        cm = array(numpy.sum(mesh[triangles[x]], axis=0) / 3, dtype=int32)
        colors[x] = src[cm[1], cm[0]].tolist() + drift
    return colors


def draw_square(mesh, triangles, colors):
    square = empty([F_SIZE, F_SIZE, 3], dtype=uint8)
    for x in range(len(triangles)):
        cv2.drawContours(square, array([mesh[triangles[x]]]), -1, colors[x].tolist(), thickness=cv2.FILLED)
    return square


def crossover(ind_one, ind_two):
    child_one = deepcopy(ind_one)
    child_two = deepcopy(ind_two)


def mutation(ind, src):
    mutation_mode = int(clip(int(exponential(1 / MODE_LAMBDA) * 2), 0, 6))
    mutation_intensity = int(clip(int(exponential(1 / INTENSITY_LAMBDA) * 2), 1, 4))
    if mutation_mode == 0:
        mutate_colors(ind, mutation_intensity)
    elif mutation_mode == 1:
        mutate_point_position(ind, mutation_intensity, src)
    elif mutation_mode == 2:
        del_add_mutation(ind, mutation_intensity, src)
    elif mutation_mode == 3:
        mutate_point_position(ind, mutation_intensity, src)
        mutate_colors(ind, mutation_intensity)
    elif mutation_mode == 4:
        del_add_mutation(ind, mutation_intensity, src)
        mutate_colors(ind, mutation_intensity)
    elif mutation_mode == 5:
        del_add_mutation(ind, mutation_intensity, src)
        mutate_point_position(ind, mutation_intensity, src)
    else:
        del_add_mutation(ind, mutation_intensity, src)
        mutate_point_position(ind, mutation_intensity, src)
        mutate_colors(ind, mutation_intensity)
    ind[3] = evaluate_fitness(ind, src)
    print(evaluate_fitness(ind, src))
    return ind


def mutate_colors(ind, mutation_intensity):
    rand_index = choice(len(ind[2]) - 1, mutation_intensity, False)
    ind[2][rand_index] = ind[2][rand_index] + \
        ndarray.astype(clip((normal(0, 1, [mutation_intensity, 3]) * 2), -COLOR_DRIFT, COLOR_DRIFT), dtype=int8)
    clip(ind[2], 0, 255)


def mutate_point_position(ind, mutation_intensity, src):
    rand_index = choice(len(ind[0]) - 1, mutation_intensity, False)
    ind[0][rand_index] = ind[0][rand_index] + \
        ndarray.astype(clip((normal(0, 2, [mutation_intensity, 2]) * 2), -POINT_DRIFT, POINT_DRIFT), dtype=int8)
    ind[1] = triangulate(ind[0])
    ind[2] = calculate_colors(ind[0], ind[1], src, zeros([3]))


def del_add_mutation(ind, mutation_intensity, src):
    is_del = randint(0, 2)
    if is_del == 1:
        rand_index = choice(len(ind[0] - 5), len(ind[0]) - mutation_intensity, False)
        ind[0] = ind[0][rand_index]
    else:
        ind[0] = concatenate([randint(0, F_SIZE, [mutation_intensity, 2]), ind[0]])
    ind[1] = triangulate(ind[0])
    ind[2] = calculate_colors(ind[0], ind[1], src, zeros([3]))


def approximate_square(source, mask):
    population = array(generate_population(source, mask))
    print(evaluate_fitness(population[3], source))
    mutation(population[3], source)
    mutation(population[4], source)
    mutation(population[5], source)
    mutation(population[6], source)
    mutation(population[13], source)
    mutation(population[14], source)
    mutation(population[15], source)
    mutation(population[16], source)
    print()


def evaluate_fitness(tgt, src):
    fit = 0
    square = draw_square(tgt[0], tgt[1], tgt[2])
    for x in range(0, F_SIZE, FITNESS_STEP):
        for y in range(0, F_SIZE, FITNESS_STEP):
            fit += abs(distance.euclidean(square[x, y], src[x, y]))
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
