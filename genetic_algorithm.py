import time
from copy import deepcopy

import numpy

from numpy import argwhere, zeros, uint8, full, arange, concatenate, empty, ndarray, array, clip, int32, int8, insert, \
    argsort
from numpy.random import exponential, randint, normal, choice
from scipy.spatial import Delaunay
from scipy.spatial import distance
import cv2

SIZE = 512
POPULATION_SIZE = 100
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

CROSSOVER_COUNT = 10
MUTATION_COUNT = 10

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


def draw_square(mesh, triangles, colors, fitness):
    square = empty([F_SIZE, F_SIZE, 3], dtype=uint8)
    for x in range(len(triangles)):
        cv2.drawContours(square, array([mesh[triangles[x]]]), -1, colors[x].tolist(), thickness=cv2.FILLED)
    return square


def crossover(ind_one, ind_two, src):
    child_one = deepcopy(ind_one)
    child_two = deepcopy(ind_two)

    crossover_mode = int(clip(int(exponential(1 / MODE_LAMBDA)), 0, 3))
    crossover_intensity = int(clip(int(exponential(1 / INTENSITY_LAMBDA) * 2), 1, 4))
    print("crossover mode is {}".format(crossover_mode))
    if crossover_mode == 0:
        exchange_colors(child_one, child_two, crossover_intensity)
    elif crossover_mode == 1:
        exchange_points(child_one, child_two, src, crossover_intensity)
    else:
        exchange_colors(child_one, child_two, crossover_intensity)
        exchange_points(child_one, child_two, src, crossover_intensity)
    child_one[3] = evaluate_fitness(child_one, src)
    child_two[3] = evaluate_fitness(child_two, src)
    print(child_one[3])
    print(child_two[3])
    return child_one, child_two


def exchange_colors(ind_one, ind_two, crossover_intensity):
    loci = choice(arange(min(len(ind_one[0]), len(ind_two))), size=crossover_intensity)
    ind_one[0][loci], ind_two[0][loci] = ind_two[0][loci], ind_one[0][loci]


def exchange_points(ind_one, ind_two, src, crossover_intensity):
    shorter, longer = (ind_one[0], ind_two[0]) if len(ind_one[0]) < len(ind_two[0]) else (ind_two[0], ind_one[0])
    shorter = concatenate([empty([len(longer) - len(shorter), 2], dtype=ndarray), shorter])
    loci = choice(len(shorter), crossover_intensity, replace=False)
    shorter[loci], longer[loci] = longer[loci], shorter[loci]
    shorter = shorter[(shorter != array([None, None]))[:, 0]]
    ind_one[0], ind_two[0] = (shorter, longer) if len(ind_one[0]) < len(ind_two[0]) else (longer, shorter)

    ind_one[0] = ndarray.astype(ind_one[0], dtype=int32)
    ind_two[0] = ndarray.astype(ind_one[0], dtype=int32)

    ind_one[1] = triangulate(ind_one[0])
    ind_one[2] = calculate_colors(ind_one[0], ind_one[1], src, zeros([3]))

    ind_two[1] = triangulate(ind_two[0])
    ind_two[2] = calculate_colors(ind_two[0], ind_two[1], src, zeros([3]))


def mutation(ind, src):
    mutated = deepcopy(ind)
    mutation_mode = int(clip(int(exponential(1 / MODE_LAMBDA) * 2), 0, 6))
    mutation_intensity = int(clip(int(exponential(1 / INTENSITY_LAMBDA) * 2), 1, 4))
    if mutation_mode == 0:
        mutate_colors(mutated, mutation_intensity)
    elif mutation_mode == 1:
        mutate_point_position(mutated, mutation_intensity, src)
    elif mutation_mode == 2:
        del_add_mutation(mutated, mutation_intensity, src)
    elif mutation_mode == 3:
        mutate_point_position(mutated, mutation_intensity, src)
        mutate_colors(mutated, mutation_intensity)
    elif mutation_mode == 4:
        del_add_mutation(mutated, mutation_intensity, src)
        mutate_colors(mutated, mutation_intensity)
    elif mutation_mode == 5:
        del_add_mutation(mutated, mutation_intensity, src)
        mutate_point_position(mutated, mutation_intensity, src)
    else:
        del_add_mutation(mutated, mutation_intensity, src)
        mutate_point_position(mutated, mutation_intensity, src)
        mutate_colors(mutated, mutation_intensity)
    mutated[3] = evaluate_fitness(mutated, src)
    return mutated

def mutate_colors(ind, mutation_intensity):
    loci = choice(len(ind[2]) - 1, mutation_intensity, False)
    ind[2][loci] = ind[2][loci] + \
                   ndarray.astype(clip((normal(0, 1, [mutation_intensity, 3]) * 2), -COLOR_DRIFT, COLOR_DRIFT),
                                  dtype=int8)
    clip(ind[2], 0, 255)


def mutate_point_position(ind, mutation_intensity, src):
    loci = choice(len(ind[0]) - 1, mutation_intensity, False)
    ind[0][loci] = ind[0][loci] + \
                   ndarray.astype(clip((normal(0, 2, [mutation_intensity, 2]) * 2), -POINT_DRIFT, POINT_DRIFT),
                                  dtype=int8)
    ind[1] = triangulate(ind[0])
    ind[2] = calculate_colors(ind[0], ind[1], src, zeros([3]))


def del_add_mutation(ind, mutation_intensity, src):
    is_del = randint(0, 2)
    if is_del == 1:
        loci = choice(len(ind[0] - 5), len(ind[0]) - mutation_intensity, False)
        ind[0] = ind[0][loci]
    else:
        ind[0] = concatenate([randint(0, F_SIZE, [mutation_intensity, 2]), ind[0]])
    ind[1] = triangulate(ind[0])
    ind[2] = calculate_colors(ind[0], ind[1], src, zeros([3]))


def approximate_square(source, mask):
    population = array(generate_population(source, mask))
    population = population[argsort(population, 3)]
    fitness = population[POPULATION_SIZE - 1][3]
    while fitness > 30:
        breeding = population[POPULATION_SIZE - CROSSOVER_COUNT, POPULATION_SIZE - 1]

        for x in range(len(breeding) - 1, step=2):
            breeding[x], breeding[x + 1] = crossover(breeding[x], breeding[x + 1], source)

        population = concatenate([breeding, population])

        mutating = choice(len(population) - 1, MUTATION_COUNT)

        for x in range(len(mutating) - 1):
            population[x] = mutation(population[x], source)

        population = population[argsort(population, 3)]
        population = population[len(population) - POPULATION_SIZE:]
        fitness = population[POPULATION_SIZE - 1][3]
    super_hero = population[POPULATION_SIZE - 1]
    return draw_square(*super_hero)


def evaluate_fitness(tgt, src):
    fit = 0
    square = draw_square(*tgt)
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
f = approximate_square(img[0:F_SIZE, 0:F_SIZE], laplacian[0:F_SIZE, 0:F_SIZE])
cv2.imshow("first_square", f)
print(time.time() - t1)
cv2.imshow('laplacian', laplacian)
cv2.imshow('sobely', sobely)

cv2.waitKey()
