import time
from copy import deepcopy

import cv2
import numpy as np
from numba import jit
from numpy import zeros, full, ndarray, clip, int8, int16, empty, \
    uint8, ceil
from numpy.random import randint, normal, choice
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("picture", help="name of picture inside the folder ./picture/in/")

PATH_IN = "./picture/in/"
PATH_OUT = "./picture/out/"

SIZE = 512

COLOR_DRIFT = 30
POINT_DRIFT = 40

INTENSITY_LAMBDA = 0.5
INTENSITY_MULT = 4

POPULATION_SIZE = 40
RADIUS = 4

SQUARE_SIZE = int(ceil(RADIUS * 2 ** 1 / 4))

DOT_COUNT = 120000
ITERATION_COUNT = 10
MUTATION_COUNT = int(POPULATION_SIZE / 3)
CROSSOVER_COUNT = int(POPULATION_SIZE / 2)

INITIAL_COLOR = 255

temp = empty([SIZE, SIZE, 3], dtype=uint8)

picture = parser.parse_args().picture


@jit()
def generate_population(src):
    population = empty([POPULATION_SIZE, 3], dtype=ndarray)
    for x in range(POPULATION_SIZE):
        ind = empty([3], dtype=ndarray)
        ind[0] = randint(SQUARE_SIZE, SQUARE_SIZE + SIZE, 2)
        ind[1] = randint(0, 255, 3)
        ind[2] = fitness(ind, src)
        population[x] = ind
    return population


@jit()
def draw_individual(picture, individual):
    cv2.circle(picture, tuple(individual[0])[::-1], RADIUS, individual[1].tolist(), thickness=cv2.FILLED)


@jit()
def fitness(individual, src):
    test_square = full([SQUARE_SIZE, SQUARE_SIZE, 3], individual[1].astype(dtype=int16), dtype=ndarray)
    shift = int(ceil(SQUARE_SIZE / 2))
    diff = test_square - src[individual[0][0] - shift: individual[0][0] + shift,
           individual[0][1] - shift: individual[0][1] + shift]
    return np.sum(np.sum(np.abs(diff)))


@jit()
def mutate(ind, src):
    mutated = deepcopy(ind)
    mutation_mode = randint(0, 3)
    if mutation_mode == 0:
        mutate_color(mutated)
    elif mutation_mode == 1:
        mutate_point(mutated)
    else:
        mutate_color(mutated)
        mutate_point(mutated)
    mutated[2] = fitness(mutated, src)
    return mutated


@jit()
def mutate_color(ind):
    ind[1] = ind[1].astype(dtype=int16)
    ind[1] = ind[1] + \
             clip((normal(0, 1, [3]) * 8), -COLOR_DRIFT, COLOR_DRIFT).astype(dtype=int8)
    ind[1] = clip(ind[1], 0, 255)
    ind[1] = ind[1].astype(dtype=uint8)
    return ind


@jit()
def mutate_point(ind):
    ind[0] = ind[0] + \
             ndarray.astype(clip((normal(0, 2, [2]) * 8), -POINT_DRIFT, POINT_DRIFT), dtype=int16)
    ind[0] = clip(ind[0], SQUARE_SIZE, SQUARE_SIZE + SIZE)
    return ind


@jit()
def crossover(ind_one, ind_two, src):
    child_one = deepcopy(ind_one)
    child_two = deepcopy(ind_two)
    crossover_mode = randint(0, 3)
    crossover_intensity = randint(0, 2)
    if crossover_mode == 0:
        cross_colors(child_one, child_two)
    elif crossover_mode == 1:
        cross_points(child_one, child_two, crossover_intensity)
    else:
        cross_colors(child_one, child_two)
        cross_points(child_one, child_two, crossover_intensity)
    child_one[2] = fitness(child_one, src)
    child_two[2] = fitness(child_two, src)
    return child_one, child_two


@jit()
def cross_colors(ind_one, ind_two):
    ind_one[1], ind_two[1] = ind_two[1], ind_one[1]


@jit()
def cross_points(ind_one, ind_two, intensity):
    if intensity == 0:
        mode = randint(0, 2)
        if mode == 0:
            ind_one[0][0], ind_two[0][0] = ind_two[0][0], ind_one[0][0]
        else:
            ind_one[0][1], ind_two[0][1] = ind_two[0][1], ind_one[0][1]
    else:
        ind_one[0], ind_two[0] = ind_two[0], ind_one[0]


@jit()
def evaluate(src, tgt):
    for x in range(int(DOT_COUNT)):
        pop = generate_population(src)
        pop = pop[pop[:, 2].argsort()[::-1]]
        for y in range(ITERATION_COUNT):
            breeding = pop[POPULATION_SIZE - CROSSOVER_COUNT: POPULATION_SIZE]
            for f in range(0, len(breeding) - 1, 2):
                pop[f], pop[f + 1] = crossover(breeding[f], breeding[f + 1], src)
            mutating = choice(len(pop) - 1, MUTATION_COUNT)
            for f in range(len(mutating) - 1):
                pop[f] = mutate(pop[f], src)
            pop = pop[pop[:, 2].argsort()[::-1]]
        draw_individual(tgt, pop[POPULATION_SIZE - 1])
        if x % 100 == 0:
            print(x)


if __name__ == '__main__':
    img_src = zeros([SIZE + 2 * SQUARE_SIZE, SIZE + 2 * SQUARE_SIZE, 3], dtype=uint8)
    img_src[SQUARE_SIZE: SQUARE_SIZE + SIZE, SQUARE_SIZE: SQUARE_SIZE + SIZE] = \
        cv2.resize(cv2.imread(PATH_IN + picture), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)

    dst = full([SIZE + SQUARE_SIZE, SIZE + SQUARE_SIZE, 3], INITIAL_COLOR, dtype=uint8)

    t1 = time.time()

    evaluate(img_src, dst)

    cv2.imshow("source", img_src)
    cv2.imshow("result", dst)
    cv2.imwrite(PATH_OUT + "new_" + picture, dst)
    print("Work time is {}".format(time.time() - t1))
    cv2.waitKey()
