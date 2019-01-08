#!/usr/bin/env python
import random

import numpy as np
from matplotlib import pyplot


def generate_sample(m, rect_size):
    return [(random.uniform(0, rect_size), random.uniform(0, rect_size)) for _ in range(m)]


def true_risk(sample, rectX_size, rectQ_size):
    q_points = [point for point in sample if point[0] <= rectQ_size and point[1] <= rectQ_size]
    rect_area = 0
    if len(q_points) > 1:
        x = list(map(lambda point: point[0], q_points))
        y = list(map(lambda point: point[1], q_points))
        rect_area = (max(x) - min(x)) * (max(y) - min(y))
    return (rectQ_size ** 2 - rect_area) / rectX_size ** 2


def single_experiment(m, rectX_size, rectQ_size):
    sample = generate_sample(m, rectX_size)
    return true_risk(sample, rectX_size, rectQ_size)


rectX_size = 1.0
rectQ_size = 0.5 ** 0.5

x = range(1, 500)
y = [single_experiment(m, rectX_size, rectQ_size) for m in x]
pyplot.plot(x, y)
pyplot.show()


def calculate_min_m(m_start, e):
    m = m_start
    true_risk = 1
    while true_risk > e:
        trials = [single_experiment(m, rectX_size, rectQ_size) for _ in range(10)]
        true_risk = np.mean(trials)
        m += 1
    return m


e = [0.1, 0.01, 0.001]
m = [None] * len(e)
for i in range(len(e)):
    m_start = 1 if i == 0 else m[i - 1]
    m[i] = calculate_min_m(m_start, e[i])

for m_i, e_i in zip(m, e):
    print('m = {} for true_risk = {}'.format(m_i, e_i))
