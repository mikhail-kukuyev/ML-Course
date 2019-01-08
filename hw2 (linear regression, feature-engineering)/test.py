#!/usr/bin/env python3

import argparse
import logging
import os
import time
from queue import PriorityQueue

from matplotlib import pyplot as plt
import numpy as np
import sklearn.datasets
import sklearn.linear_model

import features as f

FEATURES = [f.sum_feature, f.mean_feature,
            f.vertical_symmetry_feature, f.horizontal_symmetry_feature,
            f.vertical_overturn_symmetry_feature, f.horizontal_overturn_symmetry_feature,
            f.vertical_center_feature, f.horizontal_center_feature,
            f.uppper_part_feature, f.down_part_feature, f.left_part_feature, f.right_part_feature]

THRESHOLD = 0.75


def _parse_args():
    parser = argparse.ArgumentParser(prog='bsu 2018 / ml / hw 2')
    parser.add_argument('--datadir', help='path to folder to cache data', default=os.getcwd())
    return parser.parse_args()


def _filter_data(x, y, digits):
    """Create subset with only specified digits."""
    rx, ry = [], []
    for cx, cy in zip(x, y):
        if cy in digits:
            rx.append(cx)
            ry.append(digits.index(cy))
    return np.array(rx), np.array(ry)


def _main(args):
    sklearn_home = args.datadir

    logging.info('Downloading MNIST data')
    mnist = sklearn.datasets.fetch_mldata('MNIST original', data_home=sklearn_home)
    logging.info('Data is ready')

    best_answer = '\n\n'

    for da in range(10):
        for db in range(da + 1, 10):
            print('Case: {} vs {}'.format(da, db))
            X, Y = _filter_data(mnist.data, mnist.target, [da, db])

            records = PriorityQueue()
            for di in range(len(FEATURES)):
                for dj in range(di + 1, len(FEATURES)):
                    X2 = [(FEATURES[di](x), FEATURES[dj](x)) for x in X]

                    cls = sklearn.linear_model.LogisticRegression()
                    cls.fit(X2, Y)
                    result = cls.score(X2, Y)
                    records.put((-result, (FEATURES[di].__name__, FEATURES[dj].__name__)))

            for i in range(10):
                (res, (f1, f2)) = records.get()
                print('\tfeatures: {}, {}: {:.1f}%'.format(f1, f2, - res * 100))
                if i == 0:
                    best_answer += '({}, {}): ({}, {}),\n'.format(da, db, f1, f2)


    print(best_answer)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    _main(_parse_args())
