#!/usr/bin/env python3

import argparse
import logging
import os
import time

import numpy as np
import sklearn.datasets
import sklearn.linear_model

import features

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

    solved_cases = 0
    minimal_result = 1.
    average_result = 0.
    start_time = time.process_time()
    for da in range(10):
        for db in range(da + 1, 10):
            logging.info('Processing case: {} vs {}'.format(da, db))
            X, Y = _filter_data(mnist.data, mnist.target, [da, db])

            logging.info('Computing features')
            fs = features.FEATURES[(da, db)]
            assert len(fs) == 2, "We want exactly two feature functions"
            X2 = [(fs[0](x), fs[1](x)) for x in X]

            logging.info('Training logistic regression classifier')
            cls = sklearn.linear_model.LogisticRegression()
            cls.fit(X2, Y)
            logging.info('Done training')

            result = cls.score(X2, Y)
            logging.info('Case {} vs {}: {:.1f}%'.format(da, db, result * 100))
            if result >= THRESHOLD:
                logging.info('Case is solved')
                solved_cases += 1
            else:
                logging.warning('Case is not solved!')
            minimal_result = min(minimal_result, result)
            average_result += result
    elapsed_time = time.process_time() - start_time

    average_result /= 45
    print('Solved cases: {}'.format(solved_cases))
    print('Minimal result: {:.1f}%'.format(minimal_result * 100))
    print('Average result: {:.1f}%'.format(average_result * 100))
    print('Elapsed time: {:.1f} second(s)'.format(elapsed_time))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    _main(_parse_args())
