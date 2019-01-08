#!/usr/bin/env python3
import pickle
import sys

import learner
from learner import (WeakClassifier, Feature, HorizontalTwoFeature, VerticalTwoFeature, HorizontalThreeFeature,
                     FourFeature)


def run():
    model = sys.argv[-2]
    fname = sys.argv[-1]

    img = learner.read_file(fname)
    x = learner.to_integral(img)

    classifiers = pickle.load(open(model, 'rb'))

    res = 0.
    for c in classifiers:
        res += c.alpha * c.apply(x)
    print(int(res >= 0))


if __name__ == '__main__':
    run()
