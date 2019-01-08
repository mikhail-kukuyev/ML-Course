#!/usr/bin/env python3
from collections import namedtuple
import pickle
import math

import imageio
import numpy as np
import sys
import os


def read_file(fname):
    image = imageio.imread(fname)
    assert image.shape == (26, 40, 3)
    return image[:, :, 0].astype(np.int)


def load_data(folder_name):
    examples = []
    for fname in os.listdir(os.path.join(folder_name, 'cars')):
        img = read_file(os.path.join(folder_name, 'cars', fname))
        examples.append([img, -1])
    for fname in os.listdir(os.path.join(folder_name, 'faces')):
        img = read_file(os.path.join(folder_name, 'faces', fname))
        examples.append([img, 1])

    examples = np.array(examples).T
    return examples[0], examples[1]


def to_integral(img):
    integral = np.cumsum(np.cumsum(img, axis=0), axis=1)
    return np.pad(integral, (1, 1), 'constant', constant_values=(0, 0))[:-1, :-1]


class Feature:
    def __init__(self, x, y, width, height):
        self.x1 = x
        self.y1 = y
        self.x2 = x + width
        self.y2 = y + height

    @staticmethod
    def _rect_sum(itg_img, x1, y1, x2, y2):
        return itg_img[y1, x1] + itg_img[y2, x2] - itg_img[y1, x2] - itg_img[y2, x1]

    def apply(self, itg_img):
        pass

    def __str__(self):
        return f'{self.__class__.__name__}(x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2})'


class HorizontalTwoFeature(Feature):
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height)
        self.x_middle = self.x1 + (self.x2 - self.x1) // 2

    def apply(self, itg_img):
        return self._rect_sum(itg_img, self.x1, self.y1, self.x_middle, self.y2) \
               - self._rect_sum(itg_img, self.x_middle, self.y1, self.x2, self.y2)


class VerticalTwoFeature(Feature):
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height)
        self.y_middle = self.y1 + (self.y2 - self.y1) // 2

    def apply(self, itg_img):
        return self._rect_sum(itg_img, self.x1, self.y1, self.x2, self.y_middle) \
               - self._rect_sum(itg_img, self.x1, self.y_middle, self.x2, self.y2)


class HorizontalThreeFeature(Feature):
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height)
        self.x_third_1 = self.x1 + (self.x2 - self.x1) // 3
        self.x_third_2 = self.x1 + 2 * (self.x2 - self.x1) // 3

    def apply(self, itg_img):
        return self._rect_sum(itg_img, self.x1, self.y1, self.x_third_1, self.y2) \
               + self._rect_sum(itg_img, self.x_third_2, self.y1, self.x2, self.y2) \
               - self._rect_sum(itg_img, self.x_third_1, self.y1, self.x_third_2, self.y2)


class FourFeature(Feature):
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height)
        self.x_middle = self.x1 + (self.x2 - self.x1) // 2
        self.y_middle = self.y1 + (self.y2 - self.y1) // 2

    def apply(self, itg_img):
        return self._rect_sum(itg_img, self.x1, self.y1, self.x_middle, self.y_middle) \
               + self._rect_sum(itg_img, self.x_middle, self.y_middle, self.x2, self.y2) \
               - self._rect_sum(itg_img, self.x_middle, self.y1, self.x2, self.y_middle) \
               - self._rect_sum(itg_img, self.x1, self.y_middle, self.x_middle, self.y2)


class WeakClassifier:
    def __init__(self, feature, threshold=0., polarity=1, alpha=0.):
        self.feature = feature
        self.threshold = threshold
        self.polarity = polarity
        self.alpha = alpha

    def apply(self, x):
        return -self.polarity if self.threshold < self.feature.apply(x) else self.polarity

    def __str__(self):
        return f'feature={self.feature}, threshold={self.threshold}, polarity={self.polarity}, alpha={self.alpha}'


ValueRange = namedtuple('ValueRange', ['min', 'max'])
ImageArea = namedtuple('ImageArea', ['left', 'right', 'bottom', 'top'])


def compute_positions(width, height, img_area):
    return ((x, y)
            for x in range(img_area.left, img_area.right - width + 1, 2)
            for y in range(img_area.bottom, img_area.top - height + 1, 2))


def compute_sizes(base_width, base_height, width_range, height_range):
    return ((width, height)
            for width in range(base_width * width_range.min, width_range.max + 1, base_width)
            for height in range(base_height * height_range.min, height_range.max + 1, base_height))


def generate_features(img_area, width_range, height_range):
    horiz_2_feature = [HorizontalTwoFeature(x, y, width, height)
                       for width, height in compute_sizes(2, 1, width_range, height_range)
                       for x, y in compute_positions(width, height, img_area)]

    vert_2_feature = [VerticalTwoFeature(x, y, width, height)
                      for width, height in compute_sizes(1, 2, width_range, height_range)
                      for x, y in compute_positions(width, height, img_area)]

    horiz_3_feature = [HorizontalThreeFeature(x, y, width, height)
                       for width, height in compute_sizes(3, 1, width_range, height_range)
                       for x, y in compute_positions(width, height, img_area)]

    four_feature = [FourFeature(x, y, width, height)
                    for width, height in compute_sizes(2, 2, width_range, height_range)
                    for x, y in compute_positions(width, height, img_area)]

    return horiz_2_feature + vert_2_feature + horiz_3_feature + four_feature


def compute_threshold(ys, weights, values, indexes):
    negatives, positives = np.empty_like(ys), np.empty_like(ys)
    cur_negatives, cur_positives = 0., 0.

    for i in indexes:
        if ys[i] < 0:
            cur_negatives += weights[i]
        else:
            cur_positives += weights[i]
        negatives[i] = cur_negatives
        positives[i] = cur_positives

    total_negatives = negatives[indexes[-1]]
    total_positives = positives[indexes[-1]]

    errors = np.vstack((total_negatives - negatives + positives, total_positives - positives + negatives))
    error_type, index_min = np.unravel_index(np.argmin(errors), errors.shape)
    threshold = values[index_min]
    polarity = -1 if error_type == 0 else 1

    return threshold, polarity


def create_classifier(feature, xs, ys, weights):
    values = np.array([feature.apply(x) for x in xs])
    indexes = np.argsort(values)

    threshold, polarity = compute_threshold(ys, weights, values, indexes)
    classifier = WeakClassifier(feature, threshold, polarity)

    error = 0.
    for i in range(len(ys)):
        res = classifier.apply(xs[i])
        error += weights[i] * int(res != ys[i])

    return classifier, error


def ada_boost(features_count, xs, ys, features):
    weights = np.full(ys.shape, 1. / len(ys))
    weak_classifiers = []

    for t in range(features_count):
        best_classifier = None
        min_e = float('inf')

        for i in range(len(features)):
            cl, e = create_classifier(features[i], xs, ys, weights)
            if e < min_e:
                best_classifier = cl
                min_e = e

        best_classifier.alpha = math.log((1. - min_e) / min_e)
        z = 2 * math.sqrt(min_e * (1. - min_e))

        for i in range(len(ys)):
            weights[i] *= math.pow(math.e, - best_classifier.alpha * ys[i] * best_classifier.apply(xs[i])) / z

        weak_classifiers.append(best_classifier)
        # print(f'feature {t}, best classifier: {str(best_classifier)}')

    return weak_classifiers


def _main():
    folder_name = sys.argv[-2]
    model_filename = sys.argv[-1]

    xs, ys = load_data(folder_name)
    xs = np.array([to_integral(x) for x in xs])

    features = generate_features(ImageArea(left=6, right=34, bottom=4, top=22),
                                 width_range=ValueRange(4, 18), height_range=ValueRange(4, 12))
    # print(f'features count: {len(features)}')

    chosen_classifiers = ada_boost(10, xs, ys, features)
    pickle.dump(chosen_classifiers, open(model_filename, 'wb'))


if __name__ == '__main__':
    _main()
