#!/usr/bin/env python3
import random
from collections import namedtuple, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

classes = ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica')
TRAIN_FILENAME = 'train.csv'
TEST_FILENAME = 'test.csv'


def standardize(data):
    features = data[:, :-1].astype(float)
    features = (features - features.mean(axis=0)) / features.std(axis=0)
    data[:, :-1] = features
    return data


def load_data(filename):
    return pd.read_csv(filename, header=None).values


def split_data(data_filename, train_filename, test_filename, split_ratio=0.1):
    data = load_data(data_filename)
    data = standardize(data)
    np.random.shuffle(data)
    split_index = int(len(data) * split_ratio)
    pd.DataFrame(data[:split_index]).to_csv(test_filename, header=None)
    pd.DataFrame(data[split_index:]).to_csv(train_filename, header=None)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_gradient(x, y, w):
    grad = np.dot(x.T, sigmoid(x.dot(w)) - y)
    return grad if len(x.shape) == 1 else grad / len(x)


Parameters = namedtuple('Parameters', ['iterations', 'learning_rate', 'alpha', 'gamma', 'beta'])
ValidationResult = namedtuple('ValidationResult', ['parameters', 'accuracy'])


def gradient_descent(xs, ys, gd_type, parameters):
    n_iter, learning_rate, alpha = parameters.iterations, parameters.learning_rate, parameters.alpha
    gamma, beta = parameters.gamma, parameters.beta
    eps = 1.0e-8

    ws = np.zeros((n_iter, xs.shape[1]))
    avg_w, u, g, m, v = (np.zeros_like(ws[0]) for _ in range(5))

    x, y = xs, ys
    for t in range(n_iter - 1):
        if gd_type in ['stochastic', 'sgd+momentum', 'sgd+nesterov_momentum']:
            index = random.randint(0, len(xs) - 1)
            x, y = xs[index], ys[index]

        w = ws[t] + gamma * u if gd_type == 'nesterov_momentum' else ws[t]
        gradient = compute_gradient(x, y, w) + alpha * w

        if gd_type in ['stochastic', 'batch']:
            u = - learning_rate * gradient
        elif gd_type in ['sgd+momentum', 'sgd+nesterov_momentum']:
            u = gamma * u - learning_rate * gradient
        elif gd_type == 'adagrad':
            g += gradient ** 2
            u = - learning_rate * gradient / np.sqrt(g + eps)
        elif gd_type == 'rmsprop':
            g = beta * g + (1 - beta) * gradient ** 2
            u = - learning_rate * gradient / np.sqrt(g + eps)
        elif gd_type == 'adam':
            m = gamma * m + (1 - gamma) * gradient
            v = beta * v + (1 - beta) * gradient ** 2
            u = - learning_rate * m / np.sqrt(v + eps)
        else:
            raise ("Undefined gradient descent type.")

        ws[t + 1] = ws[t] + u
    return ws


def compute_accuracy(w, sample):
    correct_count = 0
    for x, y in zip(sample[:, :-1], sample[:, -1]):
        y_predict = (sigmoid(x.dot(w)) >= 0.5)
        correct_count += (y_predict == y)
    return correct_count / len(sample)


def kfold(train, k, parameters, gd_type):
    np.random.shuffle(train)
    parts = np.array_split(train, k)
    avg_accuracy = 0.0
    for i in range(k):
        validation_sample = parts[i]
        train_sample = np.concatenate(np.delete(parts, i))

        xs, ys = train_sample[:, :-1], train_sample[:, -1]
        ws = gradient_descent(xs, ys, gd_type, parameters)
        avg_accuracy += compute_accuracy(ws.mean(axis=0), validation_sample)
    avg_accuracy /= k
    return avg_accuracy


def transform_sample(classname, sample):
    bias = np.ones((sample.shape[0], 1))
    sample = np.hstack((bias, sample))
    sample[:, -1] = (sample[:, -1] == classname)
    return sample.astype(float)


def search_parameters(classname, gd_type, iterations_grid=(1000,), lr_grid=(0.01,), alpha_grid=(0.0001,),
                      gamma_grid=(None,), beta_grid=(None,)):
    train = transform_sample(classname, load_data(TRAIN_FILENAME))
    best_result = ValidationResult(Parameters(*([None] * 5)), accuracy=0.0)

    grids = [iterations_grid, lr_grid, alpha_grid, gamma_grid, beta_grid]
    for param_list in itertools.product(*grids):
        parameters = Parameters(*param_list)
        accuracy = kfold(train, 10, parameters, gd_type)
        cur_result = ValidationResult(parameters, accuracy)
        if accuracy > best_result.accuracy:
            best_result = cur_result
    return best_result


def test(classname, gd_type, parameters):
    train = transform_sample(classname, load_data(TRAIN_FILENAME))
    test = transform_sample(classname, load_data(TEST_FILENAME))

    xs, ys = train[:, :-1], train[:, -1]
    ws = gradient_descent(xs, ys, gd_type, parameters)
    return compute_accuracy(ws.mean(axis=0), test)


def compare(classname, parameters_dict):
    train = transform_sample(classname, load_data(TRAIN_FILENAME))
    xs, ys = train[:, :-1], train[:, -1]

    max_iterations = max(map(lambda ps: ps.iterations, parameters_dict.values()))
    for gd_type in parameters_dict:
        parameters_list = list(parameters_dict[gd_type])
        parameters_dict[gd_type] = Parameters(max_iterations, *(parameters_list[1:]))

    for gd_type, parameters in parameters_dict.items():
        ws = gradient_descent(xs, ys, gd_type, parameters)
        accuracy = [compute_accuracy(ws[:i + 1].mean(axis=0), train) for i in range(len(ws))]
        plt.plot(range(parameters.iterations), accuracy)

    plt.legend(parameters_dict.keys(), loc='lower right')
    plt.title(f"Accuracy for '{classname}' model")
    plt.show()


def compare_base_methods():
    gd_types = ['stochastic', 'batch']
    for classname in classes:
        parameters_dict = defaultdict()
        for gd_type in gd_types:
            best_result = search_parameters(classname, gd_type, iterations_grid=[200, 800, 1600],
                                            lr_grid=[0.1, 0.05, 0.01, 0.005, 0.001],
                                            alpha_grid=[0.01, 0.001, 0.0001, 0.0])
            print(f"Best parameters for '{classname}' with '{gd_type}': {best_result.parameters}")
            print(f"Best validation accuracy for '{classname}' with '{gd_type}: {best_result.accuracy}")
            parameters_dict[gd_type] = best_result.parameters

            accuracy = test(classname, gd_type, best_result.parameters)
            print(f"Accuracy on test sample for '{classname}' with '{gd_type}: {accuracy}\n")

        compare(classname, parameters_dict)


def compare_adaptive_methods():
    gd_types = ['sgd+momentum', 'sgd+nesterov_momentum', 'adagrad', 'rmsprop', 'adam']
    for classname in classes:
        parameters_dict = defaultdict()
        for gd_type in gd_types:
            lr_grid = [0.1] if gd_type in ['adagrad', 'rmsprop', 'adam'] else [0.1, 0.05, 0.01, 0.005, 0.001]
            gamma_grid = [0.9, 0.98, 0.999] if gd_type in ['sgd+momentum', 'sgd+nesterov_momentum', 'adam'] else [None]
            beta_grid = [0.9, 0.98, 0.999] if gd_type in ['rmsprop', 'adam'] else [None]

            best_result = search_parameters(classname, gd_type, iterations_grid=[200, 800, 1600],
                                            lr_grid=lr_grid, alpha_grid=[0.01, 0.001, 0.0001, 0.0],
                                            gamma_grid=gamma_grid, beta_grid=beta_grid)

            print(f"Best parameters for '{classname}' with '{gd_type}': {best_result.parameters}")
            print(f"Best validation accuracy for '{classname}' with '{gd_type}: {best_result.accuracy}")
            parameters_dict[gd_type] = best_result.parameters

            accuracy = test(classname, gd_type, best_result.parameters)
            print(f"Accuracy on test sample for '{classname}' with '{gd_type}: {accuracy}\n")

        compare(classname, parameters_dict)


def main():
    split_data('iris_data.csv', TRAIN_FILENAME, TEST_FILENAME)
    compare_base_methods()
    compare_adaptive_methods()


if __name__ == "__main__":
    main()
