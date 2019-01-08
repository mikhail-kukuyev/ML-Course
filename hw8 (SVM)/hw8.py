#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple
from subprocess import call
from libsvm.python.svmutil import *
from liblinear_.python.liblinearutil import *


def save_libsvm(raw_data, filename):
    x = raw_data[:, :-1]
    y = raw_data[:, -1].astype(int)
    lines = []
    for features, label in zip(x, y):
        sparse_features = [f"{i + 1}:{value}" for i, value in enumerate(features) if value != 0]
        line = ' '.join([str(label)] + sparse_features) + '\n'
        lines.append(line)
    with open(filename, 'w') as f:
        f.writelines(lines)


def preprocess_spam_data(input_file, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    data = pd.read_csv(input_file, header=None).values
    train, test = np.vsplit(data, [3450])

    save_libsvm(train, os.path.join(save_dir, 'train'))
    save_libsvm(test, os.path.join(save_dir, 'test'))

    call('./spam_scale.sh')


def split(index, arr, part_size):
    validation_part = arr[index: (index + 1) * part_size]
    train_part = arr[:index * part_size] + arr[(index + 1) * part_size:]
    return train_part, validation_part


def kfold(x, y, parameters, train_func, predict_func, k=10):
    part_size = len(y) // k
    errors = []
    for i in range(k):
        x_train, x_valid = split(i, x, part_size)
        y_train, y_valid = split(i, y, part_size)

        model = train_func(y_train, x_train, parameters)
        _, results, _ = predict_func(y_valid, x_valid, model, '-q')
        errors.append(1 - results[0] / 100)
    return np.mean(errors), np.std(errors)


ValidationResult = namedtuple('ValidationResult', ['degree', 'cost', 'error'])


def build_options(degree=2, cost=1):
    return f"-t 1 -r 1 -d {degree} -c {cost} -h 0 -q"


def plot_errors_stdevs(errors_stdevs, log_costs, d):
    errors, stdevs = zip(*errors_stdevs)
    errors, stdevs = np.array(errors), np.array(stdevs)
    plt.plot(log_costs, errors - stdevs)
    plt.plot(log_costs, errors)
    plt.plot(log_costs, errors + stdevs)
    plt.xlabel("log2(C)")
    plt.title(f"Error for degree={d}")
    plt.legend(['error - stdev', 'error', 'error + stdev'], loc='upper right')
    plt.show()


def search_parameters(x, y):
    best_result = ValidationResult(None, None, error=1.0)
    k = 10
    log_costs = list(range(-k, k + 1))
    degrees = range(4, 5)
    errors_stdevs = [[] for _ in degrees]

    for i, degree in enumerate(degrees):
        for cost in map(lambda a: 2 ** a, log_costs):
            error, std = kfold(x, y, build_options(degree, cost), svm_train, svm_predict)
            if error < best_result.error:
                best_result = ValidationResult(degree, cost, error)
            errors_stdevs[i].append((error, std))
        plot_errors_stdevs(errors_stdevs[i], log_costs, degree)

    for errs_stds_i in errors_stdevs:
        errors_i, _ = zip(*errs_stds_i)
        plt.plot(log_costs, errors_i)
    plt.xlabel("log2(C)")
    plt.title(f"Degrees comparison")
    plt.legend([f"d={d}" for d in degrees], loc='upper right')
    plt.show()

    return best_result


def error_on_degrees(x_train, y_train, x_test, y_test, cost):
    valid_errors = []
    test_errors = []
    degrees = list(range(1, 31))
    for degree in degrees:
        valid_error, _ = kfold(x_train, y_train, build_options(degree, cost), svm_train, svm_predict)
        valid_errors.append(valid_error)

        model = svm_train(y_train, x_train, build_options(degree, cost))
        _, results, _ = svm_predict(y_test, x_test, model, '-q')
        test_errors.append(1 - results[0] / 100)

    plt.plot(degrees, valid_errors)
    plt.plot(degrees, test_errors)
    plt.xlabel('polynom degree')
    plt.title(f"Error for C={cost}")
    plt.legend(['validation', 'test'], loc='upper right')
    plt.show()


def sv_amount_on_degrees(x_train, y_train, best_degree, best_cost):
    sv_amount = []
    degrees = list(range(1, 51))
    for degree in degrees:
        model = svm_train(y_train, x_train, build_options(degree, best_cost))
        sv_amount.append(model.get_nr_sv())
        if degree == best_degree:
            print(f"Support vectors amount for d={best_degree}, C={best_cost}: {model.get_nr_sv()}")
    plt.plot(degrees, sv_amount)
    plt.title("Support vectors amount")
    plt.show()


def task1():
    save_dir = 'spam_tmp'
    preprocess_spam_data('spambase', save_dir)

    y_train, x_train = svm_read_problem(os.path.join(save_dir, 'train_scaled'))

    best = search_parameters(x_train, y_train)
    print(best)

    y_test, x_test = svm_read_problem(os.path.join(save_dir, 'test_scaled'))
    error_on_degrees(x_train, y_train, x_test, y_test, best.cost)

    sv_amount_on_degrees(x_train, y_train, best.degree, best.cost)


def transform_gisette_data(data_dir, save_dir, base_name):
    os.makedirs(save_dir, exist_ok=True)
    data_path = os.path.join(data_dir, base_name)
    x = pd.read_csv(data_path + '.data', delim_whitespace=True, header=None).values
    y = pd.read_csv(data_path + '.labels', delim_whitespace=True, header=None).values
    save_libsvm(np.hstack((x, y)), os.path.join(save_dir, base_name))


def preprocess_gisette(save_dir):
    data_dir = 'gisette'
    transform_gisette_data(data_dir, save_dir, 'gisette_train')
    transform_gisette_data(data_dir, save_dir, 'gisette_valid')
    call('./gisette_scale.sh')


GisetteResult = namedtuple('GisetteResult', ['cost', 'error'])


def liblinear_options(svm_type=1, cost=1.0, eps=0.0001):
    return f"-s {svm_type} -c {cost} -e {eps} -q"


def plot_gisette_errors_stdevs(errors_stdevs, log_costs):
    errors, stdevs = zip(*errors_stdevs)
    errors, stdevs = np.array(errors), np.array(stdevs)
    plt.plot(log_costs, errors - stdevs)
    plt.plot(log_costs, errors)
    plt.plot(log_costs, errors + stdevs)
    plt.xlabel("log2(C)")
    plt.xticks(log_costs)
    plt.legend(['error - stdev', 'error', 'error + stdev'], loc='upper right')
    plt.show()


def search_gisette_parameters(x, y):
    best_result = GisetteResult(None, error=1.0)
    log_costs = list(range(-15, 3))
    errors_stdevs = []

    for cost in map(lambda a: 2 ** a, log_costs):
        error, std = kfold(x, y, liblinear_options(cost=cost), train, predict, k=5)
        if error < best_result.error:
            best_result = GisetteResult(cost, error)
        errors_stdevs.append((error, std))

    plot_gisette_errors_stdevs(errors_stdevs, log_costs)
    return best_result


def task2():
    save_dir = 'gisette_tmp'
    preprocess_gisette(save_dir)

    y_train, x_train = svm_read_problem(os.path.join(save_dir, 'train_scaled'))

    best = search_gisette_parameters(x_train, y_train)
    print(best)
    print(f"Validation accuracy: {100 * (1 - best.error)}%")

    model = train(y_train, x_train, liblinear_options(svm_type=1, cost=best.cost, eps=0.001))
    y_valid, x_valid = svm_read_problem(os.path.join(save_dir, 'valid_scaled'))
    _, results, _ = predict(y_valid, x_valid, model, '-q')
    print(f"Dev set error: {1 - results[0] / 100}")
    print(f"Dev set accuracy: {results[0]}%")


def main():
    task1()
    # task2()


if __name__ == "__main__":
    main()
