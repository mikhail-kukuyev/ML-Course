#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_mldata

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

classes = ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica')


def normalize(data):
    features = data[:, :-1].astype(float)
    features = (features - features.min(axis=0)) / features.ptp(axis=0)
    data[:, :-1] = features
    return data


def to_float(sample):
    labels_dict = dict(zip(classes, range(len(classes))))
    ys = [labels_dict[label] for label in sample[:, -1]]
    sample[:, -1] = np.array(ys)
    return sample.astype(float)


def preprocess_data(data_filename, split_ratio=0.1):
    data = pd.read_csv(data_filename, header=None).values
    data = to_float(data)
    data = normalize(data)
    np.random.shuffle(data)
    split_index = int(len(data) * split_ratio)
    return data[split_index:], data[:split_index]


def knn(train_data, test_data, k):
    xs, ys = train_data[:, :-1], train_data[:, -1]
    xs_test, ys_test = test_data[:, :-1], test_data[:, -1]
    ys_pred = np.zeros(len(ys_test))

    for i, x_test in enumerate(xs_test):
        distances = np.linalg.norm(xs - x_test, axis=1)
        neighbours_indices = np.argpartition(distances, k)[:k]
        neighbours = ys[neighbours_indices].astype(int)
        ys_pred[i] = np.bincount(neighbours).argmax()

    error = 1 - np.sum(ys_pred == ys_test) / len(ys_test)
    return error


def kfold(train_data, k_param, n_folds=10):
    np.random.shuffle(train_data)
    parts = np.array_split(train_data, n_folds)
    sum_error = 0.0
    for i in range(n_folds):
        valid_sample = parts[i]
        train_sample = np.concatenate(np.delete(parts, i))
        sum_error += knn(train_sample, valid_sample, k_param)
    return sum_error / n_folds


def search_parameters(train_data, test_data):
    k_range = list(range(1, 101))

    valid_errors = [kfold(train_data, k) for k in k_range]
    best_index = np.argmin(valid_errors)
    test_errors = [knn(train_data, test_data, k) for k in k_range]

    plt.plot(k_range, valid_errors)
    plt.plot(k_range, test_errors)
    plt.legend(['validation errors', 'test errors'], loc='upper right')
    plt.show()

    valid_acc = 100 * (1 - valid_errors[best_index])
    test_acc = 100 * (1 - test_errors[best_index])
    print(f"Best result: k={best_index - 1}, CV accuracy = {valid_acc:0.2f}%, test accuracy = {test_acc:0.2f}%")


def task1():
    train_data, test_data = preprocess_data('iris.data.csv')
    search_parameters(train_data, test_data)


def knn_sklearn(x_train, y_train, x_test, y_test):
    classifier = KNeighborsClassifier(n_neighbors=10)
    classifier.fit(x_train, y_train)
    return metrics.accuracy_score(y_test, classifier.predict(x_test))


def load_mnist(sample_size=2000):
    mnist = fetch_mldata("MNIST original")
    x, y = shuffle(mnist.data, mnist.target)
    x = x[:sample_size] / 255.0
    y = y[:sample_size]
    return train_test_split(x, y, test_size=0.1)


def task2():
    x_train, _, y_train, _ = load_mnist(5000)

    for perplexity in [10, 50, 100]:
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=300)
        tsne_data = tsne.fit_transform(x_train)

        plt.figure(figsize=(10, 6))
        plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=y_train)
        plt.show()


def pca_transform(data, n_dims):
    data -= np.mean(data, axis=0)
    U, s, _ = np.linalg.svd(data.T, full_matrices=False)
    U = U[:, :n_dims]
    return np.dot(data, U), U


def pca_recover(pca_data, U, mean):
    data = np.dot(pca_data, U.T)
    data += mean
    return data


def task3():
    x_train, x_test, y_train, y_test = load_mnist(10000)

    init_acc = knn_sklearn(x_train, y_train, x_test, y_test)
    print(f"Initial accuracy = {init_acc:.3f}")

    compressed_accs, recovered_accs = [], []
    dims_range = [5, 10, 20, 50, 100]
    for n_dims in dims_range:
        pca_train, U = pca_transform(x_train, n_dims)
        pca_test = np.dot(x_test, U)
        compressed_accs.append(knn_sklearn(pca_train, y_train, pca_test, y_test))

        recovered_train = pca_recover(pca_train, U, np.mean(x_train, axis=0))
        recovered_test = np.dot(pca_test, U.T)
        recovered_accs.append(knn_sklearn(recovered_train, y_train, recovered_test, y_test))

    plt.title("Test accuracy")
    plt.plot(dims_range, compressed_accs)
    plt.plot(dims_range, recovered_accs)
    plt.xlabel("dimensions")
    plt.legend(['compressed accuracy', 'test errors'], loc='lower right')
    plt.show()


def main():
    # task1()
    # task2()
    task3()


if __name__ == "__main__":
    main()
