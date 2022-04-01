import numpy as np


def mse(expected, actual):
    return np.mean(np.power(actual - expected, 2))


def mse_deriv(expected, actual):
    return 2 * (actual - expected) / np.size(actual)


def binary_cross_entropy(expected, actual):
    return -np.mean(expected * np.log(actual) + (1 - expected) * np.log(1 - actual))


def binary_cross_entropy_deriv(expected, actual):
    return ((1 - expected) / (1 - actual) - expected / actual) / np.size(expected)


