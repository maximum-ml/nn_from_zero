import numpy as np

from .activation_layer import ActivationLayer


def sigmoid(x):
    x = np.minimum(x, 10)
    x = np.maximum(x, -10)

    y = 1 / (1 + np.exp(-x))

    # if len(y) == 2:
    #     print(f'SIG: x={x}, y={y}, exp={np.exp(-x)}')

    return y


def sigmoid_deriv(x):
    sig = sigmoid(x)
    return sig * (1 - sig)


class SigmoidActivationLayer(ActivationLayer):
    def __init__(self):
        super().__init__(sigmoid, sigmoid_deriv)

