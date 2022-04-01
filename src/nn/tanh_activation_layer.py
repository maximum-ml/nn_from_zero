import numpy as np

from .activation_layer import ActivationLayer


# tan_activation_f = lambda x: np.tanh(x)
# tan_activation_fd = lambda x: 1 - np.tanh(x) ** 2


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1 - np.tanh(x) ** 2


class TanhActivationLayer(ActivationLayer):
    def __init__(self):
        super().__init__(tanh, tanh_deriv)
