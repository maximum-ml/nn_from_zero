import numpy as np
from .activation_layer import ActivationLayer


def relu(x):
    return np.maximum(0, x)


def relu_deriv(x):
    return np.where(x <= 0, 0, 1)

# This does not work as activation (use it only for cost function)
class ReluActivationLayer(ActivationLayer):
    def __init__(self):
        super().__init__(relu, relu_deriv)
