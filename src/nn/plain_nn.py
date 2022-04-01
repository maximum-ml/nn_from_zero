import numpy as np

from .loss_function import *


class PlainNN:
    def __init__(self, layers : list, loss_function=mse, loss_function_deriv=mse_deriv):
        self.layers = layers
        self.loss_function = loss_function
        self.loss_function_deriv = loss_function_deriv

    def fit(self, input: np.ndarray, output: np.ndarray, epochs: int, learning_rate: float, debug=False):
        for i in range(epochs):
            self.__learning_cycle(input, output, learning_rate, i, debug)

    def process_input(self, input: np.ndarray) -> np.ndarray:
        return self.__forward_propagation(input)

    def __learning_cycle(self, input, output, learning_rate, cycle, debug):
        total_error = 0     # for debug only
        for train_x, train_y in zip(input, output):
            # 1. perform forward propagation with a single input data probe
            y = self.__forward_propagation(train_x)

            # 2. calculate mse
            total_error += self.loss_function(train_y, y)
            # results.append([train_y[0][0], x[0][0]])

            final_grad = self.loss_function_deriv(train_y, y)
            # 3. perform full backward propagation cycle
            grad = final_grad
            for layer in reversed(self.layers):
                grad = layer.backward(grad, learning_rate)

        # if debug and cycle % 5 == 0:
        print(f'cycle=[{cycle+1}], error={total_error}, grad={final_grad.tolist()}')

    def __forward_propagation(self, input: np.ndarray):
        x = input
        for layer in self.layers:
            # print(f'in={x}')
            x = layer.forward(x)
            # print(f'out={x}')
        return x

    def __backward_propagation(self, initial_grad, learning_rate):
        grad = initial_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)

