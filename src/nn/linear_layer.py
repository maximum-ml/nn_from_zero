import numpy as np

from .layer import Layer


class LinearLayer(Layer):

    def __init__(self, input_size, output_size):
        self.last_input = None
        self.last_output = None
        self.weights = np.random.rand(output_size, input_size)
        self.biases = np.random.rand(output_size, 1)

    def forward(self, input: np.ndarray):
        self.last_input = input
        dot = np.dot(self.weights, input)
        self.last_output = dot + self.biases
        return self.last_output

    def backward(self, output_gradient, learning_rate):
        # 1. calculate dE/dW = dE/dy * dy/dW = dE/dy * x^T
        weights_gradient = np.dot(output_gradient, self.last_input.T)
        # 2. calculate dE/db = dE/dy * dy/dB = dE/dy
        biases_gradient = output_gradient
        # 3. adjust weights and biases
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient
        # 4. calculate dE/dx = dE/dy * dy/dx = w^T * dE/dy
        input_gradient = np.dot(self.weights.T, output_gradient)
        return input_gradient
