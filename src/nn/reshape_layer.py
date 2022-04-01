import numpy as np

from .layer import Layer


class ReshapeLayer(Layer):
    def __init__(self, input_shape: tuple, output_shape: tuple):
        super().__init__()
        self.__validate_shapes(input_shape, output_shape)
        self.input_shape = input_shape
        self.output_shape = output_shape

    def __validate_shapes(self, input_shape: tuple, output_shape: tuple):
        input_size = self.__multiply(input_shape)
        output_size = self.__multiply(output_shape)
        if input_size != output_size:
            raise Exception("Incompatible input and output shapes.")

    def forward(self, input: np.ndarray):
        self.last_input = input
        self.last_output = np.reshape(input, self.output_shape)
        return self.last_output

    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)

    @staticmethod
    def __multiply(elements):
        result = 1
        for element in elements:
            result *= element
        return result
