import numpy as np
from scipy import signal

from .layer import Layer


class ConvolutionalLayer(Layer):
    def __init__(self, input_shape: tuple, kernel_size: int, nb_of_kernels: int):
        """
            We assume the input is a 3-dimensional array -> input_shape = (z, y, x).
            Output shape = (nb_of_kernels, y', x')
        """
        super().__init__()
        # self.__validate_dimensions(input_shape, kernel_size, nb_of_kernels)
        self.input_shape = input_shape  # (z, y, x)
        self.input_depth, self.input_height, self.input_width = input_shape
        self.output_shape = (nb_of_kernels, self.input_height - kernel_size + 1, self.input_width - kernel_size + 1)

        self.kernels_shape = (nb_of_kernels, self.input_depth, kernel_size, kernel_size)
        self.kernels = np.random.rand(*self.kernels_shape)
        self.biases = np.random.rand(*self.output_shape)


    def __validate_dimensions(self, input_shape: tuple, kernel_size: int, nb_of_kernels: int):
        pass #TODO

    def forward(self, input: np.ndarray):
        self.last_input = input
        self.last_output = self.convolute(input, self.kernels)
        self.last_output += self.biases
        return self.last_output

    def convolute(self, input: np.ndarray, kernels: np.ndarray) -> np.ndarray:

        convoluted_output = np.zeros(self.output_shape)
        for kernel_idx in range(len(kernels)):
            # convoluted_output.append(np.zeros(self.output_shape))  # [Y,X]
            for input_depth_idx in range(input.shape[0]):
                convoluted_output[kernel_idx] += signal.correlate2d(input[input_depth_idx], kernels[kernel_idx, input_depth_idx], 'valid')
        return convoluted_output

    def backward(self, output_gradient, learning_rate):
        biases_gradient = output_gradient
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for kernel_idx in range(self.kernels_shape[0]):
            for input_depth_idx in range(self.input_depth):
                kernels_gradient[kernel_idx, input_depth_idx] = signal.correlate2d(self.last_input[input_depth_idx], output_gradient[kernel_idx], "valid")
                input_gradient[input_depth_idx] += signal.convolve2d(output_gradient[kernel_idx], self.kernels[kernel_idx, input_depth_idx], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * biases_gradient

        return input_gradient

