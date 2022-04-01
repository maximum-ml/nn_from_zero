from .layer import Layer


class ActivationLayer(Layer):
    def __init__(self, activation_func, activation_func_deriv):
        self.activation_func = activation_func
        self.activation_func_deriv = activation_func_deriv

    def forward(self, input):
        self.last_input = input
        self.last_output = self.activation_func(input)
        return self.last_output

    def backward(self, output_gradient, learning_rate):
        input_gradient = output_gradient * self.activation_func_deriv(self.last_input)
        return input_gradient
