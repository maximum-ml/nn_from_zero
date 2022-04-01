# ABSTRACT CLASS

class Layer:
    def __init__(self):
        self.last_input = None
        self.last_output = None

    def forward(self, input):
        # ABSTRACT - update input & output and return output
        pass

    def backward(self, output_gradient, learning_rate):
        # ABSTRACT - update parameters (W, B) and return input gradient
        pass

