import numpy as np

from nn.linear_layer import LinearLayer
from nn.relu_activation_layer import ReluActivationLayer
from nn.sigmoid_activation_layer import SigmoidActivationLayer
from nn.tanh_activation_layer import TanhActivationLayer
from nn.plain_nn import PlainNN


def tan_activation(x):
    return np.tanh(x)


def tan_activation_deriv(x):
    return 1 - np.tanh(x) ** 2




input = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
output = np.reshape([0, 1, 1, 0], (4, 1, 1))


l1_lin = LinearLayer(2, 3)
# l1_act = ActivationLayer(tan_activation, tan_activation_deriv)
l1_act = SigmoidActivationLayer()
# l1_act = ReluActivationLayer()
# l1_act = TanhActivationLayer()

l2_lin = LinearLayer(3, 1)
l2_act = SigmoidActivationLayer()
# l2_act = ReluActivationLayer()
# l2_act = TanhActivationLayer()

epochs = 2000
learning_rate = 0.35 # tanh - 0.2 (1000 epok), sigmoid = 0.35(2000 epok)

layers = [l1_lin, l1_act, l2_lin, l2_act]


network = PlainNN(layers)
network.fit(input, output, epochs, learning_rate, debug=True)

for idx in range(4):
    output = network.process_input(input[idx])
    print(f'OUT[{input[idx][0]},{input[idx][1]}] ={output}')



