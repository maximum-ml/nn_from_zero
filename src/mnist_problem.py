import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from matplotlib import pyplot as plt

from nn.plain_nn import PlainNN
from nn.convolutional_layer import ConvolutionalLayer
from nn.sigmoid_activation_layer import SigmoidActivationLayer
from nn.reshape_layer import ReshapeLayer
from nn.linear_layer import LinearLayer
from nn.loss_function import binary_cross_entropy
from nn.loss_function import binary_cross_entropy_deriv


def show_mnist_image(data):
    two_dim = (np.reshape(data, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_dim, interpolation='nearest')


def show_info(data_set: np.ndarray, label=None):
    print(f'{label}: size={data_set.size}, len={len(data_set)}, shape={data_set.shape}')


def extract_probes(data_set: tuple, numbers: tuple, limit: int):
    '''
    data_set - (x_probes, y_probes)
    numbers - list of labels(numbers) to extract
    '''
    x, y = data_set
    idx = np.isin(y, numbers)
    x_selected = x[idx]
    y_selected = y[idx]
    return x_selected[:limit], y_selected[:limit]


# load MNIST dataset from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# prepare data for processing
print(f'x_train.size={x_train.size}')
print(f'x_train.shape={x_train.shape}')
print(f'y_train.size={y_train.size}')
print(f'y_train.shape={y_train.shape}')
show_info(x_test, 'x_test')
show_info(y_test, 'y_test')

numbers_to_recognize = [0, 1, 2]

# training data set ...
x_train_sel, y_train_sel = extract_probes((x_train, y_train), numbers_to_recognize, 100)

x_train_prep = x_train_sel.reshape(len(x_train_sel), 1, 28, 28).astype('float32') /255
y_train_prep = np_utils.to_categorical(y_train_sel).reshape(len(y_train_sel), len(numbers_to_recognize), 1) # converts to 'one hot encoding'

show_info(x_train_prep, 'x_train_prep')
show_info(y_train_prep, 'y_train_prep')

# test data set ...
x_test_sel, y_test_sel = extract_probes((x_test, y_test), numbers_to_recognize, 50)

x_test_prep = x_test_sel.reshape(len(x_test_sel), 1, 28, 28).astype('float32') /255
y_test_prep = np_utils.to_categorical(y_test_sel).reshape(len(y_test_sel), len(numbers_to_recognize), 1) # converts to 'one hot encoding'

# print(y_train_sel)
# print(y_train_prep)

# Create NN

layers = [
    ConvolutionalLayer((1, 28, 28), 3, 1),
    SigmoidActivationLayer(),
    ReshapeLayer((1, 26, 26), (1 * 26 * 26, 1)),
    LinearLayer(1 * 26 * 26, len(numbers_to_recognize)),
    # SigmoidActivationLayer(),
    # LinearLayer(100, 2),
    SigmoidActivationLayer()
]

network = PlainNN(layers, binary_cross_entropy, binary_cross_entropy_deriv)

network.fit(x_train_prep, y_train_prep, 30, 0.01, True)

print("==========================")

# print(x_train_prep)
# for i in range(len(x_train_prep)) :
#     print(f"M={np.mean(x_train_prep[i])}")

missclassifications = 0
for idx in range(len(x_test_prep)):
    calculated = network.process_input(x_test_prep[idx])
    value = np.argmax(calculated)
    real_value = y_test_sel[idx]
    real = y_test_prep[idx]
    print(f'OUT[calculated={calculated.tolist()}, real={real.tolist()}, MISS={value != real_value}]')
    if value != real_value:
        missclassifications += 1
        show_mnist_image(x_test_prep[idx])
        plt.show()


print(f'missed = {missclassifications}')



