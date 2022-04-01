import numpy as np
from scipy import signal


def convolute(input: np.ndarray, kernels: np.ndarray) -> np.ndarray:

    output_shape = (len(kernels), input.shape[1] - kernels.shape[1] + 1, input.shape[2] - kernels.shape[2] + 1)

    convoluted_output = []
    for kernel_idx in range(len(kernels)):
        convoluted_output.append(np.zeros(output_shape))   # [Y,X]
        for input_depth_idx in range(input.shape[0]):
            convoluted_output[kernel_idx] += signal.correlate2d(input[input_depth_idx], kernels[kernel_idx], 'valid')
    return convoluted_output


# 2 x 4 x4
input = [
    [
        [10, 10, 10, 10],
        [10, 10, 10, 10],
        [10, 10, 10, 10],
        [10, 10, 10, 10]
    ],
    [
        [20, 20, 20, 20],
        [20, 20, 20, 20],
        [20, 20, 20, 20],
        [20, 20, 20, 20]
    ]
]

np_input = np.asarray(input)

print(f'shape={np_input.shape[-2:]}')

kerns = [[
    [1, 1],
    [1, 1]
]]

np_kerns = np.asarray(kerns)

result = convolute(np_input, np_kerns)





