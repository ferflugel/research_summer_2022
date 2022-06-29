import numpy as np


def data_generating_function(h, size, complexity):
    x = np.ones((h.shape[0], size))
    x = x * np.random.normal(0, 1, size)
    for i in range(size):
        x.T[i] = transform(x.T[i], h, complexity)
    return x


def transform(x_i, h, complexity):
    for i in range(complexity):
        if np.random.random() < 0.5:
            x_i = non_linear_transformation(x_i)
        else:
            h_i = h.T[np.random.randint(h.shape[1])]
            x_i = linear_combination(x_i, h_i)
    return x_i


def linear_combination(x_i, h_i):
    return x_i + h_i


def non_linear_transformation(x):
    functions = {0: np.sin, 1: np.exp, 2: np.tanh,
                 3: np.sign, 4: np.abs, 5: relu, 6: sigmoid}
    f_choice = np.random.randint(low=0, high=len(functions))
    return functions[f_choice](x)


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
