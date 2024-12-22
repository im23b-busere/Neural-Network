import numpy as np
from dataset import inputs, targets

# sigmoid function for putting the output in the range of 0 to 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# derivative of the sigmoid function for changing the weights and biases
def sigmoid_derivative(x):
    return x * (1 - x)

# loss function for calculating the error
def loss_function(real, prediction):
    return np.mean((real - prediction) ** 2)