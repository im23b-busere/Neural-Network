import numpy as np

# sigmoid function for converting the output in the range of 0 to 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# derivative of the sigmoid function for changing the weights and biases
def sigmoid_derivative(x):
    return x * (1 - x)

# loss function for calculating the error
def loss_function(real, prediction):
    return np.mean((real - prediction) ** 2)

# initializing random weights and biases
def initialize_parameters(input_layer_neurons, hidden_layer_neurons, output_layer_neurons, seed=0):
    np.random.seed(seed)

    hidden_weights = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))  # size to make a matrix of 2x2
    output_weights = np.random.uniform(size=(hidden_layer_neurons, output_layer_neurons))  # size to make a matrix of 2x1

    hidden_biases = np.random.uniform(size=(1, hidden_layer_neurons))  # size to make a matrix of 1x2
    output_biases = np.random.uniform(size=(1, output_layer_neurons))  # size to make a matrix of 1x1

    return hidden_weights, output_weights, hidden_biases, output_biases
