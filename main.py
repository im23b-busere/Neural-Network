import numpy as np
from numpy.ma.core import outer

from dataset import inputs, targets

# sigmoid function for converting the output in the range of 0 to 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# derivative of the sigmoid function for changing the weights and biases
def sigmoid_derivative(x):
    return x * (1 - x)

# loss function for calculating the error
def loss_function(real, prediction):
    return np.mean((real - prediction) ** 2)

# parameters
input_layer_neurons = 2
hidden_layer_neurons = 2
output_layer_neurons = 1

# initializing random weights and biases
np.random.seed(0)

hidden_weights = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons)) # size to make a matrix of 2x2
output_weights = np.random.uniform(size=(hidden_layer_neurons, output_layer_neurons)) # size to make a matrix of 2x1


hidden_biases = np.random.uniform(size=(1, hidden_layer_neurons)) # size to make a matrix of 1x2
output_biases = np.random.uniform(size=(1, output_layer_neurons)) # size to make a matrix of 1x1

# learning rate & cycles
lr = 0.1 # tells at what rate the weights and biases should be updated
cycles = 10000 # number of times the model should be trained

# main training loop
# multiplies each neuron with the weights and then adds the biases
# then applies the sigmoid function to convert the output between 0 and 1
hidden_layer_input = np.dot(inputs, hidden_weights) + hidden_biases
hidden_layer_output = sigmoid(hidden_layer_input)

# same but for the output layer
output_layer_input = np.dot(hidden_layer_output, output_weights) + output_biases
output = sigmoid(output_layer_input)


