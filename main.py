import numpy as np
import matplotlib.pyplot as plt
from dataset import inputs, targets
from model import sigmoid, sigmoid_derivative, loss_function, initialize_parameters

# error history
error_history = []

# parameters
input_layer_neurons = 2
hidden_layer_neurons = 2
output_layer_neurons = 1

# learning rate & cycles
lr = 0.1  # tells at what rate the weights and biases should be updated
cycles = 10000  # number of times the model should be trained

# initialize parameters
hidden_weights, output_weights, hidden_biases, output_biases = initialize_parameters(
    input_layer_neurons, hidden_layer_neurons, output_layer_neurons
)

# main training loop
# multiplies each neuron with the weights and then adds the biases
# then applies the sigmoid function to convert the output between 0 and 1
for cycle in range(cycles):
    # forward propagation
    hidden_layer_input = np.dot(inputs, hidden_weights) + hidden_biases
    hidden_layer_output = sigmoid(hidden_layer_input)

    # same but for the output layer
    output_layer_input = np.dot(hidden_layer_output, output_weights) + output_biases
    output = sigmoid(output_layer_input)

    # calculate the error
    error = output - targets

    # backpropagation
    # calculate the derivative of the output layer
    derivative_output = error * sigmoid_derivative(output)

    # calculate the derivative of the hidden layer
    error_hidden_layer = derivative_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # update the weights and biases
    output_weights -= hidden_layer_output.T.dot(derivative_output) * lr
    hidden_weights -= inputs.T.dot(d_hidden_layer) * lr

    output_biases -= np.sum(derivative_output, axis=0, keepdims=True) * lr
    hidden_biases -= np.sum(d_hidden_layer, axis=0, keepdims=True) * lr

    # save the error
    loss = loss_function(targets, output)
    error_history.append(loss)

    # print the error
    if cycle % 100 == 0:
        print(f"Cycle {cycle}, Error: {loss}")

# visualize the error history
plt.figure(figsize=(10, 6))
plt.plot(error_history, label='Fehlerverlauf')
plt.xlabel('Zyklen')
plt.ylabel('Fehler')
plt.title('Fehlerentwicklung während des Trainings')
plt.legend()
plt.grid()
plt.show()
