from typing import List, Tuple
from src.ActivationFunctions.ActivationFunction import *
from src.InitializationFunctions.InitializationFunction import Xavier, InitializationFunction


class Layer:
    # bias: np.array = None
    # weight: np.array = None
    # num_neurons: int = None
    # last_input: np.array = None
    # last_output: np.array = None
    # activationFunction: ActivationFunction = None

    def __init__(self, num_neurons: int,
                 prev_num_neurons: int,
                 activation_function: ActivationFunction = Relu(),
                 initialization_function: InitializationFunction = Xavier()):
        self.last_input = None
        self.last_output = None
        self.initialization_function = initialization_function
        self.num_neurons = num_neurons
        self.activationFunction = activation_function
        self.weight = initialization_function.initialize(prev_num_neurons, num_neurons).T
        self.bias = initialization_function.initialize(self.num_neurons, 1)

    def forward(self, x_input: np.array) -> np.array:
        self.last_input = x_input
        self.last_output = np.dot(self.weight, x_input)
        self.last_output += self.bias
        return self.activationFunction.calculate(self.last_output)

    def update_weight(self, learning_rate: float, grads: np.array):
        assert self.weight.shape == grads.shape
        self.weight = self.weight - grads * learning_rate / self.last_input.shape[1]

    def update_bias(self, learning_rate: float, grads: np.array):
        # takes the average gradient for each *batch* to be applied to the overall bias
        g = np.sum(grads, axis=1, keepdims=True) / self.last_input.shape[1]
        assert self.bias.shape == g.shape
        self.bias = self.bias - g * learning_rate
