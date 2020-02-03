from typing import List
from typing import Tuple

from src import CostFunction
from src.ActivationFunction import *


class Layer:
    num_neurons: int = None
    bias: np.ndarray = None
    weight: np.ndarray = None
    last_input: np.ndarray = None
    last_output: np.ndarray = None
    activationFunction: ActivationFunction = None

    def __init__(self, num_neurons: int, prev_num_neurons: int, activation_function: ActivationFunction = Relu()):
        self.num_neurons = num_neurons
        self.bias = np.random.rand(num_neurons, 1)
        self.activationFunction = activation_function
        self.weight = np.random.rand(prev_num_neurons, num_neurons)

    def forward(self, x_input: np.ndarray) -> np.ndarray:
        self.last_input = x_input
        self.last_output = self.weight.T * x_input + self.bias
        return self.activationFunction.calculate(self.last_output)

    def update_weight(self, learning_rate: float, grads: np.ndarray):
        self.weight = self.weight - grads * learning_rate

    def update_bias(self, learning_rate: float, grads: np.ndarray):
        self.bias = self.bias - grads * learning_rate


class NeuronalNetwork:
    num_inputs: int = None
    num_output: int = None
    hidden_layers: List[Layer] = None
    cost_function: CostFunction = None

    def __init__(self, num_inputs: int, hidden_architecture: List[Tuple[int, ActivationFunction]],
                 num_output: int, cost_function: CostFunction):
        self.hidden_layers = []
        self.num_output = num_output
        self.num_inputs = num_inputs
        self.cost_function = cost_function

        for layer in hidden_architecture:
            self.__add_hidden_layer(layer[0], layer[1])
        # FIXME
        self.__add_hidden_layer(num_output, Sigmoid())

    def __add_hidden_layer(self, num_neurons: int, activation_function: ActivationFunction):
        prev_num_neurons = self.__get_prev_num_neurons()
        new_layer = Layer(num_neurons, prev_num_neurons, activation_function)
        self.hidden_layers.append(new_layer)

    def __get_prev_num_neurons(self) -> int:
        is_no_hidden_layer = len(self.hidden_layers) == 0
        if is_no_hidden_layer:
            return self.num_inputs
        else:
            return self.hidden_layers[-1].num_neurons

    def train(self, data: np.ndarray, learning_rate: float = 0.001, iterations: int = 1000):
        pass

    def forward(self, x_input: np.ndarray) -> np.ndarray:
        for layer in self.hidden_layers:
            x_input = layer.forward(x_input)
        return x_input

    def back_propagation(self, result, expected) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        bias = []
        weight = []
        activations = self.forward(result)

        layer = self.hidden_layers[-1]
        derivative = layer.activationFunction.calculate_derivative
        z = layer.last_output
        delta = self.cost_function.calculate(activations, expected) * derivative(z)

        bias.append(delta)
        weight.append(np.dot(delta, layer.last_input.T))

        # FIXME check index out of the range
        for layer in self.hidden_layers[::-1]:
            z = layer.last_output
            derivative = layer.activationFunction.calculate_derivative
            sp = derivative(z)
            delta = np.dot(layer.weight.T, delta) * sp

            bias.append(delta)
            weight.append(np.dot(delta, layer.last_input.T))

        return bias, weight
