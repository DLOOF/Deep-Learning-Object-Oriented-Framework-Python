import time
from typing import List, Tuple

from src import CostFunction
from src.ActivationFunction import *


class Layer:
    bias: np.array = None
    weight: np.array = None
    num_neurons: int = None
    last_input: np.array = None
    last_output: np.array = None
    activationFunction: ActivationFunction = None

    def __init__(self, num_neurons: int, prev_num_neurons: int, activation_function: ActivationFunction = Relu()):
        self.num_neurons = num_neurons
        self.bias = np.random.rand(num_neurons, 1)
        self.activationFunction = activation_function
        self.weight = np.random.rand(prev_num_neurons, num_neurons).T

    def forward(self, x_input: np.array) -> np.array:
        self.last_input = x_input
        self.last_output = np.dot(self.weight, x_input) + self.bias
        return self.activationFunction.calculate(self.last_output)

    def update_weight(self, learning_rate: float, grads: np.array):
        assert self.weight.shape == grads.shape
        self.weight = self.weight - grads * learning_rate

    def update_bias(self, learning_rate: float, grads: np.array):
        assert self.bias.shape == grads.shape
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

    def train(self, data: np.array, learning_rate: float = 0.001, iterations: int = 1000):
        print("Starting training!")
        tic = time.time()
        data_shape = data.shape
        matrix_depth = data_shape[1]
        expected_output = self.__get_column(data, matrix_depth - 1)
        training_matrix = self.__get_columns_between_a_and_b(data, 0, matrix_depth - 1)

        for i in range(iterations):
            for j, example in enumerate(training_matrix):
                var = example.reshape(2, 1)
                output = self.forward(var)
                # FIXME check the expected value type: should be a np.array (check the case when we have single value)
                bias_grads, weight_grads = self.back_propagation(output, expected_output[j])
                self.update_biases(bias_grads, learning_rate)
                self.update_weight(weight_grads, learning_rate)
        toc = time.time()

        print(f"Training finished! {toc - tic}")

    @staticmethod
    def __get_columns_between_a_and_b(matrix: np.array, a: int, b: int):
        return matrix[:, a:b]

    @staticmethod
    def __get_column(matrix: np.array, index: int) -> np.array:
        return matrix[:, [index]]

    def forward(self, x_input: np.array) -> np.array:
        for layer in self.hidden_layers:
            x_input = layer.forward(x_input)
        return x_input

    def back_propagation(self, result, expected) -> Tuple[List[np.array], List[np.array]]:
        bias = []
        weight = []

        gradient = self.cost_function.calculate_derivative(result, expected)

        # FIXME check index out of the range
        for layer in self.hidden_layers[::-1]:
            z = layer.last_output

            # element-wise multiplication
            gradient = np.multiply(gradient, layer.activationFunction.calculate_derivative(z))

            bias.append(gradient)
            weight.append(np.dot(gradient, layer.last_input.T))

            gradient = np.dot(layer.weight.T, gradient)

        return bias, weight

    def update_biases(self, bias_grads, learning_rate):
        for i, layer in enumerate(self.hidden_layers[::-1]):
            layer.update_bias(learning_rate, bias_grads[i])
        pass

    def update_weight(self, weight_grads, learning_rate):
        for i, layer in enumerate(self.hidden_layers[::-1]):
            layer.update_weight(learning_rate, weight_grads[i])
        pass
