from src.ActivationFunction import *
from typing import List
import numpy as np


class Layer:
    num_neurons = 0
    bias = np.array(0, 0)
    weight = np.array(0, 0)
    activationFunction: ActivationFunction = Relu()

    def __init__(self, num_neurons: int, prev_num_neurons: int, activation_function: ActivationFunction):
        self.num_neurons = num_neurons
        self.bias = np.random.rand(num_neurons, 1)
        self.activationFunction = activation_function
        self.weight = np.random.rand(prev_num_neurons, num_neurons)


class NeuronalNetwork:
    num_inputs: int = 0
    num_output: int = 0
    hidden_layers: List[Layer] = []

    def __init__(self, num_inputs: int, num_output: int):
        self.hidden_layers = []
        self.num_inputs = num_inputs
        self.num_output = num_output

    def add_hidden_layer(self, num_neurons: int, activation_function: ActivationFunction):
        prev_num_neurons = self.__get_prev_num_neurons()
        new_layer = Layer(num_neurons, prev_num_neurons, activation_function)
        self.hidden_layers.append(new_layer)

    def __get_prev_num_neurons(self) -> int:
        is_no_hidden_layer = len(self.hidden_layers) == 0
        if is_no_hidden_layer:
            return self.num_inputs
        else:
            return self.hidden_layers[-1].num_neurons

    def train(self):
        pass

    def forward(self):
        pass

    def back_propagation(self):
        pass
