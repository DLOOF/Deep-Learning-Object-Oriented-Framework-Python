from src.ActivationFunction import *
from typing import List


class Layer:
    activationFunction: ActivationFunction = Relu()
    num_neurons = 0

    def __init__(self, activation_function: ActivationFunction, num_neurons: int):
        self.num_neurons = num_neurons
        self.activationFunction = activation_function


class NeuronalNetwork:
    num_inputs: int = 0
    num_output: int = 0
    hidden_layers: List[Layer] = []

    def __init__(self, num_inputs: int, hidden_layers: List[Layer], num_output: int):
        self.num_inputs = num_inputs
        self.num_output = num_output
        self.hidden_layers = hidden_layers
