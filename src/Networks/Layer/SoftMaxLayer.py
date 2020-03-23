import numpy as np

from src.ActivationFunctions.ActivationFunction import *
from src.InitializationFunctions.InitializationFunction import Xavier, InitializationFunction
from src.Networks.Layer.Layer import Layer
from src.Regularizations import NormRegularizationFunction


class SoftMaxLayer(Layer):

    def __init__(self, num_neurons: int, activation_function: ActivationFunction = SoftMax(),
                 initialization_function: InitializationFunction = Xavier()):
        super().__init__()
        self.num_neurons = num_neurons
        self.activationFunction = activation_function
        self.initialization_function = initialization_function
        self.bias = initialization_function.initialize(self.num_neurons, 1)

    def __init_weight_late__(self, x_input: np.array):
        if self.weight is None:
            prev_num_neurons = x_input.shape[0]
            self.weight = self.initialization_function.initialize(prev_num_neurons, self.num_neurons).T

    def forward(self, x_input: np.array) -> np.array:

        self.last_input_shape = x_input.shape

        x_input = x_input.flatten()

        self.last_input = x_input
        self.__init_weight_late__(x_input)

        self.last_output = (self.weight @ x_input).reshape(-1, 1) + self.bias
        self.last_activation_output = self.activationFunction.calculate(self.last_output)
        return self.last_activation_output

    def backward(self, gradient: np.array, learning_rate: float,
                 regularization_function: NormRegularizationFunction) -> np.array:
        pass

    def update_weight(self, learning_rate: float, grads: np.array):
        assert self.weight.shape == grads.shape
        # TODO check this: self.weight = self.weight - grads * learning_rate / self.last_input.shape[1]
        self.weight = self.weight - grads * learning_rate

    def update_bias(self, learning_rate: float, grads: np.array):
        assert self.bias.shape == grads.shape
        self.bias = self.bias - grads * learning_rate

    def __str__(self):
        return f"SM{self.num_neurons}"
