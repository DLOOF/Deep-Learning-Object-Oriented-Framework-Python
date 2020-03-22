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

        for i, x_gradient in enumerate(gradient):
            # e^totals
            t_exp = np.exp(self.last_output)

            # Sum of all e^totals
            s = np.sum(t_exp)

            # Gradients of out[i] against totals
            d_out_d_t = -t_exp[i] * t_exp / (s ** 2)
            d_out_d_t[i] = t_exp[i] * (s - t_exp[i]) / (s ** 2)

            # Gradients of totals against weights/biases/input
            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.weight

            # Gradients of loss against totals
            d_L_d_t = x_gradient * d_out_d_t

            # Gradients of loss against weights/biases/input
            d_L_d_w = d_L_d_t @ d_t_d_w.reshape(1, -1)
            d_L_d_b = d_L_d_t * d_t_d_b
            d_L_d_inputs = d_t_d_inputs.T @ d_L_d_t

            # Update weights / biases
            self.update_weight(learning_rate, d_L_d_w)
            self.update_bias(learning_rate, d_L_d_b)

            return d_L_d_inputs.reshape(self.last_input_shape)

    def update_weight(self, learning_rate: float, grads: np.array):
        assert self.weight.shape == grads.shape
        # TODO check this: self.weight = self.weight - grads * learning_rate / self.last_input.shape[1]
        self.weight = self.weight - grads * learning_rate

    def update_bias(self, learning_rate: float, grads: np.array):
        assert self.bias.shape == grads.shape
        self.bias = self.bias - grads * learning_rate

    def __str__(self):
        return f"SM{self.num_neurons}"
