from src.ActivationFunctions.ActivationFunction import *
from src.InitializationFunctions.InitializationFunction import Xavier, InitializationFunction
from src.Networks.Layer.Layer import Layer
from src.Regularizations import NormRegularizationFunction


class ClassicLayer(Layer):

    def __init__(self, num_neurons: int, prev_num_neurons: int, activation_function: ActivationFunction = Relu(),
                 initialization_function: InitializationFunction = Xavier()):
        super().__init__()
        self.initialization_function = initialization_function
        self.num_neurons = num_neurons
        self.activationFunction = activation_function
        self.weight = initialization_function.initialize(prev_num_neurons, num_neurons).T
        self.bias = initialization_function.initialize(self.num_neurons, 1)

    def forward(self, x_input: np.array) -> np.array:
        self.last_input = x_input
        self.last_output = self.activationFunction.calculate(np.dot(self.weight, x_input) + self.bias)
        return self.last_output

    def backward(self, gradient: np.array, learning_rate: float,
                 regularization_function: NormRegularizationFunction) -> np.array:
        dz = self.activationFunction.calculate_gradient(self.last_output)
        final_gradient = np.multiply(gradient, dz)
        bias_gradient = final_gradient + regularization_function.calculate_gradient_bias(self)
        weight_gradient = np.dot(final_gradient, self.last_input.T)
        weight_gradient += regularization_function.calculate_gradient_weights(self)

        self.update_bias(learning_rate, bias_gradient)
        self.update_weight(learning_rate, weight_gradient)

        # FIXME check the expected value type: should be a np.array (check the case when we have single value)
        return np.dot(self.weight.T, final_gradient)

    def update_weight(self, learning_rate: float, grads: np.array):
        assert self.weight.shape == grads.shape
        self.weight = self.weight - grads * learning_rate / self.last_input.shape[1]

    def update_bias(self, learning_rate: float, grads: np.array):
        # takes the average gradient for each *batch* to be applied to the overall bias
        g = np.sum(grads, axis=1, keepdims=True) / self.last_input.shape[1]
        assert self.bias.shape == g.shape
        self.bias = self.bias - g * learning_rate
