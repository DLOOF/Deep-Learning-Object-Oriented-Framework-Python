from src.ActivationFunctions.ActivationFunction import *
from src.InitializationFunctions.InitializationFunction import Xavier, InitializationFunction
from src.Networks.Layer.Layer import Layer
from src.Regularizations import NormRegularizationFunction


class SoftMaxLayer(Layer):

    def __init__(self, num_neurons: int, prev_num_neurons: int, activation_function: ActivationFunction = SoftMax(),
                 initialization_function: InitializationFunction = Xavier()):
        super().__init__()
        self.initialization_function = initialization_function
        self.num_neurons = num_neurons
        self.activationFunction = activation_function
        self.weight = initialization_function.initialize(prev_num_neurons, num_neurons).T
        self.bias = initialization_function.initialize(self.num_neurons, 1)

    def forward(self, x_input: np.array) -> np.array:
        self.last_input_shape = x_input.shape

        x_input = x_input.flatten()
        self.last_input = x_input

        self.last_output = np.dot(self.weight, x_input) + self.bias
        self.last_activation_output = self.activationFunction.calculate(self.last_output)
        return self.last_activation_output

    def backward(self, gradient: np.array, learning_rate: float,
                 regularization_function: NormRegularizationFunction) -> np.array:

        for i, x_gradient in enumerate(gradient):
            if gradient == 0:
                continue

            # e^totals
            t_exp = np.exp(self.last_output)

            # Sum of all e^totals
            S = np.sum(t_exp)

            # Gradients of out[i] against totals
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            # Gradients of totals against weights/biases/input
            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.weight

            # Gradients of loss against totals
            d_L_d_t = gradient * d_out_d_t

            # Gradients of loss against weights/biases/input
            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            d_L_d_b = d_L_d_t * d_t_d_b
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t

            # Update weights / biases
            self.update_weight(learning_rate, d_L_d_w)
            self.update_bias(learning_rate, d_L_d_b)

            return d_L_d_inputs.reshape(self.last_input_shape)

    def update_weight(self, learning_rate: float, grads: np.array):
        assert self.weight.shape == grads.shape
        self.weight = self.weight - grads * learning_rate / self.last_input.shape[1]

    def update_bias(self, learning_rate: float, grads: np.array):
        # takes the average gradient for each *batch* to be applied to the overall bias
        g = np.sum(grads, axis=1, keepdims=True) / self.last_input.shape[1]
        assert self.bias.shape == g.shape
        self.bias = self.bias - g * learning_rate
