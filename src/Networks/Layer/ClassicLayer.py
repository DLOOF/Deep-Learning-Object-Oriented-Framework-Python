from src.ActivationFunctions.ActivationFunction import *
from src.InitializationFunctions.InitializationFunction import *
from src.Networks.Layer.Layer import Layer
from src.Regularizations import NormRegularizationFunction


class ClassicLayer(Layer):

    def __init__(self, num_neurons: int, activation_function: ActivationFunction = Relu(),
                 initialization_function: InitializationFunction = He()):
        super().__init__()
        self.num_neurons = num_neurons
        self.activationFunction = activation_function
        self.initialization_function = initialization_function
        self.bias = initialization_function.initialize(1, self.num_neurons)

    def forward(self, x_input: np.array) -> np.array:
        self.__init_weight_late__(x_input)

        self.last_input = x_input
        self.last_output = (x_input @ self.weight) + self.bias
        self.last_activation_output = self.activationFunction.calculate(self.last_output)

        return self.last_activation_output

    def __init_weight_late__(self, x_input: np.array):
        if self.weight is None:
            _, prev_num_neurons = x_input.shape
            self.weight = self.initialization_function.initialize(prev_num_neurons, self.num_neurons)

    def backward(self, gradient: np.array, learning_rate: float,
                 regularization_function: NormRegularizationFunction) -> np.array:
        dz = self.activationFunction.calculate_gradient(self.last_activation_output)
        final_gradient = self.activationFunction.operation()(dz, gradient)

        bias_gradient = final_gradient + regularization_function.calculate_gradient_bias(self)
        bias_gradient = np.sum(bias_gradient, axis=0, keepdims=True) / self.last_input.shape[0]

        weight_gradient = self.last_input.T @ final_gradient
        weight_gradient += regularization_function.calculate_gradient_weights(self)
        weight_gradient /= self.last_input.shape[0]

        final_gradient = final_gradient @ self.weight.T

        self.update_bias(learning_rate, bias_gradient)
        self.update_weight(learning_rate, weight_gradient)

        # FIXME check the expected value type: should be a np.array (check the case when we have single value)
        return final_gradient

    def update_weight(self, learning_rate: float, grads: np.array):
        assert self.weight.shape == grads.shape
        self.weight = self.weight + self.optimizer.calculate_weight(grads, learning_rate)

    def update_bias(self, learning_rate: float, grads: np.array):
        # takes the average gradient for each *batch* to be applied to the overall bias
        assert self.bias.shape == grads.shape
        self.bias = self.bias + self.optimizer.calculate_bias(grads, learning_rate)

    def __str__(self):
        return f"{self.activationFunction}{self.num_neurons}"
