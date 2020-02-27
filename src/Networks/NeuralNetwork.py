import time
from typing import List, Tuple

from src.CostFunctions import CostFunction
from src.ActivationFunctions.ActivationFunction import *
import matplotlib.pyplot as plt

from src.Networks.SupervisedModel import SupervisedModel
from src.Regularizations.EarlyStoppingRegularization import StoppingCondition, VoidStoppingCondition


class Layer:
    bias: np.array = None
    weight: np.array = None
    num_neurons: int = None
    last_input: np.array = None
    last_output: np.array = None
    activationFunction: ActivationFunction = None

    def __init__(self, num_neurons: int, prev_num_neurons: int, activation_function: ActivationFunction = Relu()):
        self.num_neurons = num_neurons
        self.bias = None
        self.activationFunction = activation_function
        self.weight = np.random.rand(prev_num_neurons, num_neurons).T

    def forward(self, x_input: np.array) -> np.array:
        if self.bias is None:
            self.bias = np.random.randn(self.num_neurons, x_input.shape[1])
        self.last_input = x_input
        self.last_output = np.dot(self.weight, x_input)
        self.last_output = self.last_output + self.bias
        return self.activationFunction.calculate(self.last_output)

    def update_weight(self, learning_rate: float, grads: np.array):
        assert self.weight.shape == grads.shape
        self.weight = self.weight - grads * learning_rate

    def update_bias(self, learning_rate: float, grads: np.array):
        assert self.bias.shape == grads.shape
        self.bias = self.bias - grads * learning_rate


class NeuralNetwork(SupervisedModel):

    def __init__(self, num_inputs: int, hidden_architecture: List[Tuple[int, ActivationFunction]],
                 num_output: int, cost_function: CostFunction, learning_rate: float = 0.001,
                 iterations: int = 1000, stopping_condition: StoppingCondition = VoidStoppingCondition()):
        self.hidden_layers = []
        self.num_output = num_output
        self.num_inputs = num_inputs
        self.cost_function = cost_function
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.stopping_condition = stopping_condition

        for layer in hidden_architecture:
            self.__add_layer(layer[0], layer[1])
        # TODO should be able to specify the output activation function e.g. Softmax
        self.__add_layer(num_output, Sigmoid())

    def __add_layer(self, num_neurons: int, activation_function: ActivationFunction):
        prev_num_neurons = self.__get_prev_num_neurons()
        new_layer = Layer(num_neurons, prev_num_neurons, activation_function)
        self.hidden_layers.append(new_layer)

    def __get_prev_num_neurons(self) -> int:
        is_no_hidden_layer = len(self.hidden_layers) == 0
        if is_no_hidden_layer:
            return self.num_inputs
        else:
            return self.hidden_layers[-1].num_neurons

    def train(self, input_data: np.array, expected_output: np.array):
        print("Starting training!")
        tic = time.time()
        iteration_outputs = []
        stept = []

        for i in range(self.iterations):
            # FIXME: should add the validation set to the stopping condition
            if self.stopping_condition.should_stop(self, input_data, expected_output):
                print("Stopping model training early")
                break

            # FIXME!
            output = self.predict(input_data)
            # FIXME check the expected value type: should be a np.array (check the case when we have single value)
            bias_grads, weight_grads = self.back_propagation(output, expected_output)
            self.update_biases(bias_grads, self.learning_rate)
            self.update_weight(weight_grads, self.learning_rate)

            if i % 100 == 0 and i != 0:
                output = self.predict(input_data)
                error = self.cost_function.calculate(output, expected_output)
                iteration_outputs.append(np.max(error))
                stept.append(i)

                print(i, error * 100)

        toc = time.time()
        fig, ax = plt.subplots()
        ax.plot(stept, iteration_outputs)
        ax.set_xlabel('iterations')
        ax.set_ylabel('error')
        ax.set_title('Xor')
        plt.show()

        print(f"Training finished! {toc - tic}")

    def predict(self, x_input: np.array) -> np.array:
        for layer in self.hidden_layers:
            x_input = layer.forward(x_input)
        return x_input

    def back_propagation(self, result, expected) -> Tuple[List[np.array], List[np.array]]:
        bias = []
        weight = []

        gradient = self.cost_function.calculate_gradient(result, expected)

        for layer in self.hidden_layers[::-1]:
            z = layer.last_output
            zz = layer.activationFunction.calculate(z)

            # element-wise multiplication
            gradient = np.multiply(gradient, layer.activationFunction.calculate_gradient(zz))

            # np.sum(gradient, axis=1, keepdims=True)
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
