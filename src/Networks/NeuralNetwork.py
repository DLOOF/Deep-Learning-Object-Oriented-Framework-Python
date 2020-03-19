import time
from typing import List, Tuple

from src.BatchFunctions import BatchFunction
from src.BatchFunctions.BatchFunction import BatchMode
from src.CostFunctions import CostFunction
from src.ActivationFunctions.ActivationFunction import *
import matplotlib.pyplot as plt

from src.Networks.Layer import Layer
from src.Networks.SupervisedModel import SupervisedModel
from src.Regularizations.NormRegularizationFunction import NormRegularizationFunction, VoidNormRegularizationFunction, L2WeightDecay
from src.Regularizations.EarlyStoppingRegularization import StoppingCondition, VoidStoppingCondition


class NeuralNetwork(SupervisedModel):

    def __init__(self,
                 num_inputs: int,
                 hidden_architecture: List[Tuple[int, ActivationFunction]],
                 num_output: int,
                 cost_function: CostFunction,
                 learning_rate: float = 0.001,
                 epochs: int = 1000,
                 stopping_condition: StoppingCondition = VoidStoppingCondition(),
                 output_activation_function: ActivationFunction = Sigmoid(),
                 regularization_rate: float = 0.00,
                 regularization_function: NormRegularizationFunction = L2WeightDecay()):
        self.hidden_layers = []
        self.num_output = num_output
        self.num_inputs = num_inputs
        self.cost_function = cost_function
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.stopping_condition = stopping_condition
        self.regularization_rate = regularization_rate
        self.regularization_function = regularization_function

        for layer in hidden_architecture:
            self.__add_layer(layer[0], layer[1])
        self.__add_layer(num_output, output_activation_function)

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

    def train(self, input_data: np.array, expected_output: np.array, batch_function: BatchFunction = None):
        print("Starting training!")
        tic = time.time()
        iteration_outputs = []
        stept = []

        # by default use batch mode (e.g. the whole dataset at once)
        if batch_function is None:
            batch_function = BatchMode(input_data, expected_output)

        for i in range(self.epochs):
            # FIXME: should add the validation set to the stopping condition
            if self.stopping_condition.should_stop(self, input_data, expected_output):
                print("Stopping model training early")
                break

            for batch_input, batch_expected in batch_function.get_batch():
                output = self.predict(batch_input)
                # FIXME check the expected value type: should be a np.array (check the case when we have single value)
                bias_grads, weight_grads = self.back_propagation(output, batch_expected)
                self.update_biases(bias_grads, self.learning_rate)
                self.update_weight(weight_grads, self.learning_rate)

            if i % 1 == 0 and i != 0:
                regularization_penalty = 0.0
                for layer in self.hidden_layers:
                    regularization_penalty += np.sum(self.regularization_rate \
                                              * self.regularization_function.calculate(layer))

                output = self.predict(input_data)
                output_real = np.argmax(output, axis=0) # only on classification, not on prediction
                expected_real = np.argmax(expected_output, axis=0)
                mae = np.absolute(output_real - expected_real).mean()
                mse = np.power(output_real - expected_real, 2.0).mean()
                accuracy = np.sum(output_real == expected_real) / output_real.size
                loss = self.cost_function.calculate(output, expected_output)

                iteration_outputs.append(accuracy)
                stept.append(i)

                print("%d - mae %.3f%% mse %.3f%% loss %.3f acc %.3f" % (i, mae, mse, loss, accuracy))
                # print("Overall accuracy: %.3f%%" % (accuracy))

        toc = time.time()
        fig, ax = plt.subplots()
        ax.plot(stept, iteration_outputs)
        ax.set_xlabel('iterations')
        ax.set_ylabel('error')
        ax.set_title('Training error')
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

            bias.append(gradient
                        + self.regularization_rate * self.regularization_function.calculate_gradient_bias(layer))
            weight.append(np.dot(gradient, layer.last_input.T)
                          + self.regularization_rate * self.regularization_function.calculate_gradient_weights(layer))

            gradient = np.dot(layer.weight.T, gradient)

        return bias, weight

    def update_biases(self, bias_grads, learning_rate):
        for i, layer in enumerate(self.hidden_layers[::-1]):
            layer.update_bias(learning_rate, bias_grads[i])

    def update_weight(self, weight_grads, learning_rate):
        for i, layer in enumerate(self.hidden_layers[::-1]):
            layer.update_weight(learning_rate, weight_grads[i])
