import time

import matplotlib.pyplot as plt
import numpy as np

from src.BatchFunctions.BatchFunction import BatchMode, BatchFunction
from src.CostFunctions import CostFunction
from src.Networks.SupervisedModel import SupervisedModel
from src.Optimizers.Optimizers import Optimizer, SGD
from src.Regularizations.EarlyStoppingRegularization import StoppingCondition, VoidStoppingCondition
from src.Regularizations.NormRegularizationFunction import NormRegularizationFunction, L2WeightDecay


class NeuralNetwork(SupervisedModel):

    def __init__(self,
                 layers,
                 cost_function: CostFunction,
                 learning_rate: float = 0.001,
                 epochs: int = 1000,
                 stopping_condition: StoppingCondition = VoidStoppingCondition(),
                 regularization_function: NormRegularizationFunction = L2WeightDecay(0.01),
                 optimizer: Optimizer = SGD()):

        self.layers = layers
        self.cost_function = cost_function
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.stopping_condition = stopping_condition
        self.regularization_function = regularization_function
        self.optimizer = optimizer
        self._add_optimizer()

    def _add_optimizer(self):
        for layer in self.layers:
            layer.add_optimizer(self.optimizer.copy_instance())

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
                self.back_propagation(output, batch_expected)

            if i % 1 == 0 and i != 0:
                regularization_penalty = 0.0
                for layer in self.layers:
                    regularization_penalty += np.sum(self.regularization_function.calculate(layer))

                output = self.predict(input_data)
                output_real = np.argmax(output, axis=1)  # only on classification, not on prediction
                expected_real = np.argmax(expected_output, axis=1)
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
        for layer in self.layers:
            x_input = layer.forward(x_input)
        return x_input

    def back_propagation(self, result, expected):
        gradient = self.cost_function.calculate_gradient(result, expected)
        for layer in self.layers[::-1]:
            gradient = layer.backward(gradient, self.learning_rate, self.regularization_function)

    def __str__(self):
        string = ""
        for layer in self.layers:
            string += str(layer) + "-"
        string += f"->{self.learning_rate}"
        return string
