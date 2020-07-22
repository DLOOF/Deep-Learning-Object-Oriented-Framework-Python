import time

import numpy as np

from src.BatchFunctions.BatchFunction import BatchMode, BatchFunction
from src.Callbacks.Callback import Callback
from src.CostFunctions import CostFunction
from src.Metrics import Metrics
from src.Networks.SupervisedModel import SupervisedModel
from src.Optimizers.Optimizers import Optimizer, SGD
from src.Regularizations.NormRegularizationFunction import NormRegularizationFunction, L2WeightDecay


class NeuralNetwork(SupervisedModel):

    def __init__(self,
                 layers,
                 cost_function: CostFunction,
                 learning_rate: float = 0.001,
                 epochs: int = 1000,
                 callbacks: [Callback] = [],
                 regularization_function: NormRegularizationFunction = L2WeightDecay(0.01),
                 optimizer: Optimizer = SGD(),
                 metrics: [Metrics] = []
                 ):

        self.layers = layers
        self.cost_function = cost_function
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.callbacks = callbacks
        self.regularization_function = regularization_function
        self.optimizer = optimizer
        self.metrics = metrics
        self._add_optimizer()

    def _add_optimizer(self):
        for layer in self.layers:
            layer.add_optimizer(self.optimizer.copy_instance())

    def train(self, input_data: np.array, expected_output: np.array, batch_function: BatchFunction = None):
        print("Starting training!")
        tic = time.time()

        epochs_metrics = {str(x): [] for x in self.metrics}

        # by default use batch mode (e.g. the whole dataset at once)
        if batch_function is None:
            batch_function = BatchMode(input_data, expected_output)

        for i in range(self.epochs):
            for j, (batch_input, batch_expected) in enumerate(batch_function.get_batch()):
                output = self.predict(batch_input)
                self._back_propagation(output, batch_expected)

                # for metric in self.metrics:
                #     print(f"{metric}: {metric.calculate(output, batch_expected)}")

            for metric in self.metrics:
                result = metric.calculate(self.predict(input_data), expected_output)
                print(f"EPOCH {i} ---- {metric}: {result:,}")
                epochs_metrics[str(metric)].append(result)

            for c in self.callbacks:
                c.call(epochs_metrics)

        toc = time.time()
        print(f"Training finished! {toc - tic}")

    def predict(self, x_input: np.array) -> np.array:
        for layer in self.layers:
            x_input = layer.forward(x_input)
        return x_input

    def _back_propagation(self, result, expected):
        gradient = self.cost_function.calculate_gradient(result, expected)
        for layer in self.layers[::-1]:
            gradient = layer.backward(gradient, self.learning_rate, self.regularization_function)

    def __str__(self):
        string = ""
        for layer in self.layers:
            string += str(layer) + "-"
        string += f"->{self.learning_rate}"
        return string
