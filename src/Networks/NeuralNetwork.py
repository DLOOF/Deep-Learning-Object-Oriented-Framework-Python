import time

import numpy as np

from src.BatchFunctions.BatchFunction import BatchMode, BatchFunction
from src.Callbacks.Callback import Callback
from src.CostFunctions import CostFunction
from src.Metrics import Metrics
from src.Networks.SupervisedModel import SupervisedModel
from src.Optimizers.Optimizers import Optimizer, SGD
from src.Regularizations.NormRegularizationFunction import NormRegularizationFunction, \
    VoidNormRegularizationFunction


class NeuralNetwork(SupervisedModel):

    def __init__(self,
                 layers,
                 cost_function: CostFunction,
                 learning_rate: float = 0.001,
                 epochs: int = 1000,
                 callbacks: [Callback] = [],
                 regularization_function: NormRegularizationFunction = VoidNormRegularizationFunction(),
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

    def __validate_input(input_data: np.array, output_data: np.array):
        assert input_data is not None and output_data is not None, f"input and output should not be None"
        assert len(input_data.shape) >= 1 and len(output_data.shape) >= 1, f"input {input_data.shape} or output {output_data.shape} is " \
                                                                 f"invalid "
        assert input_data.shape[0] == output_data.shape[0], f"batch dimension should be equal in input ({input_data.shape[0]}) and " \
                                                  f"output ({output_data.shape[0]}) "

    def train(self, input_data: np.array,
              expected_output: np.array,
              batch_function: BatchFunction = None,
              validation_data: [np.array] = None):

        NeuralNetwork.__validate_input(input_data, expected_output)

        x_val, y_val = None, None
        x_val, y_val = self.validate_validation_data(validation_data, x_val, y_val)

        # ---------------------------------------

        print("Starting training!")
        tic = time.time()

        epochs_metrics = {str(x): [] for x in self.metrics}
        self.init_validation_metrics(epochs_metrics, x_val)

        # by default use batch mode (e.g. the whole dataset at once)
        if batch_function is None:
            batch_function = BatchMode(input_data, expected_output)

        for epoch in range(self.epochs):
            for j, (batch_input, batch_expected) in enumerate(batch_function.get_batch()):
                output = self.predict(batch_input)
                self._back_propagation(output, batch_expected)

                # for metric in self.metrics:
                #     print(f"{metric}: {metric.calculate(output, batch_expected)}")

            self.register_epoch_metrics(epochs_metrics, expected_output, epoch, input_data, x_val, y_val)

            for c in self.callbacks:
                c.call(epochs_metrics)

        toc = time.time()
        print(f"Training finished! {toc - tic}")

    def register_epoch_metrics(self, epochs_metrics, expected_output, i, input_data, x_val, y_val):
        for metric in self.metrics:
            result = metric.calculate(self.predict(input_data), expected_output)
            epoch_str = f"EPOCH {i} ---- {metric}: {result:,}"

            if x_val is not None:
                result_val = metric.calculate(self.predict(x_val), y_val)
                epoch_str += f" val_{metric}: {result_val:,}"
                epochs_metrics[f"val_{metric}"].append(result_val)

            print(epoch_str)
            epochs_metrics[str(metric)].append(result)

    def init_validation_metrics(self, epochs_metrics, x_val):
        if x_val is not None:
            for metric in self.metrics:
                epochs_metrics[f"val_{metric}"] = []

    def validate_validation_data(self, validation_data, x_val, y_val):
        if validation_data is not None:
            assert len(validation_data) == 2, f"validation data is invalid, should be a tuple (x, y)"
            x_val, y_val = validation_data
            NeuralNetwork.__validate_input(x_val, y_val)
        return x_val, y_val

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
