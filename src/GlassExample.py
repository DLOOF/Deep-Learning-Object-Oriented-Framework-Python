import pickle
from abc import ABC, abstractmethod
import numpy as np

from src.ActivationFunction import Relu
from src.CostFunction import MeanSquaredError
from src.ExampleTemplate import ExampleTemplate


class GlassExample(ExampleTemplate):
    def get_data(self) -> np.array:
        return np.genfromtxt("../datasets/glass.csv", delimiter=",", skip_header=1)

    def define_data(self):
        self.training_data = self.get_data()[:, 0:-1]
        self.expected_output = self.get_data()[:, -1]

        training_samples = int(self.get_data().shape[0] * .8)
        self.training_data, self.expected_output = self.training_data[:training_samples, :], self.expected_output[:training_samples, :]

        self.expected_output = self.expected_output.astype(int)
        tmp = np.zeros((self.expected_output.size, self.expected_output.max()))
        tmp[np.arange(self.expected_output.size), self.expected_output - 1] = 1
        self.expected_output = tmp

    def define_architecture(self):
        self.input_neurons = 9
        self.output_neurons = 7
        layer_1 = (9, Relu())
        layer_2 = (9, Relu())
        layer_3 = (9, Relu())
        layer_4 = (9, Relu())
        self.architecture.append(layer_1)
        self.architecture.append(layer_2)
        self.architecture.append(layer_3)
        self.architecture.append(layer_4)
        self.cost_function = MeanSquaredError()

    def define_training_parameters(self):
        self.learning_rate = 0.1
        self.iterations = 10_0

    def run_tests(self):
        print(
            f"[1.51711,14.23,0,2.08,73.36,0,8.62,1.67,0,7] = {self.neuronal_net.forward(np.array([1.51711, 14.23, 0, 2.08, 73.36, 0, 8.62, 1.67, 0]).reshape(-1, 1))}")


if __name__ == '__main__':
    example = GlassExample("glass_example")
    example.run()
