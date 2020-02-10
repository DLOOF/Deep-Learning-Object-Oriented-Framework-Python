import pickle
from abc import ABC, abstractmethod
import numpy as np

from src.ActivationFunction import Relu
from src.CostFunction import MeanSquaredError
from src.ExampleTemplate import ExampleTemplate
from src.NeuronalNetwork import NeuronalNetwork


def calculate_error(neural_net: NeuronalNetwork, out: np.array, expected: np.array):
    return neural_net.cost_function.calculate(out.reshape(-1, 1), expected.reshape(-1, 1)) * 100


class GlassExample(ExampleTemplate):
    def get_data(self) -> np.array:
        return np.genfromtxt("../datasets/glass.csv", delimiter=",", skip_header=1)

    def define_data(self):
        self.input_data = self.get_data()[:, 0:-1]
        self.output_data = self.get_data()[:, -1]
        output = self.output_data.astype(int)
        tmp = np.zeros((output.size, output.max()))
        tmp[np.arange(output.size), output - 1] = 1
        self.output_data = tmp

        self.training_samples = int(self.input_data.shape[0] * .8)
        self.training_data, self.expected_output = self.input_data[:self.training_samples, :], self.output_data[:self.training_samples, :]

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
        # self.architecture.append(layer_4)
        self.cost_function = MeanSquaredError()

    def define_training_hyperparameters(self):
        self.learning_rate = 0.1
        self.iterations = 10_000

    def run_tests(self):
        for sample_in, sample_out in zip(self.input_data[self.training_samples:, :], self.output_data[self.training_samples:, :]):
            output = self.neural_net.forward(sample_in.reshape(-1, 1))
            print(f"[{sample_in}] = {output.T} ({calculate_error(self.neural_net, output, sample_out).T})")


if __name__ == '__main__':
    example = GlassExample("glass_example")
    example.run()
