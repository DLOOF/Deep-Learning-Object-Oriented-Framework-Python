import pickle

from src.ActivationFunction import *
from src.CostFunction import *
from src.ExampleTemplate import ExampleTemplate
from src.NeuronalNetwork import NeuronalNetwork


class RelationExample(ExampleTemplate):

    def get_data(self) -> np.array:
        return np.array(
            [[1, 1, 0, 0, 0, 0, 0, 1],
             [1, 0, 1, 0, 0, 0, 0, 1],
             [1, 0, 0, 1, 0, 0, 1, 0],
             [1, 0, 0, 0, 1, 0, 1, 0],
             [1, 0, 0, 0, 0, 1, 1, 0],
             [0, 1, 1, 0, 0, 0, 0, 1],
             [0, 1, 0, 1, 0, 0, 1, 0],
             [0, 1, 0, 0, 1, 0, 1, 0],
             [0, 1, 0, 0, 0, 1, 1, 0],
             [0, 0, 1, 1, 0, 0, 1, 0],
             [0, 0, 1, 0, 1, 0, 1, 0],
             [0, 0, 1, 0, 0, 1, 1, 0],
             [0, 0, 0, 1, 1, 0, 0, 1],
             [0, 0, 0, 1, 0, 1, 0, 1],
             [0, 0, 0, 0, 1, 1, 0, 1]]
        )

    def define_data(self):
        self.training_data = self.get_columns_between_a_and_b(self.get_data(), 0, 5)
        self.expected_output = self.get_columns_between_a_and_b(self.get_data(), 6, 7)

    def define_architecture(self):
        self.architecture = []
        self.input_neurons = 6
        self.output_neurons = 2

        layer_1 = (8, Relu())
        layer_2 = (8, Relu())

        self.architecture.append(layer_1)
        self.architecture.append(layer_2)

        self.cost_function = MeanSquaredError()

    def define_training_parameters(self):
        self.learning_rate = 0.1
        self.iterations = 10_000

    def run_tests(self):
        print(
            f" [1, 1, 0, 0, 0, 0] ans = [0, 1] = {self.neuronal_net.forward(np.array([1, 1, 0, 0, 0, 0]).reshape(-1, 1))}")
        print(
            f" [1, 0, 1, 0, 0, 0] ans = [0, 1] = {self.neuronal_net.forward(np.array([1, 0, 1, 0, 0, 0]).reshape(-1, 1))}")
        print(
            f" [1, 0, 0, 1, 0, 0] ans = [1, 0] = {self.neuronal_net.forward(np.array([1, 0, 0, 1, 0, 0]).reshape(-1, 1))}")
        print(
            f" [1, 0, 0, 0, 1, 0] ans = [1, 0] = {self.neuronal_net.forward(np.array([1, 0, 0, 0, 1, 0]).reshape(-1, 1))}")
        print(
            f" [1, 0, 0, 0, 0, 1] ans = [1, 0] = {self.neuronal_net.forward(np.array([1, 0, 0, 0, 0, 1]).reshape(-1, 1))}")
        print(
            f" [0, 1, 1, 0, 0, 0] ans = [0, 1] = {self.neuronal_net.forward(np.array([0, 1, 1, 0, 0, 0]).reshape(-1, 1))}")
        print(
            f" [0, 1, 0, 1, 0, 0] ans = [1, 0] = {self.neuronal_net.forward(np.array([0, 1, 0, 1, 0, 0]).reshape(-1, 1))}")
        print(
            f" [0, 1, 0, 0, 1, 0] ans = [1, 0] = {self.neuronal_net.forward(np.array([0, 1, 0, 0, 1, 0]).reshape(-1, 1))}")
        print(
            f" [0, 1, 0, 0, 0, 1] ans = [1, 0] = {self.neuronal_net.forward(np.array([0, 1, 0, 0, 0, 1]).reshape(-1, 1))}")
        print(
            f" [0, 0, 1, 1, 0, 0] ans = [1, 0] = {self.neuronal_net.forward(np.array([0, 0, 1, 1, 0, 0]).reshape(-1, 1))}")
        print(
            f" [0, 0, 1, 0, 1, 0] ans = [1, 0] = {self.neuronal_net.forward(np.array([0, 0, 1, 0, 1, 0]).reshape(-1, 1))}")
        print(
            f" [0, 0, 1, 0, 0, 1] ans = [1, 0] = {self.neuronal_net.forward(np.array([0, 0, 1, 0, 0, 1]).reshape(-1, 1))}")
        print(
            f" [0, 0, 0, 1, 1, 0] ans = [0, 1] = {self.neuronal_net.forward(np.array([0, 0, 0, 1, 1, 0]).reshape(-1, 1))}")
        print(
            f" [0, 0, 0, 1, 0, 1] ans = [0, 1] = {self.neuronal_net.forward(np.array([0, 0, 0, 1, 0, 1]).reshape(-1, 1))}")
        print(
            f" [0, 0, 0, 0, 1, 1] ans = [0, 1] = {self.neuronal_net.forward(np.array([0, 0, 0, 0, 1, 1]).reshape(-1, 1))}")


if __name__ == '__main__':
    relation_example = RelationExample("relation_example")
    relation_example.run()
