import pickle
from abc import ABC, abstractmethod

import numpy as np

from src.NeuronalNetwork import NeuronalNetwork


class ExampleTemplate(ABC):
    expected_output = None
    output_neurons = None
    input_neurons = None
    training_data = None
    learning_rate = None
    cost_function = None
    raining_data = None
    neural_net = None
    iterations = None
    architecture = []
    experiment_name = None

    def __init__(self, experiment_name):
        self.experiment_name = experiment_name

    def build_neuronal_net(self):
        self.neural_net = NeuronalNetwork(self.input_neurons, self.architecture, self.output_neurons,
                                          self.cost_function)

    def train_and_save_nn(self):
        self.neural_net.train(self.training_data, self.expected_output, self.learning_rate, self.iterations)
        with open(f'{self.experiment_name}-{self.input_neurons}-{len(self.architecture)}-{self.output_neurons}.nn',
                  'wb') as file:
            pickle.dump(self.neural_net, file)

    def load_net_from_file(self):
        with open(f'{self.experiment_name}-{self.input_neurons}-{len(self.architecture)}-{self.output_neurons}.nn',
                  'rb') as file:
            self.neural_net = pickle.load(file)

    @staticmethod
    def get_column(matrix: np.array, index: int) -> np.array:
        return matrix[:, [index]]

    @staticmethod
    def get_columns_between_a_and_b(matrix: np.array, a: int, b: int):
        return matrix[:, a:b + 1]

    @abstractmethod
    def get_data(self) -> np.array:
        pass

    @abstractmethod
    def define_data(self):
        pass

    @abstractmethod
    def define_architecture(self):
        pass

    @abstractmethod
    def define_training_hyperparameters(self):
        pass

    @abstractmethod
    def run_tests(self):
        pass

    def run(self):
        # Config NN
        self.define_data()
        self.define_architecture()
        self.define_training_hyperparameters()

        # Build NN
        self.build_neuronal_net()

        # Recover a backup
        self.try_to_recover_old_training()

        # Run tests
        self.run_tests()

    def try_to_recover_old_training(self):
        try:
            self.load_net_from_file()
        except:
            self.train_and_save_nn()
