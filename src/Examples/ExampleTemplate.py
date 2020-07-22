import pickle
from abc import ABC, abstractmethod

import numpy as np

from src.BatchFunctions import BatchFunction
from src.Networks.NeuralNetwork import NeuralNetwork
from src.Optimizers.Optimizers import SGD


class ExampleTemplate(ABC):

    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.expected_output = None
        self.training_data = None
        self.learning_rate = None
        self.cost_function = None
        self.neural_net = None
        self.iterations = None
        self.optimizer = SGD()
        self.architecture = []
        self.batch_function: BatchFunction = None
        self.metrics = []
        self.callbacks = []

    def build_neuronal_net(self):
        self.neural_net = NeuralNetwork(self.architecture,
                                        self.cost_function,
                                        self.learning_rate,
                                        self.iterations,
                                        optimizer=self.optimizer,
                                        metrics=self.metrics,
                                        callbacks=self.callbacks)

    def train_and_save_nn(self):
        self.neural_net.train(self.training_data, self.expected_output, self.batch_function)
        with open(f'{self.neural_net}.nn', 'wb') as file:
            pickle.dump(self.neural_net, file)

    def load_net_from_file(self):
        with open(f'{self.neural_net}.nn', 'rb') as file:
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
        self.train_and_save_nn()
        # try:
        #     pass
        #     # self.load_net_from_file()
        # except:
