import numpy as np
from abc import ABC, abstractmethod


class SupervisedModel(ABC):

    @abstractmethod
    def train(self, input_data: np.array, expected_output: np.array):
        pass

    @abstractmethod
    def predict(self, input_data: np.array):
        pass