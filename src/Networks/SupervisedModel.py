from abc import ABC, abstractmethod

import numpy as np


class SupervisedModel(ABC):

    @abstractmethod
    def train(self, input_data: np.array, expected_output: np.array):
        pass

    @abstractmethod
    def predict(self, input_data: np.array):
        pass