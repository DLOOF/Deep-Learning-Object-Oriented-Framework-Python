import numpy as np
from abc import ABC, abstractmethod
from typing import List

from src.Networks.SupervisedModel import SupervisedModel


class BaggingRegularization(SupervisedModel):

    def __init__(self, models: List[SupervisedModel]):
        super().__init__()
        self.models = models

    @abstractmethod
    def train(self, input_data: np.array, expected_output: np.array):
        pass

    @abstractmethod
    def predict(self, input_data: np.array) -> np.array:
        pass


class SequentialBaggingRegularization(BaggingRegularization):

    def train(self, input_data: np.array, expected_output: np.array):
        for model in self.models:
            model.train(input_data, expected_output)

    def predict(self, input_data: np.array) -> np.array:
        outputs = []
        for model in self.models:
            out = model.predict(input_data)
            outputs.append(out)

        average = np.average(outputs, axis=0)
        return average
