from abc import ABC, abstractmethod

import numpy as np


class Metric(ABC):

    def __init__(self):
        self.name = None

    @abstractmethod
    def calculate(self, predicted: np.array, expected: np.array) -> float:
        pass

    def __str__(self):
        return self.name


class MseMetric(Metric):

    def __init__(self):
        super().__init__()
        self.name = "mse"

    def calculate(self, predicted: np.array, expected: np.array) -> float:
        return np.square(predicted - expected).mean()


class MaeMetric(Metric):

    def __init__(self):
        super().__init__()
        self.name = "mae"

    def calculate(self, predicted: np.array, expected: np.array) -> float:
        return np.abs(predicted - expected).mean()
