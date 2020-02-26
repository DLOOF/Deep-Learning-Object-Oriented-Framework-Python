from abc import ABC, abstractmethod
import numpy as np


class RegularizationFunction(ABC):

    @abstractmethod
    def calculate(self, value: np.array) -> float:
        pass

    @abstractmethod
    def calculate_gradient(self, value: np.array) -> np.array:
        pass


class L2WeightDecay(RegularizationFunction):
    def calculate_gradient(self, value: np.array) -> np.array:
        return value

    def calculate(self, value: np.array) -> float:
        assert value.ndim == 1, "must be a unidimensional vector: %d dimensions" % value.ndim
        return np.dot(value.T, value) * 0.5
