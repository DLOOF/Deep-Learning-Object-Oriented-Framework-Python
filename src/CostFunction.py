from abc import ABC, abstractmethod
import numpy as np

# TODO check this: https://stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks
#  -alongside-applications


class CostFunction(ABC):

    @abstractmethod
    def calculate(self, value: np.ndarray, expected_value: np.ndarray) -> np.ndarray:
        pass


class DistanceFunction(CostFunction):

    def calculate(self, value: np.ndarray, expected_value: np.ndarray) -> np.ndarray:
        return value - expected_value


class DistanceAbsFunction(CostFunction):

    def calculate(self, value: np.ndarray, expected_value: np.ndarray) -> np.ndarray:
        return np.abs(value - expected_value)


class DistanceQuadraticFunction(CostFunction):

    def calculate(self, value: np.ndarray, expected_value: np.ndarray) -> np.ndarray:
        return (value - expected_value) ** 2
