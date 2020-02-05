from abc import ABC, abstractmethod
import numpy as np


# TODO check this: https://stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks
#  -alongside-applications


class CostFunction(ABC):

    @abstractmethod
    def calculate(self, value: np.array, expected_value: np.array) -> np.array:
        pass

    @abstractmethod
    def calculate_derivative(self, value: np.array, expected_value: np.array) -> np.array:
        pass


class MeanAbsoluteError(CostFunction):

    def calculate(self, value: np.array, expected_value: np.array) -> np.array:
        return np.abs(value - expected_value)

    def calculate_derivative(self, value: np.array, expected_value: np.array) -> np.array:
        return np.vectorize(lambda x: -1 if x > 0 else 1)(value - expected_value)


class MeanSquaredError(CostFunction):

    def calculate(self, value: np.array, expected_value: np.array) -> np.array:
        return np.power(value - expected_value, 2) / 2

    def calculate_derivative(self, value: np.array, expected_value: np.array) -> np.array:
        return value - expected_value


class CrossEntropy(CostFunction):

    def calculate(self, value: np.array, expected_value: np.array) -> np.array:
        return - (expected_value * np.log(value) + (1 - expected_value) * np.log(1 - value))

    def calculate_derivative(self, value: np.array, expected_value: np.array) -> np.array:
        return np.divide(value - expected_value, np.multiply(1 - value, value))
