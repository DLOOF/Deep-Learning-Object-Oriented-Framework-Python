from abc import ABC, abstractmethod

import numpy as np


# TODO check this: https://stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks
#  -alongside-applications


class CostFunction(ABC):

    @abstractmethod
    def calculate(self, value: np.array, expected_value: np.array) -> np.array:
        pass

    @abstractmethod
    def calculate_gradient(self, value: np.array, expected_value: np.array) -> np.array:
        pass


class MeanAbsoluteError(CostFunction):

    #FIXME
    def calculate(self, value: np.array, expected_value: np.array) -> np.array:
        return np.sum(np.absolute(value - expected_value))

    def calculate_gradient(self, value: np.array, expected_value: np.array) -> np.array:
        return np.vectorize(lambda x: -1 if x > 0 else 1)(value - expected_value)


class MeanSquaredError(CostFunction):

    def calculate(self, value: np.array, expected_value: np.array) -> np.array:
        sqr = value - expected_value
        sqr = np.linalg.norm(sqr, 2)**2
        sqr /= 2.0
        return sqr

    def calculate_gradient(self, value: np.array, expected_value: np.array) -> np.array:
        return value - expected_value


class CrossEntropy(CostFunction):

    def calculate(self, value: np.array, expected_value: np.array) -> float:
        # FIXME needs to be fixed this method has to return a scalar rather than a vector
        # FIXME it depends whether if it is calculating for multiple classes instead of binary classes
        assert value.ndim == expected_value.ndim and expected_value.ndim == 1
        raise NotImplementedError("This method needs to be fixed")
        return - (expected_value * np.log(value) + (1 - expected_value) * np.log(1 - value))

    def calculate_gradient(self, value: np.array, expected_value: np.array) -> np.array:
        return np.divide(value - expected_value, np.multiply(1 - value, value))
