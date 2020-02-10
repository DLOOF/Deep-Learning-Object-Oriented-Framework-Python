from abc import ABC, abstractmethod
import numpy as np


# TODO: check this: https://stackoverflow.com/questions/42594695/how-to-apply-a-function-map-values-of-each-element
#  -in-a-2d-numpy-array-matrix
# and this https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.expit.html

class ActivationFunction(ABC):

    @abstractmethod
    def calculate(self, value: np.array) -> np.array:
        pass

    @abstractmethod
    def calculate_derivative(self, value: np.array) -> np.array:
        pass


class Relu(ActivationFunction):

    def calculate_derivative(self, value: np.array) -> np.array:
        return np.vectorize(lambda x: 0 if x <= 0 else 1)(value)

    def calculate(self, value: np.array) -> np.array:
        return np.vectorize(lambda x: max(0.0, x))(value)


class Sigmoid(ActivationFunction):

    def calculate_derivative(self, value: np.array) -> np.array:
        return self.calculate(value) * (1 - self.calculate(value))

    def calculate(self, value: np.array) -> np.array:
        return 1 / (1 + np.exp(-value))


class TanH(ActivationFunction):

    def calculate_derivative(self, value: np.array) -> np.array:
        return 1 - np.tanh(value) ** 2

    def calculate(self, value: np.array) -> np.array:
        return np.tanh(value)
