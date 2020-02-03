from abc import ABC, abstractmethod
import math


# TODO: check this: https://stackoverflow.com/questions/42594695/how-to-apply-a-function-map-values-of-each-element
#  -in-a-2d-numpy-array-matrix
# and this https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.expit.html

class ActivationFunction(ABC):

    @abstractmethod
    def calculate(self, value: float) -> float:
        pass

    @abstractmethod
    def calculate_derivative(self, value: float) -> float:
        pass


class Relu(ActivationFunction):

    def calculate_derivative(self, value: float) -> float:
        return 0 if value <= 0 else 1

    def calculate(self, value: float) -> float:
        return max(0.0, value)


class Sigmoid(ActivationFunction):

    def calculate_derivative(self, value: float) -> float:
        return self.calculate(value) * (1 - self.calculate(value))

    def calculate(self, value: float) -> float:
        return 1 / (1 + math.exp(-value))


class TanH(ActivationFunction):

    def calculate_derivative(self, value: float) -> float:
        return 1 - math.tanh(value) ** 2

    def calculate(self, value: float) -> float:
        return math.tanh(value)
