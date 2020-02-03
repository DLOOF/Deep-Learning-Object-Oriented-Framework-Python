from abc import ABC, abstractmethod
from math import fabs


class CostFunction(ABC):

    @abstractmethod
    def calculate(self, value: float, expected_value: float) -> float:
        pass


class DistanceFunction(CostFunction):

    def calculate(self, value: float, expected_value: float) -> float:
        return value - expected_value


class DistanceAbsFunction(CostFunction):

    def calculate(self, value: float, expected_value: float) -> float:
        return fabs(value - expected_value)


class DistanceQuadraticFunction(CostFunction):

    def calculate(self, value: float, expected_value: float) -> float:
        return (value - expected_value) ** 2
