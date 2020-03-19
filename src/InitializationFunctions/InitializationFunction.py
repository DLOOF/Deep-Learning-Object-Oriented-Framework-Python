from abc import ABC, abstractmethod
import numpy as np


class InitializationFunction(ABC):

    @abstractmethod
    def initialize(self, x, y):
        pass


class Random(InitializationFunction):

    def initialize(self, x, y):
        return np.random.randn(x, y)


class He(InitializationFunction):
    def initialize(self, x, y):
        return np.random.randn(x, y) * np.sqrt(2 / y)


class Xavier(InitializationFunction):

    def initialize(self, x, y):
        return np.random.randn(x, y) * np.sqrt(1 / y)


class Other(InitializationFunction):

    def initialize(self, x, y):
        return np.random.randn(x, y) * np.sqrt(2 / (x + y))
