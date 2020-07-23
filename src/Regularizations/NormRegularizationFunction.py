from abc import ABC, abstractmethod

import numpy as np

from src.Networks.Layer import Layer


class NormRegularizationFunction(ABC):

    def __init__(self, regularization_rate: float = 0.0):
        self.regularization_rate = regularization_rate

    @abstractmethod
    def calculate(self, layer: Layer) -> float:
        pass

    @abstractmethod
    def calculate_gradient_weights(self, layer: Layer) -> np.array:
        pass

    @abstractmethod
    def calculate_gradient_bias(self, layer: Layer) -> np.array:
        pass


class VoidNormRegularizationFunction(NormRegularizationFunction):
    def calculate_gradient_weights(self, layer: Layer) -> np.array:
        return np.zeros(layer.weight.shape)

    def calculate_gradient_bias(self, layer: Layer) -> np.array:
        return np.zeros(layer.bias.shape)

    def calculate(self, layer: Layer) -> float:
        return 0.0


class L2WeightDecay(NormRegularizationFunction):
    def calculate_gradient_weights(self, layer: Layer) -> np.array:
        return layer.weight * self.regularization_rate

    def calculate_gradient_bias(self, layer: Layer) -> np.array:
        return np.zeros(layer.bias.shape)

    def calculate(self, layer: Layer) -> float:
        return layer.weight.T @ layer.weight * (0.5 * self.regularization_rate)
