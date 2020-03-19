from abc import ABC, abstractmethod

import numpy as np

from src.Regularizations import NormRegularizationFunction


class Layer(ABC):

    def __init__(self):
        self.bias = None
        self.weight = None
        self.last_input = None
        self.last_output = None
        self.last_input_shape = None
        self.last_activation_output = None

    @abstractmethod
    def forward(self, x_input: np.array) -> np.array:
        pass

    @abstractmethod
    def backward(self, gradient: np.array, learning_rate: float,
                 regularization_function: NormRegularizationFunction) -> np.array:
        pass

    @abstractmethod
    def update_weight(self, learning_rate: float, grads: np.array):
        pass

    @abstractmethod
    def update_bias(self, learning_rate: float, grads: np.array):
        pass
