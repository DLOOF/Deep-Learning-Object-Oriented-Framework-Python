import numpy as np
from abc import ABC, abstractmethod
from src.ActivationFunctions.ActivationFunction import *


class Layer(ABC):

    @abstractmethod
    def forward(self, x_input: np.array) -> np.array:
        pass

    @abstractmethod
    def update_weight(self, learning_rate: float, grads: np.array):
        pass

    @abstractmethod
    def update_bias(self, learning_rate: float, grads: np.array):
        pass
