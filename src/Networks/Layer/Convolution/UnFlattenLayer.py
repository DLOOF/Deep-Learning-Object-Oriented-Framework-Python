import numpy as np

from src.Networks.Layer.Layer import Layer
from src.Regularizations import NormRegularizationFunction


class UnFlattenLayer(Layer):

    def __init__(self, width, height):
        super().__init__()
        self.width = width
        self.height = height

    def forward(self, x_input: np.array) -> np.array:
        return x_input.reshape(self.width, self.height, -1)

    def backward(self, gradient: np.array, learning_rate: float,
                 regularization_function: NormRegularizationFunction) -> np.array:
        return gradient

    def update_weight(self, learning_rate: float, grads: np.array):
        pass

    def update_bias(self, learning_rate: float, grads: np.array):
        pass
