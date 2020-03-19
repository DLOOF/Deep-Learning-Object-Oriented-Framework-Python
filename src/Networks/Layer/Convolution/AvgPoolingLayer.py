from src.Networks.Layer.Layer import Layer
from src.ActivationFunctions.ActivationFunction import *


class AvgPoolingLayer(Layer):

    def forward(self, x_input: np.array) -> np.array:
        pass

    def update_weight(self, learning_rate: float, grads: np.array):
        pass

    def update_bias(self, learning_rate: float, grads: np.array):
        pass
