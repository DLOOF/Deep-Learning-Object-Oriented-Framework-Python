from abc import ABC, abstractmethod
import numpy as np
from src.ActivationFunctions.ActivationFunction import ActivationFunction


class RegularizationActivationFunction(ActivationFunction):

    def __init__(self, activation_function: ActivationFunction):
        assert activation_function is not None
        assert isinstance(activation_function, ActivationFunction)
        self.activation_function = activation_function
        super().__init__()

    @abstractmethod
    def regularize(self, value: np.array) -> np.array:
        pass

    @abstractmethod
    def regularization_gradient(self, value: np.array) -> np.array:
        pass

    def calculate_derivative(self, value: np.array) -> np.array:
        return self.regularization_gradient(value)

    def calculate(self, value: np.array) -> np.array:
        return self.regularize(value)


class Dropout(RegularizationActivationFunction):

    """
    Dropout (Hinton et. al. 2012) regularization technique
    Each element of a layer’s output is kept with probability p, otherwise being set to 0 with probability (1 − p)

    constructor:
        p: the probability of keeping a
        activation_function: The activation function layer to regularize
    """
    def __init__(self, p: float, activation_function: ActivationFunction):
        assert 0.0 <= p <= 1.0, "a probability should be in [0, 1], got %f" % p
        super().__init__(activation_function)
        self.p = p
        self.previous_mask = None

    def regularize(self, value: np.array) -> np.array:
        assert value.ndim == 1, "must be a unidimensional vector: %d dimensions" % value.ndim
        self.previous_mask = np.random.binomial(n=1, p=self.p, size=value.shape)
        return np.multiply(self.previous_mask, value)

    def regularization_gradient(self, value: np.array) -> np.array:
        assert value.ndim == 1, "must be a unidimensional vector: %d dimensions" % value.ndim
        return np.multiply(self.previous_mask, self.activation_function.calculate_derivative(value))
