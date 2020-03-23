from abc import ABC, abstractmethod

import numpy as np


# TODO: check this: https://stackoverflow.com/questions/42594695/how-to-apply-a-function-map-values-of-each-element
#  -in-a-2d-numpy-array-matrix
# and this https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.expit.html

class ActivationFunction(ABC):

    def operation(self):
        return np.multiply

    @abstractmethod
    def calculate(self, value: np.array) -> np.array:
        pass

    @abstractmethod
    def calculate_gradient(self, value: np.array) -> np.array:
        pass


class Relu(ActivationFunction):

    def calculate_gradient(self, value: np.array) -> np.array:
        return np.vectorize(lambda x: 0 if x <= 0 else 1)(value)

    def calculate(self, value: np.array) -> np.array:
        return np.vectorize(lambda x: max(0.0, x))(value)

    def __str__(self):
        return "R"


class Sigmoid(ActivationFunction):

    def calculate_gradient(self, value: np.array) -> np.array:
        return self.calculate(value) * (1 - self.calculate(value))

    def calculate(self, value: np.array) -> np.array:
        return 1 / (1 + np.exp(-value))

    def __str__(self):
        return "S"


class TanH(ActivationFunction):

    def calculate_gradient(self, value: np.array) -> np.array:
        return 1 - np.tanh(value) ** 2

    def calculate(self, value: np.array) -> np.array:
        return np.tanh(value)

    def __str__(self):
        return "T"


class SoftMax(ActivationFunction):

    def __init__(self):
        self.last_output = None
        self.last_input = None

    def operation(self):
        return lambda x, y: np.einsum('ijk,ik->ij', x, y.T).T

    def calculate_gradient(self, da: np.array) -> np.array:
        # z, da shapes - (m, n)
        z = self.last_input
        m, n = z.T.shape
        p = self.calculate(z.T)
        # First we create for each example feature vector, it's outer product with itself
        # ( p1^2  p1*p2  p1*p3 .... )
        # ( p2*p1 p2^2   p2*p3 .... )
        # ( ...                     )
        tensor1 = np.einsum('ij,ik->ijk', p, p)  # (m, n, n)
        # Second we need to create an (n,n) identity of the feature vector
        # ( p1  0  0  ...  )
        # ( 0   p2 0  ...  )
        # ( ...            )
        tensor2 = np.einsum('ij,jk->ijk', p, np.eye(n, n))  # (m, n, n)
        # Then we need to subtract the first tensor from the second
        # ( p1 - p1^2   -p1*p2   -p1*p3  ... )
        # ( -p1*p2     p2 - p2^2   -p2*p3 ...)
        # ( ...                              )
        dSoftmax = tensor2 - tensor1
        # Finally, we multiply the dSoftmax (da/dz) by da (dL/da) to get the gradient w.r.t. Z
        # dz = np.einsum('ijk,ik->ij', dSoftmax, da.T)  # (m, n)
        return dSoftmax

    def calculate(self, z: np.array) -> np.array:
        self.last_input = z
        e = np.exp(z - np.max(z))
        s = np.sum(e, axis=1, keepdims=True)
        self.last_output = e / s
        return self.last_output
