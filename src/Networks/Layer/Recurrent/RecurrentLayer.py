import numpy as np

from src.ActivationFunctions.ActivationFunction import TanH, ActivationFunction
from src.InitializationFunctions.InitializationFunction import InitializationFunction, Random
from src.Networks.Layer.Layer import Layer
from src.Regularizations import NormRegularizationFunction


class ConvolutionLayer(Layer):
    # FIXME: In this moment this class is a convolution filter 3x3

    def __init__(self, num_neurons, num_output,
                 activation_function: ActivationFunction = TanH,
                 initialization_function: InitializationFunction = Random()):
        super().__init__()
        self.num_neurons = num_neurons
        self.num_output = num_output
        self.initialization_function = initialization_function
        self.activation_function = activation_function
        self.last_inputs = None

        self.w = initialization_function.initialize(num_neurons, num_neurons)
        self.u = None
        self.v = initialization_function.initialize(num_output, num_neurons)

        self.b = np.zeros((num_neurons, 1))
        self.c = np.zeros((num_output, 1))

        self.hs = []

    def forward(self, x_input: np.array) -> np.array:
        self.__init_weight_late__(x_input)
        self.last_inputs = x_input

        h = np.zeros((self.num_neurons, 1))
        self.hs.append(h)

        for x in x_input:
            h = self.activation_function.calculate(self.u @ x + self.w @ h + self.b)
            self.hs.append(h)

        y = self.v @ h + self.c

        return y

    def backward(self, gradient: np.array, learning_rate: float,
                 regularization_function: NormRegularizationFunction) -> np.array:
        n = len(self.last_inputs)

        dv = gradient @ self.hs[n]

        dw = np.zeros(self.w.shape)
        du = np.zeros(self.u.shape)

        db = np.zeros(self.b.shape)
        dc = gradient

        dh = self.v @ gradient

        for i, h in enumerate(self.hs[::-1]):
            temp = self.activation_function.calculate_gradient(self.hs[i + 1]) * dh

            db += temp

            dw += temp @ h.T
            du += temp @ self.last_inputs[i].T

            dh = self.w @ temp

        for d in [du, dw, dv, db, dc]:
            np.clip(d, -1, 1, out=d)

        self.w += - learning_rate * dw
        self.u += - learning_rate * du
        self.v += - learning_rate * dv

        self.b += - learning_rate * db
        self.c += - learning_rate * dc

    def update_weight(self, learning_rate: float, grads: np.array):
        pass

    def update_bias(self, learning_rate: float, grads: np.array):
        pass

    def __str__(self):
        return f"rnn-{self.num_neurons}"

    def __init_weight_late__(self, x_input):
        if self.u is None:
            self.u = self.initialization_function.initialize(self.num_neurons, x_input.shape[0])
