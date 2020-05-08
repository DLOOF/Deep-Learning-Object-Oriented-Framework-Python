import numpy as np

from src.ActivationFunctions.ActivationFunction import TanH, ActivationFunction
from src.InitializationFunctions.InitializationFunction import InitializationFunction, Random
from src.Networks.Layer.Layer import Layer
from src.Regularizations import NormRegularizationFunction


class RecurrentLayerManyToOne(Layer):

    def __init__(self, num_neurons, num_output,
                 activation_function: ActivationFunction = TanH(),
                 initialization_function: InitializationFunction = Random()):
        super().__init__()
        self.num_neurons = num_neurons
        self.num_output = num_output
        self.initialization_function = initialization_function
        self.activation_function = activation_function
        self.last_inputs = None

        self.w = initialization_function.initialize(num_neurons, num_neurons)
        self.u = None # since the input size is not known, it is declared on the first iteration
        self.v = initialization_function.initialize(num_output, num_neurons)

        self.b = np.zeros((num_neurons, 1))
        self.c = np.zeros((num_output, 1))

        self.hidden_states = []

    def forward(self, x_input: np.array) -> np.array:
        self.__init_weight_late__(x_input)
        self.last_inputs = x_input

        batch_size, sequence_length, _ = x_input.shape

        h = np.zeros((batch_size, self.num_neurons))
        self.hidden_states.append(h)

        for t in range(sequence_length):
            x = x_input[:,t,:]
            print(x.shape)
            a = x @ self.u.T
            a += h @ self.w.T
            a += self.b.T
            h = self.activation_function.calculate(a)
            self.hidden_states.append(h)

        o = h @ self.v.T + self.c.T

        return o

    def backward(self, gradient: np.array, learning_rate: float,
                 regularization_function: NormRegularizationFunction) -> np.array:
        dv = gradient @ self.hidden_states[-1]

        dw = np.zeros(self.w.shape)
        du = np.zeros(self.u.shape)

        db = np.zeros(self.b.shape)
        dc = gradient

        dh = self.v @ gradient

        for t, h in enumerate(self.hidden_states[::-1]):
            temp = np.multiply(self.activation_function.calculate_gradient(self.hidden_states[t + 1]), dh)

            db += temp

            dw += temp @ h.T
            du += temp @ self.last_inputs[t].T

            dh = self.w @ temp

        for d in [du, dw, dv, db, dc]:
            np.clip(d, -1, 1, out=d)

        self.update_weight(learning_rate, [dw, du, dv])
        self.update_bias(learning_rate, [db, dc])

    def update_weight(self, learning_rate: float, grads: np.array):
        dw, du, dv = grads
        self.w += self.optimizer.calculate_weight(dw, learning_rate)
        self.u += self.optimizer.calculate_weight(du, learning_rate)
        self.v += self.optimizer.calculate_weight(dv, learning_rate)

    def update_bias(self, learning_rate: float, grads: np.array):
        db, dc = grads
        self.b += self.optimizer.calculate_bias(db, learning_rate)
        self.c += self.optimizer.calculate_bias(dc, learning_rate)

    def __str__(self):
        return f"rnn-{self.num_neurons}"

    def __init_weight_late__(self, x_input):
        if self.u is None:
            self.u = self.initialization_function.initialize(self.num_neurons, x_input.shape[-1])
