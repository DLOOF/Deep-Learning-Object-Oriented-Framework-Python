from src import *
import numpy as np

from src.ActivationFunction import *
from src.CostFunction import *
from src.NeuronalNetwork import NeuronalNetwork


def get_data():
    return np.array([
        [1, 1, 0],
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
    ]
    )


if __name__ == '__main__':
    architecture = []
    layer_1 = (2, Relu())
    layer_2 = (2, Relu())
    architecture.append(layer_1)
    architecture.append(layer_2)
    costFunction = MeanSquaredError()
    input_neurons = 2
    output_neurons = 1
    neuronal_net = NeuronalNetwork(input_neurons, architecture, output_neurons, costFunction)

    learning_rate = 0.1
    iterations = 10000
    neuronal_net.train(get_data(), learning_rate, iterations)

    print(f" [0, 0] = {neuronal_net.forward(np.array([0, 0]).reshape(2, 1))}")
    print(f" [0, 1] = {neuronal_net.forward(np.array([0, 1]).reshape(2, 1))}")
    print(f" [1, 0] = {neuronal_net.forward(np.array([1, 0]).reshape(2, 1))}")
    print(f" [1, 1] = {neuronal_net.forward(np.array([1, 1]).reshape(2, 1))}")

