from src import *
import numpy as np

from src.ActivationFunction import Relu
from src.CostFunction import MeanSquaredError
from src.NeuronalNetwork import NeuronalNetwork


def get_data():
    np.array([[0, 0, 0],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 0]])


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

    learning_rate = 0.01
    iterations = 100
    neuronal_net.train(get_data(), learning_rate, iterations)
