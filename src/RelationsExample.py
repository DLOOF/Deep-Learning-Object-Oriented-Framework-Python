from src import *
import numpy as np

from src.ActivationFunction import *
from src.CostFunction import *
from src.NeuronalNetwork import NeuronalNetwork


def get_data():
    return np.array(
        [[1, 1, 0, 0, 0, 0, 0, 1],
         [1, 0, 1, 0, 0, 0, 0, 1],
         [1, 0, 0, 1, 0, 0, 1, 0],
         [1, 0, 0, 0, 1, 0, 1, 0],
         [1, 0, 0, 0, 0, 1, 1, 0],
         [0, 1, 1, 0, 0, 0, 0, 1],
         [0, 1, 0, 1, 0, 0, 1, 0],
         [0, 1, 0, 0, 1, 0, 1, 0],
         [0, 1, 0, 0, 0, 1, 1, 0],
         [0, 0, 1, 1, 0, 0, 1, 0],
         [0, 0, 1, 0, 1, 0, 1, 0],
         [0, 0, 1, 0, 0, 1, 1, 0],
         [0, 0, 0, 1, 1, 0, 0, 1],
         [0, 0, 0, 1, 0, 1, 0, 1],
         [0, 0, 0, 0, 1, 1, 0, 1]]
    )


def get_columns_between_a_and_b(matrix: np.array, a: int, b: int):
    return matrix[:, a:b + 1]


def get_column(matrix: np.array, index: int) -> np.array:
    return matrix[:, [index]]


if __name__ == '__main__':
    architecture = []
    layer_1 = (8, Relu())
    layer_2 = (8, Relu())
    layer_3 = (8, Relu())
    layer_4 = (8, Relu())
    architecture.append(layer_1)
    architecture.append(layer_2)
    # architecture.append(layer_3)
    # architecture.append(layer_4)
    costFunction = MeanSquaredError()
    input_neurons = 6
    output_neurons = 2
    neuronal_net = NeuronalNetwork(input_neurons, architecture, output_neurons, costFunction)

    learning_rate = 0.1
    iterations = 10000
    data = get_data()
    training_data = get_columns_between_a_and_b(data, 0, 5)
    expected_output = get_columns_between_a_and_b(data, 6, 7)
    neuronal_net.train(training_data, expected_output, learning_rate, iterations)

    print(f" [1, 1, 0, 0, 0, 0] ans = [0, 1] = {neuronal_net.forward(np.array([1, 1, 0, 0, 0, 0]).reshape(-1,1))}")
    print(f" [1, 0, 1, 0, 0, 0] ans = [0, 1] = {neuronal_net.forward(np.array([1, 0, 1, 0, 0, 0]).reshape(-1,1))}")
    print(f" [1, 0, 0, 1, 0, 0] ans = [1, 0] = {neuronal_net.forward(np.array([1, 0, 0, 1, 0, 0]).reshape(-1,1))}")
    print(f" [1, 0, 0, 0, 1, 0] ans = [1, 0] = {neuronal_net.forward(np.array([1, 0, 0, 0, 1, 0]).reshape(-1,1))}")
    print(f" [1, 0, 0, 0, 0, 1] ans = [1, 0] = {neuronal_net.forward(np.array([1, 0, 0, 0, 0, 1]).reshape(-1,1))}")
    print(f" [0, 1, 1, 0, 0, 0] ans = [0, 1] = {neuronal_net.forward(np.array([0, 1, 1, 0, 0, 0]).reshape(-1,1))}")
    print(f" [0, 1, 0, 1, 0, 0] ans = [1, 0] = {neuronal_net.forward(np.array([0, 1, 0, 1, 0, 0]).reshape(-1,1))}")
    print(f" [0, 1, 0, 0, 1, 0] ans = [1, 0] = {neuronal_net.forward(np.array([0, 1, 0, 0, 1, 0]).reshape(-1,1))}")
    print(f" [0, 1, 0, 0, 0, 1] ans = [1, 0] = {neuronal_net.forward(np.array([0, 1, 0, 0, 0, 1]).reshape(-1,1))}")
    print(f" [0, 0, 1, 1, 0, 0] ans = [1, 0] = {neuronal_net.forward(np.array([0, 0, 1, 1, 0, 0]).reshape(-1,1))}")
    print(f" [0, 0, 1, 0, 1, 0] ans = [1, 0] = {neuronal_net.forward(np.array([0, 0, 1, 0, 1, 0]).reshape(-1,1))}")
    print(f" [0, 0, 1, 0, 0, 1] ans = [1, 0] = {neuronal_net.forward(np.array([0, 0, 1, 0, 0, 1]).reshape(-1,1))}")
    print(f" [0, 0, 0, 1, 1, 0] ans = [0, 1] = {neuronal_net.forward(np.array([0, 0, 0, 1, 1, 0]).reshape(-1,1))}")
    print(f" [0, 0, 0, 1, 0, 1] ans = [0, 1] = {neuronal_net.forward(np.array([0, 0, 0, 1, 0, 1]).reshape(-1,1))}")
    print(f" [0, 0, 0, 0, 1, 1] ans = [0, 1] = {neuronal_net.forward(np.array([0, 0, 0, 0, 1, 1]).reshape(-1,1))}")























