import numpy as np
from mnist import MNIST
from sklearn.preprocessing import OneHotEncoder

from src.ActivationFunctions.ActivationFunction import *
from src.CostFunctions.CostFunction import *
from src.ExampleTemplate import ExampleTemplate


class Mnist_Example(ExampleTemplate):

    def run_tests(self):
        train_x, train_y, test_x, test_y = self.get_data()
        for i, x in enumerate(test_x):
            y = self.neural_net.predict(x)
            print(f"expected: {test_y[i]} --> output: {y}")
        pass

    def get_data(self):
        mndata = MNIST('../../datasets/mnist')
        mndata.gz = True
        images, labels = mndata.load_training()
        train_x = np.array(images)
        train_y = np.array(labels)

        images, labels = mndata.load_testing()
        test_x = np.array(images)
        test_y = np.array(labels)
        return train_x, train_y, test_x, test_y

    @staticmethod
    def pre_process_data(train_x, train_y, test_x, test_y):
        # Normalize
        train_x = train_x / 255.
        test_x = test_x / 255.

        enc = OneHotEncoder(sparse=False, categories='auto')
        train_y = enc.fit_transform(train_y.reshape(len(train_y), -1))

        test_y = enc.transform(test_y.reshape(len(test_y), -1))

        return train_x, train_y, test_x, test_y

    def define_architecture(self):
        self.architecture = []
        self.input_neurons = 28 * 28
        self.output_neurons = 10

        layer_1 = (10, Relu())
        layer_2 = (10, Relu())
        layer_3 = (10, Relu())
        layer_4 = (10, Relu())
        layer_5 = (20, Relu())
        layer_6 = (20, Relu())
        layer_7 = (20, Relu())
        layer_8 = (20, Relu())
        layer_9 = (20, Relu())
        layer_10 = (30, Relu())
        layer_11 = (30, Relu())

        self.architecture.append(layer_1)
        # self.architecture.append(layer_2)
        # self.architecture.append(layer_3)
        # self.architecture.append(layer_4)
        # self.architecture.append(layer_5)
        # self.architecture.append(layer_6)
        # self.architecture.append(layer_7)
        # self.architecture.append(layer_8)
        # self.architecture.append(layer_9)
        # self.architecture.append(layer_10)
        # self.architecture.append(layer_11)

        self.cost_function = CrossEntropy()

    def define_training_hyperparameters(self):
        self.learning_rate = 0.1
        self.iterations = 1_000

    def define_data(self):
        train_x, train_y, test_x, test_y = self.get_data()
        train_x, train_y, test_x, test_y = self.pre_process_data(train_x, train_y, test_x, test_y)
        self.training_data = train_x
        self.expected_output = train_y


if __name__ == '__main__':
    example = Mnist_Example("mnist")
    example.run()
