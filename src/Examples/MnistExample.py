from functools import lru_cache

from mnist import MNIST

from src.ActivationFunctions.ActivationFunction import SoftMax
from src.BatchFunctions.BatchFunction import MiniBatchNormalized
from src.CostFunctions.CostFunction import *
from src.Examples.ExampleTemplate import ExampleTemplate
from src.InitializationFunctions.InitializationFunction import He
from src.Networks.Layer.ClassicLayer import ClassicLayer, Sigmoid, TanH, Xavier
from src.Networks.Layer.Convolution.AvgPoolingLayer import MaxPoolingLayer
from src.Networks.Layer.Convolution.ConvolutionLayer import ConvolutionLayer
from src.Networks.Layer.SoftMaxLayer import SoftMaxLayer


class MnistExample(ExampleTemplate):

    def __encode_to_one_hot_vector(self, value: np.array):
        output = value.astype(int)
        tmp = np.zeros((output.size, output.max() +1))
        tmp[np.arange(output.size), output] = 1

        return tmp

    @lru_cache()
    def get_data(self) -> np.array:
        mnist = MNIST('../../datasets/mnist')
        x_train, y_train = mnist.load_training()  # 60000 samples
        x_test, y_test = mnist.load_testing()  # 10000 samples

        x_train = np.asarray(x_train).astype(np.float32)
        y_train = self.__encode_to_one_hot_vector(np.asarray(y_train).astype(np.int32))
        x_test = np.asarray(x_test).astype(np.float32)
        y_test = self.__encode_to_one_hot_vector(np.asarray(y_test).astype(np.int32))


        # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        # print(x_train, y_train, x_test, y_test)

        self.training_data = x_train
        self.expected_output = y_train
        return x_train, y_train, x_test, y_test

    def define_data(self):
        x_train, y_train, x_test, y_test = self.get_data()
        self.training_data = x_train.T
        self.expected_output = y_train.T

    def define_architecture(self):
        self.architecture = []

        # layer_1 = ConvolutionLayer(9)
        # layer_2 = MaxPoolingLayer()
        # layer_0 = ClassicLayer(784)
        layer_3 = ClassicLayer(784, TanH(), Xavier())
        # layer_4 = ClassicLayer(300, TanH(), Xavier())
        # layer_5 = ClassicLayer(300, TanH(), Xavier())
        layer_6 = ClassicLayer(10, SoftMax(), He())

        # self.architecture.append(layer_1)
        # self.architecture.append(layer_2)
        self.architecture.append(layer_3)
        # self.architecture.append(layer_4)
        # self.architecture.append(layer_5)
        self.architecture.append(layer_6)

        self.cost_function = MeanSquaredError()

    def define_training_hyperparameters(self):
        self.learning_rate = 0.2
        self.iterations = 7
        self.batch_function = MiniBatchNormalized(self.training_data, self.expected_output, 200)

    def run_tests(self):
        _, _, x_test, y_test = self.get_data()
        import random
        errors = []
        for x, y in random.sample(list(zip(x_test, y_test)), k=2000):
            # print(x, y, sep="\t")
            y_predicted = self.neural_net.predict(x)
            d = y - y_predicted
            error = np.linalg.norm(d, 2)
            errors.append(error)

        print("Mean error %.3f" % np.mean(np.asarray(errors)))


if __name__ == '__main__':
    np.random.seed(0)
    xor_example = MnistExample("mnist_example")
    xor_example.run()
