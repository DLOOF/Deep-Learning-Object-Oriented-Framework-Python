from src.BatchFunctions.BatchFunction import MiniBatch
from src.CostFunctions.CostFunction import *
from src.Examples.ExampleTemplate import ExampleTemplate
from src.Networks.Layer.ClassicLayer import ClassicLayer, Sigmoid


class XorExample(ExampleTemplate):

    def get_data(self) -> np.array:
        return np.array(
            [[1, 1, 0],
             [0, 0, 0],
             [0, 1, 1],
             [1, 0, 1]]
        )

    def define_data(self):
        self.training_data = self.get_columns_between_a_and_b(self.get_data(), 0, 1).T
        self.expected_output = self.get_column(self.get_data(), 2).T

    def define_architecture(self):
        self.architecture = []

        layer_1 = ClassicLayer(2)
        layer_2 = ClassicLayer(4)
        layer_3 = ClassicLayer(2, Sigmoid())
        layer_4 = ClassicLayer(4)

        self.architecture.append(layer_1)
        self.architecture.append(layer_2)
        self.architecture.append(layer_3)
        # self.architecture.append(layer_4)

        self.cost_function = MeanSquaredError()

    def define_training_hyperparameters(self):
        self.learning_rate = 0.1
        self.iterations = 10_000
        self.batch_function = MiniBatch(self.training_data, self.expected_output, 1)

    def run_tests(self):
        print(f" [0, 0] = {self.neural_net.predict(np.array([0, 0]).reshape(-1, 1))}")
        print(f" [0, 1] = {self.neural_net.predict(np.array([0, 1]).reshape(-1, 1))}")
        print(f" [1, 0] = {self.neural_net.predict(np.array([1, 0]).reshape(-1, 1))}")
        print(f" [1, 1] = {self.neural_net.predict(np.array([1, 1]).reshape(-1, 1))}")


if __name__ == '__main__':
    np.random.seed(0)
    xor_example = XorExample("xor_example")
    xor_example.run()
