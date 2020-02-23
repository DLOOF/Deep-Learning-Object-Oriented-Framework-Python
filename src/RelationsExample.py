from src.ActivationFunctions.ActivationFunction import *
from src.CostFunctions.CostFunction import *
from src.ExampleTemplate import ExampleTemplate


class RelationExample(ExampleTemplate):

    def get_data(self) -> np.array:
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

    def define_data(self):
        self.training_data = self.get_columns_between_a_and_b(self.get_data(), 0, 5)
        self.expected_output = self.get_columns_between_a_and_b(self.get_data(), 6, 7)

    def define_architecture(self):
        self.architecture = []
        self.input_neurons = 6
        self.output_neurons = 2

        layer_1 = (8, Relu())
        layer_2 = (8, Relu())

        self.architecture.append(layer_1)
        self.architecture.append(layer_2)

        self.cost_function = MeanSquaredError()

    def define_training_hyperparameters(self):
        self.learning_rate = 0.1
        self.iterations = 10_000

    def run_tests(self):
        print(
            f" [1, 1, 0, 0, 0, 0] ans = [0, 1] = {self.neural_net.forward(np.array([1, 1, 0, 0, 0, 0]).reshape(-1, 1)).T}")
        print(
            f" [1, 0, 1, 0, 0, 0] ans = [0, 1] = {self.neural_net.forward(np.array([1, 0, 1, 0, 0, 0]).reshape(-1, 1)).T}")
        print(
            f" [1, 0, 0, 1, 0, 0] ans = [1, 0] = {self.neural_net.forward(np.array([1, 0, 0, 1, 0, 0]).reshape(-1, 1)).T}")
        print(
            f" [1, 0, 0, 0, 1, 0] ans = [1, 0] = {self.neural_net.forward(np.array([1, 0, 0, 0, 1, 0]).reshape(-1, 1)).T}")
        print(
            f" [1, 0, 0, 0, 0, 1] ans = [1, 0] = {self.neural_net.forward(np.array([1, 0, 0, 0, 0, 1]).reshape(-1, 1)).T}")
        print(
            f" [0, 1, 1, 0, 0, 0] ans = [0, 1] = {self.neural_net.forward(np.array([0, 1, 1, 0, 0, 0]).reshape(-1, 1)).T}")
        print(
            f" [0, 1, 0, 1, 0, 0] ans = [1, 0] = {self.neural_net.forward(np.array([0, 1, 0, 1, 0, 0]).reshape(-1, 1)).T}")
        print(
            f" [0, 1, 0, 0, 1, 0] ans = [1, 0] = {self.neural_net.forward(np.array([0, 1, 0, 0, 1, 0]).reshape(-1, 1)).T}")
        print(
            f" [0, 1, 0, 0, 0, 1] ans = [1, 0] = {self.neural_net.forward(np.array([0, 1, 0, 0, 0, 1]).reshape(-1, 1)).T}")
        print(
            f" [0, 0, 1, 1, 0, 0] ans = [1, 0] = {self.neural_net.forward(np.array([0, 0, 1, 1, 0, 0]).reshape(-1, 1)).T}")
        print(
            f" [0, 0, 1, 0, 1, 0] ans = [1, 0] = {self.neural_net.forward(np.array([0, 0, 1, 0, 1, 0]).reshape(-1, 1)).T}")
        print(
            f" [0, 0, 1, 0, 0, 1] ans = [1, 0] = {self.neural_net.forward(np.array([0, 0, 1, 0, 0, 1]).reshape(-1, 1)).T}")
        print(
            f" [0, 0, 0, 1, 1, 0] ans = [0, 1] = {self.neural_net.forward(np.array([0, 0, 0, 1, 1, 0]).reshape(-1, 1)).T}")
        print(
            f" [0, 0, 0, 1, 0, 1] ans = [0, 1] = {self.neural_net.forward(np.array([0, 0, 0, 1, 0, 1]).reshape(-1, 1)).T}")
        print(
            f" [0, 0, 0, 0, 1, 1] ans = [0, 1] = {self.neural_net.forward(np.array([0, 0, 0, 0, 1, 1]).reshape(-1, 1)).T}")


if __name__ == '__main__':
    np.random.seed(0)
    relation_example = RelationExample("relation_example")
    relation_example.run()
