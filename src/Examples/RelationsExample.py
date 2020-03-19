from src.BatchFunctions.BatchFunction import MiniBatch
from src.CostFunctions.CostFunction import *
from src.Examples.ExampleTemplate import ExampleTemplate
from src.Networks.Layer.ClassicLayer import ClassicLayer, Sigmoid


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
        self.training_data = self.get_columns_between_a_and_b(self.get_data(), 0, 5).T
        self.expected_output = self.get_columns_between_a_and_b(self.get_data(), 6, 7).T

    def define_architecture(self):
        self.architecture = []

        layer_1 = ClassicLayer(6)
        layer_2 = ClassicLayer(8)
        layer_3 = ClassicLayer(8)
        layer_4 = ClassicLayer(2, Sigmoid())

        self.architecture.append(layer_1)
        self.architecture.append(layer_2)
        self.architecture.append(layer_3)
        self.architecture.append(layer_4)

        self.cost_function = MeanSquaredError()

    def define_training_hyperparameters(self):
        self.learning_rate = 0.05
        self.iterations = 1_500
        self.batch_function = MiniBatch(self.training_data, self.expected_output, 1)

    def run_tests(self):
        r = self.neural_net.predict(np.array([1, 1, 0, 0, 0, 0]).reshape(-1, 1)).T
        print(f" [1, 1, 0, 0, 0, 0] ans = [0, 1] = {r} = {np.argmax(r)}")
        r = self.neural_net.predict(np.array([1, 0, 1, 0, 0, 0]).reshape(-1, 1)).T
        print(f" [1, 0, 1, 0, 0, 0] ans = [0, 1] = {r} = {np.argmax(r)}")
        r = self.neural_net.predict(np.array([1, 0, 0, 1, 0, 0]).reshape(-1, 1)).T
        print(f" [1, 0, 0, 1, 0, 0] ans = [1, 0] = {r} = {np.argmax(r)}")
        r = self.neural_net.predict(np.array([1, 0, 0, 0, 1, 0]).reshape(-1, 1)).T
        print(f" [1, 0, 0, 0, 1, 0] ans = [1, 0] = {r} = {np.argmax(r)}")
        r = self.neural_net.predict(np.array([1, 0, 0, 0, 0, 1]).reshape(-1, 1)).T
        print(f" [1, 0, 0, 0, 0, 1] ans = [1, 0] = {r} = {np.argmax(r)}")
        r = self.neural_net.predict(np.array([0, 1, 1, 0, 0, 0]).reshape(-1, 1)).T
        print(f" [0, 1, 1, 0, 0, 0] ans = [0, 1] = {r} = {np.argmax(r)}")
        r = self.neural_net.predict(np.array([0, 1, 0, 1, 0, 0]).reshape(-1, 1)).T
        print(f" [0, 1, 0, 1, 0, 0] ans = [1, 0] = {r} = {np.argmax(r)}")
        r = self.neural_net.predict(np.array([0, 1, 0, 0, 1, 0]).reshape(-1, 1)).T
        print(f" [0, 1, 0, 0, 1, 0] ans = [1, 0] = {r} = {np.argmax(r)}")
        r = self.neural_net.predict(np.array([0, 1, 0, 0, 0, 1]).reshape(-1, 1)).T
        print(f" [0, 1, 0, 0, 0, 1] ans = [1, 0] = {r} = {np.argmax(r)}")
        r = self.neural_net.predict(np.array([0, 0, 1, 1, 0, 0]).reshape(-1, 1)).T
        print(f" [0, 0, 1, 1, 0, 0] ans = [1, 0] = {r} = {np.argmax(r)}")
        r = self.neural_net.predict(np.array([0, 0, 1, 0, 1, 0]).reshape(-1, 1)).T
        print(f" [0, 0, 1, 0, 1, 0] ans = [1, 0] = {r} = {np.argmax(r)}")
        r = self.neural_net.predict(np.array([0, 0, 1, 0, 0, 1]).reshape(-1, 1)).T
        print(f" [0, 0, 1, 0, 0, 1] ans = [1, 0] = {r} = {np.argmax(r)}")
        r = self.neural_net.predict(np.array([0, 0, 0, 1, 1, 0]).reshape(-1, 1)).T
        print(f" [0, 0, 0, 1, 1, 0] ans = [0, 1] = {r} = {np.argmax(r)}")
        r = self.neural_net.predict(np.array([0, 0, 0, 1, 0, 1]).reshape(-1, 1)).T
        print(f" [0, 0, 0, 1, 0, 1] ans = [0, 1] = {r} = {np.argmax(r)}")
        r = self.neural_net.predict(np.array([0, 0, 0, 0, 1, 1]).reshape(-1, 1)).T
        print(f" [0, 0, 0, 0, 1, 1] ans = [0, 1] = {r} = {np.argmax(r)}")


if __name__ == '__main__':
    np.random.seed(0)
    relation_example = RelationExample("relation_example")
    relation_example.run()
