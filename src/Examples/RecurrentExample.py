from src.BatchFunctions.BatchFunction import MiniBatch
from src.CostFunctions.CostFunction import *
from src.Examples.ExampleTemplate import ExampleTemplate
from src.Networks.Layer.ClassicLayer import ClassicLayer, Sigmoid
from src.Networks.Layer.Recurrent.RecurrentLayerManyToOne import RecurrentLayerManyToOne
from src.Optimizers.Optimizers import SGD, Adam


class RecurrentExample(ExampleTemplate):

    def _get_bow(self, data):
        docs = [doc for doc in data.keys()]
        words = dict()

        max_seq_length = 0
        for doc in docs:
            seq = doc.split()
            max_seq_length = max(max_seq_length, len(seq))
            for word in seq:
                words[word] = words.get(word, 0) + 1

        words_count = [(v, k) for k, v in words.items()]
        words_count.sort()

        words2vtr = dict()
        vtr2words = dict()
        n = len(words_count)

        for i, (_, word) in enumerate(words_count):
            z = np.zeros(n, dtype=np.int32)
            z[i] = 1
            words2vtr[word] = z
            vtr2words[i] = word

        return words2vtr, vtr2words, n, max_seq_length

    def _to_bow(self, doc, words2vtr, n, seq_length):
        r = np.zeros(n)
        i = 0
        for i, word in enumerate(doc.split()):
            r = np.vstack([r, words2vtr[word]])

        for _ in range(seq_length-i):
            r = np.vstack([r, np.zeros(n)])

        return r

    def _convert_dataset_to_bow(self, dataset, words2vtr, n, seq_length):
        r = []
        true, false = np.array([1, 0]), np.array([0, 1])
        for k, v in dataset.items():
            kvtr = self._to_bow(k, words2vtr, n, seq_length)
            r.append((kvtr, (true if v else false)))

        X = None
        y = None
        for k, v in r:
            if X is None:
                dim = k.shape
                X = np.expand_dims(k, axis=0)
                y = np.atleast_2d(v)
            else:
                k = np.expand_dims(k, axis=0)
                X = np.vstack([X, k])
                y = np.vstack([y, v])

        print(X.shape, y.shape)

        return X, y

    def get_data(self) -> np.array:
        train_data = {
            'good': True,
            'bad': False,
            'happy': True,
            'sad': False,
            'not good': False,
            'not bad': True,
            'not happy': False,
            'not sad': True,
            'very good': True,
            'very bad': False,
            'very happy': True,
            'very sad': False,
            'i am happy': True,
            'this is good': True,
            'i am bad': False,
            'this is bad': False,
            'i am sad': False,
            'this is sad': False,
            'i am not happy': False,
            'this is not good': False,
            'i am not bad': True,
            'this is not sad': True,
            'i am very happy': True,
            'this is very good': True,
            'i am very bad': False,
            'this is very sad': False,
            'this is very happy': True,
            'i am good not bad': True,
            'this is good not bad': True,
            'i am bad not good': False,
            'i am good and happy': True,
            'this is not good and not happy': False,
            'i am not at all good': False,
            'i am not at all bad': True,
            'i am not at all happy': False,
            'this is not at all sad': True,
            'this is not at all happy': False,
            'i am good right now': True,
            'i am bad right now': False,
            'this is bad right now': False,
            'i am sad right now': False,
            'i was good earlier': True,
            'i was happy earlier': True,
            'i was bad earlier': False,
            'i was sad earlier': False,
            'i am very bad right now': False,
            'this is very good right now': True,
            'this is very sad right now': False,
            'this was bad earlier': False,
            'this was very good earlier': True,
            'this was very bad earlier': False,
            'this was very happy earlier': True,
            'this was very sad earlier': False,
            'i was good and not bad earlier': True,
            'i was not good and not happy earlier': False,
            'i am not at all bad or sad right now': True,
            'i am not at all good or happy right now': False,
            'this was not happy and not good earlier': False,
        }

        test_data = {
            'this is happy': True,
            'i am good': True,
            'this is not happy': False,
            'i am not good': False,
            'this is not bad': True,
            'i am not sad': True,
            'i am very good': True,
            'this is very bad': False,
            'i am very sad': False,
            'this is bad not good': False,
            'this is good and happy': True,
            'i am not good and not happy': False,
            'i am not at all sad': True,
            'this is not at all good': False,
            'this is not at all bad': True,
            'this is good right now': True,
            'this is sad right now': False,
            'this is very bad right now': False,
            'this was good earlier': True,
            'i was not happy and not good earlier': False,
        }

        words2vtr, vtr2words, n, seq_length = self._get_bow(train_data)
        X, y = self._convert_dataset_to_bow(train_data, words2vtr, n, seq_length)

        return X, y, n

    def define_data(self):
        self.training_data, self.expected_output, _ = self.get_data()

    def define_architecture(self):
        self.architecture = []

        _, _, n = self.get_data()

        layer_1 = RecurrentLayerManyToOne(8, n)
        layer_2 = ClassicLayer(2, Sigmoid())

        self.architecture.append(layer_1)
        self.architecture.append(layer_2)

        self.cost_function = MeanSquaredError()

    def define_training_hyperparameters(self):
        self.learning_rate = 0.001
        self.iterations = 10_000
        self.batch_function = MiniBatch(self.training_data, self.expected_output, 58//2)
        self.optimizer = Adam()

    def run_tests(self):
        print(f" [0, 0] = {self.neural_net.predict(np.array([0, 0]).reshape(-1, 1))}")
        print(f" [0, 1] = {self.neural_net.predict(np.array([0, 1]).reshape(-1, 1))}")
        print(f" [1, 0] = {self.neural_net.predict(np.array([1, 0]).reshape(-1, 1))}")
        print(f" [1, 1] = {self.neural_net.predict(np.array([1, 1]).reshape(-1, 1))}")


if __name__ == '__main__':
    # np.random.seed(0)
    xor_example = RecurrentExample("recurrent_example")
    xor_example.run()
