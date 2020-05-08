from src.ActivationFunctions.ActivationFunction import *
from src.Networks.Layer.Layer import Layer
from src.Regularizations import NormRegularizationFunction


class Convolution2DLayer(Layer):
    # FIXME: In this moment this class is a convolution filter 3x3

    def __init__(self, channels, filter_size=3):
        super().__init__()
        self.channels = channels
        self.filters = np.random.randn(channels, filter_size, filter_size) / 9

    @staticmethod
    def __iterate_regions__(image):
        h, w = image.shape
        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j

    def forward(self, x_input: np.array) -> np.array:
        assert x_input.ndim >= 2

        self.last_input = x_input

        output = []
        for ex in x_input:
            h, w = ex.shape

            convolution = np.zeros((h - 2, w - 2, self.channels))
            for im_region, i, j in self.__iterate_regions__(ex):
                convolution[i, j] = np.sum(im_region * self.filters)

            output.append(convolution)

        self.last_activation_output = np.array(output)
        return self.last_activation_output

    def backward(self, gradient: np.array, learning_rate: float,
                 regularization_function: NormRegularizationFunction) -> np.array:

        final_gradient = np.zeros(self.filters.shape)
        for im_region, i, j in self.__iterate_regions__(self.last_input):
            for f in range(self.channels):
                final_gradient[f] += gradient[i, j, f] * im_region

        self.update_weight(learning_rate, final_gradient)
        # TODO check this gradient!
        return final_gradient

    def update_weight(self, learning_rate: float, grads: np.array):
        self.filters -= learning_rate * grads

    def update_bias(self, learning_rate: float, grads: np.array):
        pass

    def __str__(self):
        return f"sm-{self.num_neurons}"
