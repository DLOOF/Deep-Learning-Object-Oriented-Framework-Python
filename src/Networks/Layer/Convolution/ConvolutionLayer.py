from src.ActivationFunctions.ActivationFunction import *
from src.Networks.Layer.Layer import Layer
from src.Regularizations import NormRegularizationFunction


class ConvolutionLayer(Layer):
    # FIXME: In this moment this class is a convolution filter 3x3

    def __init__(self, num_filters):
        super().__init__()
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    @staticmethod
    def __iterate_regions__(image):
        h, w = image.shape
        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j

    def forward(self, x_input: np.array) -> np.array:
        self.last_input = x_input

        h, w = x_input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for im_region, i, j in self.__iterate_regions__(x_input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
        return output

    def backward(self, gradient: np.array, learning_rate: float,
                 regularization_function: NormRegularizationFunction) -> np.array:

        final_gradient = np.zeros(self.filters.shape)
        for im_region, i, j in self.__iterate_regions__(self.last_input):
            for f in range(self.num_filters):
                final_gradient[f] += gradient[i, j, f] * im_region

        self.update_weight(learning_rate, final_gradient)
        # TODO check this gradient!
        return final_gradient

    def update_weight(self, learning_rate: float, grads: np.array):
        self.filters -= learning_rate * grads

    def update_bias(self, learning_rate: float, grads: np.array):
        pass