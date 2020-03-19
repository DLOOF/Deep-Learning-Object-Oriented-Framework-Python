from src.ActivationFunctions.ActivationFunction import *
from src.Networks.Layer.Layer import Layer
from src.Regularizations import NormRegularizationFunction


class MaxPoolingLayer(Layer):
    # FIXME: In this moment this class is a max polling of 2x2

    @staticmethod
    def __iterate_regions__(image):
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j

    def forward(self, x_input: np.array) -> np.array:
        self.last_input = x_input

        # we assume that the previous layer was convolution layer
        h, w, num_filters = x_input.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        for im_region, i, j in self.__iterate_regions__(x_input):
            output[i, j] = np.average(im_region, axis=(0, 1))

        self.last_output = output
        return self.last_output

    def backward(self, gradient: np.array, learning_rate: float,
                 regularization_function: NormRegularizationFunction) -> np.array:

        final_gradient = np.zeros(self.last_input.shape)

        for im_region, i, j in self.__iterate_regions__(self.last_input):
            h, w, f = im_region.shape
            a_avg = np.average(im_region, axis=(0, 1))
            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        # If this pixel was the max value, copy the gradient to it.
                        if im_region[i2, j2, f2] == a_avg[f2]:
                            final_gradient[i * 2 + i2, j * 2 + j2, f2] = gradient[i, j, f2]

        return final_gradient

    def update_weight(self, learning_rate: float, grads: np.array):
        pass

    def update_bias(self, learning_rate: float, grads: np.array):
        pass
