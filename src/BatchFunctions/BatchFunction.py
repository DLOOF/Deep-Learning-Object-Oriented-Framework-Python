from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from math import ceil


class BatchFunction(ABC):

    @abstractmethod
    def __init__(self, input_data: np.array, expected_output: np.array):
        self.input_data = input_data
        self.expected_output = expected_output

    """
    This method should raise StopIterator exception when there are not more batches to process
    """

    @abstractmethod
    def get_batch(self) -> Tuple[np.array, np.array]:
        pass


class BatchMode(BatchFunction):

    def __init__(self, input_data: np.array, expected_output: np.array):
        super().__init__(input_data, expected_output)

    def get_batch(self) -> Tuple[np.array, np.array]:
        yield self.input_data, self.expected_output


class MiniBatch(BatchFunction):

    def __init__(self, input_data: np.array, expected_output: np.array, batch_size: int):
        super().__init__(input_data, expected_output)
        assert 1 <= batch_size <= input_data.shape[0]
        n = input_data.shape[0]
        self.batch_size = batch_size

        self.iterations = ceil(n / batch_size)
        self.it = 0

    def get_batch(self) -> Tuple[np.array, np.array]:
        n = self.input_data.shape[1]

        joined = np.vstack((self.input_data, self.expected_output)).T
        np.random.shuffle(joined)
        joined = joined.T

        # TODO: let the batch size be for any size, now is requiring that is an divisor of n
        for batch in np.hsplit(joined, n / self.batch_size):
            input_batch, output_batch = np.vsplit(batch, [self.input_data.shape[0]])
            yield input_batch, output_batch


class MiniBatchNormalized(MiniBatch):
    epsilon = 1e-8

    def __init__(self, input_data: np.array, expected_output: np.array, batch_size: int):
        super().__init__(input_data, expected_output, batch_size)

    def get_batch(self) -> Tuple[np.array, np.array]:
        batch_data, batch_expected = super(MiniBatch, self).get_batch()
        batch_mean = np.mean(batch_data, axis=1)
        batch_variance = np.var(batch_data, axis=1)
        batch_data_normalized = batch_data - batch_mean
        batch_data_normalized = np.divide(batch_data_normalized, np.sqrt(batch_variance + MiniBatchNormalized.epsilon))

        # TODO: there are two learnable parameters \gamma and \beta that need to be adjusted such that
        #  y_i = \gamma \hat{x_i} + \beta, for the time being we won't use them

        return batch_data_normalized, batch_expected
