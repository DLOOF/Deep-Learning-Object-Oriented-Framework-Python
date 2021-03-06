from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


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
        yield self.input_data, np.atleast_2d(self.expected_output).T


class MiniBatch(BatchFunction):

    def __init__(self, input_data: np.array, expected_output: np.array, batch_size: int):
        super().__init__(input_data, expected_output)
        assert 1 <= batch_size <= input_data.shape[0]

        self.batch_size = batch_size
        self.input_data = np.atleast_2d(self.input_data)
        self.expected_output = np.atleast_2d(self.expected_output).T

    def get_batch(self) -> Tuple[np.array, np.array]:
        examples, *output_size = self.input_data.shape
        idxs = np.arange(examples, dtype=np.int32)
        np.random.shuffle(idxs)

        # TODO: let the batch size be for any size, now is requiring that is a divisor of n
        for batch_idxs in np.hsplit(idxs, examples / self.batch_size):
            input_batch, output_batch = self.input_data[batch_idxs,:], self.expected_output[batch_idxs,:]
            yield input_batch, output_batch


class MiniBatchNormalized(MiniBatch):
    epsilon = 1e-8

    def __init__(self, input_data: np.array, expected_output: np.array, batch_size: int):
        super().__init__(input_data, expected_output, batch_size)

    def get_batch(self) -> Tuple[np.array, np.array]:
        for batch_data, batch_expected in super(MiniBatchNormalized, self).get_batch():
            batch_mean = np.atleast_2d(np.mean(batch_data, axis=0))
            batch_variance = np.atleast_2d(np.var(batch_data, axis=0))
            batch_data_normalized = batch_data - np.broadcast_to(batch_mean, batch_data.shape)
            batch_data_normalized = np.divide(batch_data_normalized, np.sqrt(batch_variance + MiniBatchNormalized.epsilon))

            # TODO: there are two learnable parameters \gamma and \beta that need to be adjusted such that
            #  y_i = \gamma \hat{x_i} + \beta, for the time being we won't use them

            yield batch_data_normalized, batch_expected
