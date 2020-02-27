import numpy as np
from typing import List


def partition_data(input_data: np.array, training_set_percentage: float, development_set_percentage: float,
                   testing_set_percentage: float) -> List[np.array]:

    assert round(training_set_percentage + development_set_percentage + testing_set_percentage) == 1

    data_size = input_data.shape[0]
    training_set_size = int(data_size * training_set_percentage)
    development_set_size = int(data_size * development_set_percentage)
    testing_set_size = data_size - training_set_size - development_set_size

    training_set = input_data[:training_set_size]
    development_set = input_data[training_set_size:development_set_size]
    testing_set = input_data[training_set_size + development_set_size:testing_set_size]

    return [training_set, development_set, testing_set]
