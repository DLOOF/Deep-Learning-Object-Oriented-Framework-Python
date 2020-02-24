from abc import ABC, abstractmethod
import numpy as np

from src.Networks.SupervisedModel import SupervisedModel


class StoppingCondition(ABC):

    @abstractmethod
    def should_stop(self, model: SupervisedModel, input_data: np.array, expected_output: np.array) -> bool:
        pass


class VoidStoppingCondition(StoppingCondition):

    def should_stop(self, model: SupervisedModel, input_data: np.array, expected_output: np.array) -> bool:
        return False


class EarlyStoppingRegularization(StoppingCondition):

    def should_stop(self, model: SupervisedModel, input_data: np.array, expected_output: np.array):
        raise NotImplementedError  # TODO
        pass
