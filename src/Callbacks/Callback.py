from abc import abstractmethod, ABC

import matplotlib.pyplot as plt
import numpy as np


class Callback(ABC):

    @abstractmethod
    def call(self, metrics_history):
        pass


class GraphicCallback(Callback):

    def call(self, metrics_history):
        fig, axs = plt.subplots(len(metrics_history))

        axs = np.array(axs).flatten()

        n = 0
        for name, history in metrics_history.items():
            axs[n].plot(list(range(len(history))), history)
            axs[n].set_title(name)
            n += 1

        plt.show()
