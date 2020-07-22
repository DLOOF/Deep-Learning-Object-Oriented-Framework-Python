from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):

    def __init__(self):
        self.r_bias = None
        self.r_weight = None

    @abstractmethod
    def copy_instance(self):
        pass

    @abstractmethod
    def _calculate(self, grad, learning_rate, r):
        pass

    def calculate_bias(self, grad, learning_rate):
        val, r = self._calculate(grad, learning_rate, self.r_bias)
        self.r_bias = r
        return val

    def calculate_weight(self, grad, learning_rate):
        val, r = self._calculate(grad, learning_rate, self.r_weight)
        self.r_weight = r
        return val


class SGDMomentum(Optimizer):

    def __init__(self, momentum_rate=0.5, initial_velocity=0.5):
        super().__init__()
        self.momentum_rate = momentum_rate
        self.initial_velocity = initial_velocity

    def copy_instance(self):
        return SGDMomentum(self.momentum_rate, self.initial_velocity)

    def _calculate(self, grad, learning_rate, r):
        if r is None:
            r = self.initial_velocity
        v = self.momentum_rate * r - learning_rate * grad
        return v, v


class SGD(SGDMomentum):

    def __init__(self):
        super(SGD, self).__init__()

    def copy_instance(self):
        return SGDMomentum(momentum_rate=0.0, initial_velocity=0.0)


class AdaGrad(Optimizer):

    def copy_instance(self):
        return AdaGrad(self.delta)

    def __init__(self, delta=1e-7):
        super().__init__()
        self.delta = delta

    def _calculate(self, grad, learning_rate, r):
        grad_grad = np.multiply(grad, grad)
        if r is None:
            r = grad_grad
        else:
            r += grad_grad
        denominator = self.delta + np.sqrt(r)
        return - np.multiply(np.divide(learning_rate, denominator), grad), r


class RMSProp(Optimizer):

    def copy_instance(self):
        return RMSProp(self.decay_rate, self.delta)

    def __init__(self, decay_rate=0.9, delta=1e-6):
        super().__init__()
        self.delta = delta
        self.decay_rate = decay_rate

    def _calculate(self, grad, learning_rate, r):
        grad_grad = (1.0 - self.decay_rate) * np.multiply(grad, grad)
        if r is None:
            r = grad_grad
        else:
            r = r * self.decay_rate + grad_grad
        denominator = self.delta + np.sqrt(r)
        return - np.multiply(np.divide(learning_rate, denominator), grad), r


class Adam(Optimizer):
    def copy_instance(self):
        return Adam(self.p1, self.p2, self.delta)

    def __init__(self, p1=0.9, p2=0.999, delta=1e-8):
        super().__init__()
        self.p1 = p1
        self.p2 = p2
        self.delta = delta

    def _calculate(self, grad, learning_rate, params):
        if params is None:
            params = [0, np.zeros(grad.shape), np.zeros(grad.shape)]
        t, s, r = params
        t += 1
        s = self.p1 * s + (1.0 - self.p1) * grad
        r = self.p2 * r + (1.0 - self.p2) * np.multiply(grad, grad)

        s_hat = np.divide(s, 1.0 - self.p1 ** t)
        r_hat = np.divide(r, 1.0 - self.p2 ** t)

        gradient = np.divide(s_hat, np.sqrt(r_hat) + self.delta)
        norm = np.linalg.norm(gradient)
        if norm > 1.0:
            gradient /= norm

        return - learning_rate * gradient, [t, s, r]
