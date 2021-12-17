from abc import ABC, abstractmethod


class MeanField(ABC):
    r"""
    Models a mean field \mu_t(\cdot)
    """

    def __init__(self, state_space):
        self.state_space = state_space

    @abstractmethod
    def evaluate_integral(self, t, f):
        r"""
        Evaluates the integral \int f(x) \mu_t(dx)
        :param t: time t
        :param f: function f
        :return: integral \int f(x) \mu_t(dx)
        """
        pass
