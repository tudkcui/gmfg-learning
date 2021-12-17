from abc import ABC, abstractmethod

from simulator.mean_fields.base import MeanField


class FiniteMeanField(MeanField, ABC):
    r"""
    Models a finite mean field \mu_t(\cdot)
    """

    @abstractmethod
    def pmf(self, t):
        """
        Evaluates the finite state marginal vector at time t
        :param t: time t
        :return: vector of state fractions
        """
        pass

    def evaluate_integral(self, t, f):
        mu = self.pmf(t)
        return sum([f(x) * mu[x] for x in range(len(mu))])


class ConstantFiniteMeanField(FiniteMeanField):
    r"""
    Models a constant finite mean field \mu_t(\cdot)
    """

    def __init__(self, state_space, mu):
        super().__init__(state_space)
        self.mu_0 = mu

    def pmf(self, t):
        return self.mu_0


class ExactFiniteMeanField(FiniteMeanField):
    def __init__(self, state_space, mus):
        super().__init__(state_space)
        self.mus = mus

    def pmf(self, t):
        return self.mus[t]
