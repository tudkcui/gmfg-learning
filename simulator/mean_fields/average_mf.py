import numpy as np

from simulator.mean_fields.base import MeanField
from simulator.mean_fields.finite_mf import FiniteMeanField


class AverageMeanField(MeanField):
    r"""
    Models an averaged discrete mean field.
    """

    def __init__(self, state_space, mean_fields):
        super().__init__(state_space)
        self.mean_fields = mean_fields

    def evaluate_integral(self, t, f):
        return sum([mf.evaluate_integral(t, f) for mf in self.mean_fields]) / len(self.mean_fields)


class AverageFiniteMeanField(FiniteMeanField):
    """
    Implements the average policy of a list of finite policies.
    """

    def __init__(self, state_space, mean_fields, weights, timesteps):
        super().__init__(state_space)
        self.pmfs = []
        for t in range(timesteps):
            pmf = np.sum([weights[i] * mean_fields[i].pmf(t) for i in range(len(mean_fields))], axis=0)
            self.pmfs.append(pmf)

    def pmf(self, t):
        return self.pmfs[t]
