from abc import ABC, abstractmethod

from simulator.mean_fields.base import MeanField
import numpy as np


class DiscretizedGraphonMeanField(MeanField, ABC):
    """
    Models a finite mean field \mu_t(\cdot)
    """

    def __init__(self, state_space, mu_alphas, alphas):
        super().__init__(state_space)
        self.mu_alphas = mu_alphas
        self.alphas = alphas

    def evaluate_integral(self, t, f):
        return np.mean([mu.evaluate_integral(t, lambda x: f(tuple([alpha, x])))
                        for mu, alpha in zip(self.mu_alphas, self.alphas)])
