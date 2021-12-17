import numpy as np

from solver.policy.finite_policy import FiniteFeedbackPolicy


class RandomFinitePolicy(FiniteFeedbackPolicy):
    """
    Random action for discrete action spaces.
    """

    def __init__(self, state_space, action_space, weights=None):
        super().__init__(state_space, action_space)
        self.weights = weights

    def pmf(self, t, x):
        if self.weights is None:
            return np.ones(self.action_space.n) / self.action_space.n
        else:
            return self.weights
