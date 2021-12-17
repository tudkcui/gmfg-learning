import numpy as np

from solver.policy.base import FeedbackPolicy


class DiscretizedGraphonFeedbackPolicy(FeedbackPolicy):
    """
    Implements a finite action space graphon feedback policy.
    """

    def __init__(self, state_space, action_space, policies_alpha, alphas):
        super().__init__(state_space, action_space)
        self.policies_alpha = policies_alpha
        self.alphas = alphas

    def act(self, t, x):
        """
        At time t, act on observation x to obtain random action u
        :param t: time
        :param x: observation
        :return: action
        """
        pmf = self.pmf(t, x)
        return np.random.choice(range(len(pmf)), 1, p=pmf).item()

    def pmf(self, t, x):
        """
        At time t, act on observation x to obtain action pmf
        :param t: time
        :param x: observation
        :return: action pmf
        """
        alpha_bin = (np.abs(self.alphas - x[0])).argmin()
        return self.policies_alpha[alpha_bin].pmf(t, x[1])
