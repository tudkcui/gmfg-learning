from abc import abstractmethod

import numpy as np

from solver.policy.base import FeedbackPolicy


class FiniteFeedbackPolicy(FeedbackPolicy):
    """
    Implements a finite action space feedback policy.
    """

    def __init__(self, state_space, action_space):
        super().__init__(state_space, action_space)

    def act(self, t, x):
        """
        At time t, act on observation x to obtain random action u
        :param t: time
        :param x: observation
        :return: action
        """
        pmf = self.pmf(t, x)
        return np.random.choice(range(len(pmf)), 1, p=pmf).item()

    @abstractmethod
    def pmf(self, t, x):
        """
        At time t, act on observation x to obtain action pmf
        :param t: time
        :param x: observation
        :return: action pmf
        """
        pass


class QMaxPolicy(FiniteFeedbackPolicy):
    def __init__(self, state_space, action_space, Qs):
        super().__init__(state_space, action_space)
        self.Qs = Qs

    def pmf(self, t, x):
        unit_vec = np.zeros(self.action_space.n)
        unit_vec[np.argmax(self.Qs[t][x])] = 1
        return unit_vec


class QSoftMaxPolicy(FiniteFeedbackPolicy):
    def __init__(self, state_space, action_space, Qs, tau):
        super().__init__(state_space, action_space)
        self.Qs = Qs
        self.tau = tau

    def pmf(self, t, x):
        Qs_norm = self.Qs[t][x] - max(self.Qs[t][x])
        return np.exp(self.tau * np.array(Qs_norm)) / sum(np.exp(self.tau * np.array(Qs_norm)))
