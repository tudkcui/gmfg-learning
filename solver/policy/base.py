from abc import ABC, abstractmethod


class FeedbackPolicy(ABC):
    """
    Implements a feedback policy.
    """

    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space

    @abstractmethod
    def act(self, t, x):
        """
        At time t, act on observation x to obtain action u
        :param t: time
        :param x: observation
        :return: action
        """
        pass
