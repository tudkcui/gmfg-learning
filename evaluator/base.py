from abc import abstractmethod, ABC

from games.base import MeanFieldGame
from simulator.mean_fields.base import MeanField
from solver.policy.base import FeedbackPolicy


class PolicyEvaluator(ABC):
    """
    Implements an evaluator for the performance of policies.
    """

    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, game: MeanFieldGame, mu: MeanField, policy: FeedbackPolicy):
        """
        Evaluates the expected return of the policy under mean field mu in the game.
        :return: tuple of expected return, additional info dict
        """
        pass
