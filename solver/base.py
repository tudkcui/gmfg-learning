from abc import ABC, abstractmethod

from games.base import MeanFieldGame
from simulator.mean_fields.base import MeanField


class Solver(ABC):
    """
    Optimal Control Solver
    """
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def solve(self, mfg: MeanFieldGame, mu: MeanField):
        """
        Solves the optimal control problem for fixed mean field mu
        :param mfg: game to solve
        :param mu: mean field mu
        :return: tuple of optimal feedback policy and info
        """
        pass
