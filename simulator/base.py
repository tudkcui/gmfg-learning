from abc import ABC, abstractmethod

from games.base import MeanFieldGame


class Simulator(ABC):
    """
    Models a simulator that generates the mean fields for a given feedback policy and game.
    """

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def simulate(self, game: MeanFieldGame, policy):
        """
        Simulate mean field, e.g. by lots of realizations
        :param game: game
        :param policy: feedback policy
        :return: tuple of mean field and info
        """
        pass
