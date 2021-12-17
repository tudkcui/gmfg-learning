import numpy as np
import torch
from gym.spaces import Discrete
from torch.distributions import Categorical

from games.graphon_mfg import FiniteGraphonMeanFieldGame


class InvestmentGraphon(FiniteGraphonMeanFieldGame):
    """
    Models the Investment game.
    """

    def __init__(self, graphon = (lambda x,y: 1-max(x,y)), time_steps: int = 50, num_qualities: int = 10,
                 payoff_quality: float = 0.3, cost_investment: float = 2):
        self.graphon = graphon
        self.payoff_quality = payoff_quality
        self.cost_investment = cost_investment
        self.num_qualities = num_qualities

        def initial_state_distribution(x):
            return Categorical(probs=torch.tensor([1] + [0] * (num_qualities-1)))
        agent_observation_space = Discrete(num_qualities)
        agent_action_space = Discrete(2)
        super().__init__(agent_observation_space, agent_action_space, time_steps, initial_state_distribution, graphon)

    def transition_probs_g(self, t, x, u, g):
        if u == 0 or x[1] == (self.num_qualities - 1):
            probs = np.zeros(self.num_qualities)
            probs[x[1]] = 1
            return probs
        elif u == 1:
            probs = np.zeros(self.num_qualities)
            probs[x[1]] = (1 + x[1]) / self.num_qualities
            probs[x[1] + 1] = (self.num_qualities - x[1] - 1) / self.num_qualities
            return probs

    def reward_g(self, t, x, u, g):
        average_quality = g.evaluate_integral(t, lambda dy: dy[1])
        return self.payoff_quality * x[1] / (1 + average_quality) \
               - self.cost_investment * u
