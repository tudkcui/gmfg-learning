import numpy as np
import torch
from gym.spaces import Discrete
from torch.distributions import Categorical

from games.graphon_mfg import FiniteGraphonMeanFieldGame


class SISGraphon(FiniteGraphonMeanFieldGame):
    """
    Models the SIS game with 1st order Euler transition probability approximation.
    """
    def __init__(self, graphon = (lambda x,y: 1-max(x,y)), infection_rate: float = 0.8, recovery_rate: float = 0.2,
                 initial_infection_prob = (lambda x: 0.5), time_steps: int = 50, cost_infection: float = 2,
                 cost_action: float = 0.5):
        self.graphon = graphon
        self.infection_rate = infection_rate
        self.recovery_rate = recovery_rate
        self.initial_infection_prob = initial_infection_prob
        self.cost_infection = cost_infection
        self.cost_action = cost_action

        def initial_state_distribution(x):
            return Categorical(probs=torch.tensor([1 - initial_infection_prob(x), initial_infection_prob(x)]))
        agent_observation_space = Discrete(2)
        agent_action_space = Discrete(2)
        super().__init__(agent_observation_space, agent_action_space, time_steps, initial_state_distribution, graphon)

    def transition_probs_g(self, t, x, u, g):
        if x[1] == 0:
            transition_prob = g.evaluate_integral(t, lambda dy: (1 - u) * self.infection_rate * dy[1])
            return np.array([1 - transition_prob, transition_prob])
        elif x[1] == 1:
            transition_prob = self.recovery_rate
            return np.array([transition_prob, 1 - transition_prob])

    def reward_g(self, t, x, u, g):
        return - self.cost_infection * x[1] - self.cost_action * u
