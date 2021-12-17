from abc import ABC, abstractmethod

import numpy as np
import torch
from gym import spaces
from torch.distributions import Uniform

from games.base import MeanFieldGame, ChainedTupleDistribution
from simulator.mean_fields.base import MeanField
from solver.policy.graphon_policy import DiscretizedGraphonFeedbackPolicy


class GraphonMeanFieldGame(MeanFieldGame, ABC):
    """
    Models a finite graphon mean field game in discrete time by state extension with alpha (agent graph index).
    """

    def __init__(self, agent_observation_space, agent_action_space, time_steps, initial_state_distribution,
                 graphon):
        """
        Initializes
        :param agent_observation_space: observation space
        :param agent_action_space: action space
        :param time_steps: time horizon
        :param initial_state_distribution: random function returning initial state
        :param graphon: the graphon function from [0,1]^2 to [0,1]
        """
        self.graphon = graphon
        ext_obs_space = spaces.Tuple((spaces.Box(0, 1, shape=()), agent_observation_space))
        ext_isd = ChainedTupleDistribution(Uniform(torch.tensor([0.]), torch.tensor([1.])), initial_state_distribution)

        super().__init__(ext_obs_space, agent_action_space, time_steps, ext_isd)


class FiniteGraphonMeanFieldGame(GraphonMeanFieldGame, ABC):
    """
    Models a graphon mean field game with discrete, finite state space. The states are tuple(alpha, state).
    """

    def __init__(self, agent_observation_space, agent_action_space, time_steps, initial_state_distribution, graphon):
        super().__init__(agent_observation_space, agent_action_space, time_steps, initial_state_distribution, graphon)

    def next_state(self, t, x, u, mu):
        return tuple([x[0], np.random.choice(range(self.agent_observation_space[1].n), 1, None,
                                             p=self.transition_probs(t, x, u, mu)).item()])

    def next_state_g(self, t, x, u, g):
        return tuple([x[0], np.random.choice(range(self.agent_observation_space[1].n), 1, None,
                                             p=self.transition_probs_g(t, x, u, g)).item()])

    def observation(self, t, x, u, mu, next_state):
        return next_state

    def get_neighborhood_mf(self, t, x, u, mu):
        class NeighborhoodMeanField(MeanField):
            def __init__(self, state_space, graphon):
                super().__init__(state_space)
                self.graphon = graphon

            def evaluate_integral(self, t_inner, f):
                assert t == t_inner
                return mu.evaluate_integral(t_inner, lambda dy: self.graphon(x[0], dy[0]) * f(dy))

        return NeighborhoodMeanField(mu.state_space, self.graphon)

    @abstractmethod
    def transition_probs_g(self, t, x, u, g):
        pass

    @abstractmethod
    def reward_g(self, t, x, u, g):
        pass

    def reward(self, t, x, u, mu):
        return self.reward_g(t, x, u, self.get_neighborhood_mf(t, x, u, mu))

    def transition_probs(self, t, x, u, mu: MeanField):
        """
        Returns the row of the transition probability matrix in state x if using action u under mean field ensemble mu
        :param t: time t
        :param x: extended state x, i.e. tuple (alpha, x)
        :param u: action u
        :param mu: mean field mu
        :return: row of the transition probability matrix
        """
        return self.transition_probs_g(t, x, u, self.get_neighborhood_mf(t, x, u, mu))

    def transition_probability_matrix(self, t, alpha, policy: DiscretizedGraphonFeedbackPolicy, mu: MeanField):
        """
        Returns the full transition probability matrix if using policy u under mean field ensemble mu
        :param t: time t
        :param alpha: graphon index
        :param policy: policy u
        :param mu: mean field mu
        :return: the transition probability matrix
        """
        return np.sum([np.array([policy.pmf(t, tuple([alpha, x]))[u] * self.transition_probs(t, tuple([alpha, x]), u, mu)
                                 for x in range(self.agent_observation_space[1].n)])
                       for u in range(policy.action_space.n)], axis=0)
