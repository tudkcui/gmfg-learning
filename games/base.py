from abc import ABC, abstractmethod

from torch.distributions import Categorical


class ChainedTupleDistribution:
    def __init__(self, dist1, dist2):
        self.dist1 = dist1
        self.dist2 = dist2


class MeanFieldGame(ABC):
    """
    Models a mean field game in discrete time plus the corresponding induced MDP for a representative agent.
    """

    def __init__(self, agent_observation_space, agent_action_space, time_steps, initial_state_distribution):
        """
        Initializes
        :param agent_action_space: action space
        :param agent_observation_space: observation space
        :param time_steps: time horizon
        :param initial_state_distribution: Torch distribution
        """
        self.agent_observation_space = agent_observation_space
        self.agent_action_space = agent_action_space
        self.time_steps = time_steps
        self.initial_state_distribution = initial_state_distribution

    def sample_state(self, dist):
        if isinstance(dist, tuple):
            return tuple([self.sample_state(d) for d in dist])
        elif isinstance(dist, ChainedTupleDistribution):
            x = self.sample_state(dist.dist1)
            y = self.sample_state(dist.dist2(x))
            return x, y
        elif isinstance(dist, Categorical):
            return dist.sample().numpy().item()
        else:
            return dist.sample().numpy().squeeze()

    def sample_initial_state(self):
        return self.sample_state(self.initial_state_distribution)

    @abstractmethod
    def next_state(self, t, x, u, mu):
        pass

    @abstractmethod
    def observation(self, t, x, u, mu, next_state):
        pass

    @abstractmethod
    def reward(self, t, x, u, mu):
        pass

    def done(self, t, x, u, mu):
        return t >= self.time_steps - 1
