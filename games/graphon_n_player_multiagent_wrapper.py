import numpy as np
from gym import Env
from gym.spaces import Box, Tuple
from ray.rllib import MultiAgentEnv

from games.graphon_mfg import FiniteGraphonMeanFieldGame
from simulator.mean_fields.particle_mf import NormalizedParticleMeanField, ParticleMeanField
from solver.policy.base import FeedbackPolicy


class GraphonNPlayerMultiAgentWrapper(MultiAgentEnv):
    """
    Instantiates a finite-N MultiAgentEnv env from discrete graphon Mean Field Games.
    """

    def __init__(self, mfg: FiniteGraphonMeanFieldGame, n_players, time_obs_augment=True):
        self.mfg = mfg
        self.time_obs_augment = time_obs_augment

        self.action_space = mfg.agent_action_space
        self.observation_space = mfg.agent_observation_space
        self.reward_range = None
        self.metadata = None

        if self.time_obs_augment:
            self.observation_space = Tuple([self.observation_space, Box(0, self.mfg.time_steps, shape=())])

        self.n_players = n_players

        self.t = None
        self.x = None
        self.adj_matrix = None
        self.reset()

    def augment_obs(self, t, obs):
        if self.time_obs_augment:
            return tuple([obs, np.array(t)])
        else:
            return obs

    def reset(self):
        self.x = [self.mfg.sample_initial_state() for _ in range(self.n_players)]
        self.t = 0

        # """ Equidistant alpha for testing """
        # for i in range(self.n_players):
        #     self.x[i][0] = i * (1 / (self.n_players - 1))

        self.adj_matrix = np.zeros([self.n_players, self.n_players])
        for i in range(self.n_players):
            for j in range(self.n_players):
                if j > i:
                    edge_prob = self.mfg.graphon(self.x[i][0], self.x[j][0])
                    edge = np.random.choice([0, 1], p=[1-edge_prob, edge_prob])
                    self.adj_matrix[i][j] = edge
                    self.adj_matrix[j][i] = edge

        return {node_idx: self.augment_obs(self.t, self.mfg.observation(self.t, None, None, None, self.x[node_idx]))
                for node_idx in range(self.n_players)}

    def render(self, mode='human'):
        pass

    def get_masked_mu(self, x, node_index):
        masked_x = [x[i] for i in np.where(self.adj_matrix[node_index])[0]]
        return NormalizedParticleMeanField(self.mfg.agent_observation_space, masked_x, self.n_players)

    def step(self, u):
        next_state = []
        observations, rewards, dones = {}, {}, {}
        for node_index in range(self.n_players):
            G = self.get_masked_mu(self.x, node_index)
            next_state.append(self.mfg.next_state_g(self.t, self.x[node_index], u[node_index], G))
            rewards[node_index] = self.mfg.reward_g(self.t, self.x[node_index], u[node_index], G)
            observations[node_index] = self.augment_obs(self.t + 1, self.mfg.observation(self.t, self.x[node_index], u[node_index],
                                                                                         G, next_state[node_index]))
            mu = ParticleMeanField(self.mfg.agent_observation_space, self.x)
            dones[node_index] = self.mfg.done(self.t, self.x[node_index], u[node_index], mu)
            dones['__all__'] = self.mfg.done(self.t, self.x[node_index], u[node_index], mu)

        self.x = next_state
        self.t = self.t + 1
        return observations, rewards, dones, {}
