import numpy as np
from gym.spaces import Box, Tuple

from games.graphon_mfg import FiniteGraphonMeanFieldGame
from simulator.mean_fields.particle_mf import NormalizedParticleMeanField
from solver.policy.base import FeedbackPolicy


class UniformGraphonNPlayerSimulator:
    """
    Instantiates a finite-N game from discrete Graphon Mean Field Games where all N-1 other players play the MFE.
    """

    def __init__(self, mfg: FiniteGraphonMeanFieldGame, policy: FeedbackPolicy, n_players, time_obs_augment=True, **kwargs):
        self.mfg = mfg
        self.time_obs_augment = time_obs_augment

        self.action_space = mfg.agent_action_space
        self.observation_space = mfg.agent_observation_space

        if self.time_obs_augment:
            self.observation_space = Tuple([self.observation_space, Box(0, self.mfg.time_steps, shape=())])

        self.policy = policy
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

    def reset(self, fixed_alphas=None):
        self.x = [self.mfg.sample_initial_state() for _ in range(self.n_players)]
        self.t = 0

        if fixed_alphas is not None:
            for i in range(self.n_players):
                self.x[i] = tuple([fixed_alphas[i], self.x[i][1]])

        self.adj_matrix = np.zeros([self.n_players, self.n_players])
        for i in range(self.n_players):
            for j in range(self.n_players):
                if j > i:
                    edge_prob = self.mfg.graphon(self.x[i][0], self.x[j][0])
                    edge = np.random.choice([0, 1], p=[1-edge_prob, edge_prob])
                    self.adj_matrix[i][j] = edge
                    self.adj_matrix[j][i] = edge

        return None

    def render(self, mode='human'):
        pass

    def get_masked_mu(self, x, node_index):
        masked_x = [x[i] for i in np.where(self.adj_matrix[node_index])[0]]
        return NormalizedParticleMeanField(self.mfg.agent_observation_space, masked_x, self.n_players)

    def step(self):
        next_state = []
        observations = []
        rewards = []
        for node_index in range(self.n_players):
            G = self.get_masked_mu(self.x, node_index)
            u = self.policy.act(self.t, self.x[node_index])
            next_state.append(self.mfg.next_state_g(self.t, self.x[node_index], u, G))

            reward = self.mfg.reward_g(self.t, self.x[node_index], u, G)
            observation = self.augment_obs(self.t + 1, self.mfg.observation(self.t, self.x[node_index], u, G,
                                                                        next_state[node_index]))
            rewards.append(reward)
            observations.append(observation)

        done = self.mfg.done(self.t, None, None, None)

        self.x = next_state
        self.t = self.t + 1
        return observations, rewards, done, {}
