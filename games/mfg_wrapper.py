import numpy as np
from gym import Env
from gym.spaces import Box, Tuple
from numpy import float32

from games.base import MeanFieldGame
from simulator.mean_fields.base import MeanField


class MFGGymWrapper(Env):
    def __init__(self, mfg: MeanFieldGame, mu: MeanField, time_obs_augment=True):
        self.mfg = mfg
        self.mu = mu
        self.time_obs_augment = time_obs_augment

        self.action_space = mfg.agent_action_space
        self.observation_space = mfg.agent_observation_space

        if self.time_obs_augment:
            self.observation_space = Tuple([self.observation_space, Box(0, self.mfg.time_steps, shape=())])

        self.x = None
        self.t = None
        self.reset()

    def augment_obs(self, t, obs):
        if self.time_obs_augment:
            return tuple([obs, np.array(t, dtype=float32)])
        else:
            return obs

    def reset(self):
        self.x = self.mfg.sample_initial_state()
        self.t = 0
        return self.augment_obs(self.t, self.mfg.observation(self.t, None, None, self.mu, self.x))

    def render(self, mode='human'):
        pass

    def step(self, u):
        next_state = self.mfg.next_state(self.t, self.x, u, self.mu)
        game_observation = self.mfg.observation(self.t, self.x, u, self.mu, next_state)
        observation = self.augment_obs(self.t + 1, game_observation)
        reward = self.mfg.reward(self.t, self.x, u, self.mu)
        done = self.mfg.done(self.t, self.x, u, self.mu)

        self.x = next_state
        self.t = self.t + 1
        return observation, reward, done, {}
