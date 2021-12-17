#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from collections import namedtuple

import numpy as np
import torch

Transition = namedtuple('Transition', ['agent_index', 'all_obs', 'state', 'prev_mean_neighbor_action', 'action', 'reward', 'next_state', 'mask'])


class Storage:
    def __init__(self, memory_size, keys=None):
        if keys is None:
            keys = []
        keys = keys + ['state', 'action', 'reward', 'mask',
                       'v', 'q', 'pi', 'log_pi', 'entropy',
                       'advantage', 'ret', 'q_a', 'log_pi_a',
                       'mean', 'next_state', 'prev_mean_neighbor_action', 'all_obs', 'agent_index']
        self.keys = keys
        self.memory_size = memory_size
        self.reset()

    def feed(self, data):
        for k, v in data.items():
            if k not in self.keys:
                raise RuntimeError('Undefined key')
            getattr(self, k).append(v)

    def placeholder(self):
        for k in self.keys:
            v = getattr(self, k)
            if len(v) == 0:
                setattr(self, k, [None] * self.memory_size)

    def reset(self):
        for key in self.keys:
            setattr(self, key, [])
        self.pos = 0
        self._size = 0

    def extract(self, keys):
        data = [getattr(self, k)[:self.memory_size] for k in keys]
        data = map(lambda x: torch.cat(x, dim=0), data)
        Entry = namedtuple('Entry', keys)
        return Entry(*list(data))


class UniformReplay(Storage):
    TransitionCLS = Transition

    def __init__(self, memory_size, batch_size, n_step=1, discount=1, history_length=1, keys=None):
        super(UniformReplay, self).__init__(memory_size, keys)
        self.batch_size = batch_size
        self.n_step = n_step
        self.discount = discount
        self.history_length = history_length
        self.pos = 0
        self._size = 0

    def compute_valid_indices(self):
        indices = []
        indices.extend(list(range(self.history_length - 1, self.pos - self.n_step)))
        indices.extend(list(range(self.pos + self.history_length - 1, self.size() - self.n_step)))
        return np.asarray(indices)

    def feed(self, data):
        for k, vs in data.items():
            if k not in self.keys:
                raise RuntimeError('Undefined key')
            storage = getattr(self, k)
            pos = self.pos
            size = self.size()
            for v in vs:
                if pos >= len(storage):
                    storage.append(v)
                    size += 1
                else:
                    storage[self.pos] = v
                pos = (pos + 1) % self.memory_size
        self.pos = pos
        self._size = size

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        sampled_data = []
        while len(sampled_data) < batch_size:
            transition = self.construct_transition(np.random.randint(0, self.size()))
            if transition is not None:
                sampled_data.append(transition)
        sampled_data = zip(*sampled_data)
        sampled_data = list(map(lambda x: np.asarray(x), sampled_data))
        return Transition(*sampled_data)

    def valid_index(self, index):
        if index - self.history_length + 1 >= 0 and index + self.n_step < self.pos:
            return True
        if index - self.history_length + 1 >= self.pos and index + self.n_step < self.size():
            return True
        return False

    def construct_transition(self, index):
        if not self.valid_index(index):
            return None
        s_start = index - self.history_length + 1
        s_end = index
        if s_start < 0:
            raise RuntimeError('Invalid index')
        next_s_start = s_start + self.n_step
        next_s_end = s_end + self.n_step
        if s_end < self.pos and next_s_end >= self.pos:
            raise RuntimeError('Invalid index')

        agent_index = [self.agent_index[i] for i in range(s_end, s_end + self.n_step)]
        all_obs = [self.all_obs[i] for i in range(s_end, s_end + self.n_step)]
        state = [self.state[i] for i in range(s_start, s_end + 1)]
        prev_mean_neighbor_action = [self.prev_mean_neighbor_action[i] for i in range(s_end, s_end + self.n_step)]
        next_state = [self.state[i] for i in range(next_s_start, next_s_end + 1)]
        action = self.action[s_end]
        reward = [self.reward[i] for i in range(s_end, s_end + self.n_step)]
        mask = [self.mask[i] for i in range(s_end, s_end + self.n_step)]
        if self.history_length == 1:
            # eliminate the extra dimension if no frame stack
            state = state[0]
            next_state = next_state[0]
        state = np.array(state)
        next_state = np.array(next_state)
        cum_r = 0
        cum_mask = 1
        for i in reversed(np.arange(self.n_step)):
            cum_r = reward[i] + mask[i] * self.discount * cum_r
            cum_mask = cum_mask and mask[i]
        return Transition(agent_index=agent_index, all_obs=all_obs, state=state, prev_mean_neighbor_action=prev_mean_neighbor_action, action=action, reward=cum_r, next_state=next_state, mask=cum_mask)

    def size(self):
        return self._size

    def full(self):
        return self._size == self.memory_size

    def update_priorities(self, info):
        raise NotImplementedError
