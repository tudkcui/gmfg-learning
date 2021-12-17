from collections import OrderedDict, Counter

from gym.spaces import Discrete

from simulator.mean_fields.base import MeanField


class NormalizedParticleMeanField(MeanField):
    """
    Implements a state marginal at a specific time step using samples
    """

    def __init__(self, state_space, particles, normalizer):
        super().__init__(state_space)
        self.normalizer = normalizer
        if isinstance(self.state_space, Discrete):
            self.masses = [0 if self.normalizer == 0 else
                           sum(map(lambda x: x == s, particles)) / self.normalizer
                           for s in range(self.state_space.n)]
        else:
            self.particles = particles

    def evaluate_integral(self, t, f):
        if isinstance(self.state_space, Discrete):
            return sum([f(x) * self.masses[x] for x in range(len(self.masses))])
        elif self.normalizer == 0:
            return 0
        else:
            return sum([f(x) for x in self.particles]) / self.normalizer


class ParticleMeanField(NormalizedParticleMeanField):
    def __init__(self, state_space, particles):
        super().__init__(state_space, particles, len(particles))
