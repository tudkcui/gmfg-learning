from games.base import MeanFieldGame
from simulator.base import Simulator
from simulator.mean_fields.average_mf import AverageMeanField
from simulator.mean_fields.composite_mf import CompositeMeanField
from simulator.mean_fields.particle_mf import ParticleMeanField
from solver.policy.base import FeedbackPolicy


class StochasticSimulator(Simulator):
    """
    Stochastic simulator for mean field simulation in MFGs.
    """

    def __init__(self, num_particles=200, num_samples=5, **kwargs):
        super().__init__(**kwargs)
        self.num_particles = num_particles
        self.num_samples = num_samples

    def simulate(self, game: MeanFieldGame, policy: FeedbackPolicy):
        mean_fields = []
        for _ in range(self.num_samples):
            mus = []

            """ Initialize """
            curr_particles = [game.sample_initial_state() for _ in range(self.num_particles)]
            curr_mu = ParticleMeanField(game.agent_observation_space, curr_particles)
            mus.append(curr_mu)

            """ Simulate one step with empirical distribution of particles """
            for t in range(game.time_steps):
                curr_particles = list(map(lambda x: game.next_state(t, x, policy.act(t, x), curr_mu), curr_particles))
                curr_mu = ParticleMeanField(game.agent_observation_space, curr_particles)
                mus.append(curr_mu)

            """ Add particle flow as sample"""
            mean_fields.append(CompositeMeanField(game.agent_observation_space, mus))

        """ Take average over all samples """
        return AverageMeanField(game.agent_observation_space, mean_fields), {}
