import numpy as np

from games.graphon_mfg import FiniteGraphonMeanFieldGame
from simulator.base import Simulator
from simulator.mean_fields.finite_mf import ConstantFiniteMeanField, ExactFiniteMeanField
from simulator.mean_fields.graphon_mf import DiscretizedGraphonMeanField
from solver.policy.graphon_policy import DiscretizedGraphonFeedbackPolicy


class DiscretizedGraphonExactSimulatorFinite(Simulator):
    """
    Exact discretized solutions for finite state spaces
    """

    def __init__(self, num_alphas=101, **kwargs):
        super().__init__(**kwargs)
        self.num_alphas = num_alphas
        self.alphas = np.linspace(0, 1, self.num_alphas)

    def simulate(self, game: FiniteGraphonMeanFieldGame, policy: DiscretizedGraphonFeedbackPolicy):
        mus = []
        mu_alphas_curr = []
        for alpha in self.alphas:
            mu_alphas_curr.append(game.initial_state_distribution.dist2(alpha).probs.numpy())

        for t in range(game.time_steps):
            mus.append(mu_alphas_curr)
            mu_alphas_next = []
            for idx, alpha in zip(range(len(self.alphas)), self.alphas):
                p = game.transition_probability_matrix(t, alpha, policy,
                                                       DiscretizedGraphonMeanField(game.agent_observation_space,
                                                                                   [ConstantFiniteMeanField(
                                                                                       game.agent_observation_space,
                                                                                       mu_alpha)
                                                                                    for mu_alpha in mu_alphas_curr],
                                                                                   self.alphas))
                mu_alpha = np.matmul(mu_alphas_curr[idx], p)
                mu_alphas_next.append(mu_alpha)

            mu_alphas_curr = mu_alphas_next

        """ Reshape """
        final_mus = [
            [mus[t][alpha_idx] for t in range(game.time_steps)]
            for alpha_idx in range(self.num_alphas)
        ]

        info = {'mus': mus}
        return DiscretizedGraphonMeanField(game.agent_observation_space,
                                           [ExactFiniteMeanField(game.agent_observation_space, mus)
                                            for mus, alpha in zip(final_mus, self.alphas)], self.alphas), info
