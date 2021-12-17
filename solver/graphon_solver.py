import numpy as np

from games.graphon_mfg import FiniteGraphonMeanFieldGame
from simulator.mean_fields.base import MeanField
from solver.base import Solver
from solver.policy.finite_policy import QMaxPolicy, QSoftMaxPolicy
from solver.policy.graphon_policy import DiscretizedGraphonFeedbackPolicy


class DiscretizedGraphonExactSolverFinite(Solver):
    """
    Exact solutions for finite state spaces
    """

    def __init__(self, eta=0, num_alphas=101, **kwargs):
        super().__init__(**kwargs)
        self.eta = eta
        self.num_alphas = num_alphas
        self.alphas = np.linspace(0, 1, self.num_alphas)

    def solve(self, mfg: FiniteGraphonMeanFieldGame, mu: MeanField):
        Q_alphas = []

        for alpha in self.alphas:
            Vs = []
            Qs = []
            curr_V = [0 for _ in range(mfg.agent_observation_space[1].n)]

            for t in range(mfg.time_steps).__reversed__():
                Q_t = []
                for x in range(mfg.agent_observation_space[1].n):
                    x = tuple([alpha, x])
                    Q_tx = np.array([mfg.reward(t, x, u, mu) + (1 - mfg.done(t, x, u, mu)) *
                                     np.vdot(curr_V, mfg.transition_probs(t, x, u, mu))
                                     for u in range(mfg.agent_action_space.n)])
                    Q_t.append(Q_tx)
                curr_V = [np.max(Q_t[x]) for x in range(len(curr_V))]

                Vs.append(curr_V)
                Qs.append(Q_t)

            Vs.reverse()
            Qs.reverse()
            Q_alphas.append(Qs)

        def get_policy(Qs):
            if self.eta != 0:
                return QSoftMaxPolicy(mfg.agent_observation_space, mfg.agent_action_space, Qs, 1 / self.eta)
            else:
                return QMaxPolicy(mfg.agent_observation_space, mfg.agent_action_space, Qs)

        policy = DiscretizedGraphonFeedbackPolicy(mfg.agent_observation_space, mfg.agent_action_space,
                                                  [
                                                      get_policy(Qs)
                                                      for Qs, alpha in zip(Q_alphas, self.alphas)
                                                  ], self.alphas)

        return policy, {"Q": Q_alphas}
