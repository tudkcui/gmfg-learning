import numpy as np

from evaluator.base import PolicyEvaluator
from games.graphon_mfg import FiniteGraphonMeanFieldGame
from simulator.mean_fields.base import MeanField
from solver.policy.graphon_policy import DiscretizedGraphonFeedbackPolicy


class DiscretizedGraphonEvaluatorFinite(PolicyEvaluator):
    """
    Exact solver for MDP induced by graphon MFG.
    """

    def __init__(self, num_alphas=101, **kwargs):
        super().__init__()
        self.num_alphas = num_alphas
        self.alphas = np.linspace(0, 1, self.num_alphas)

    def evaluate(self, mfg: FiniteGraphonMeanFieldGame, mu: MeanField, pi: DiscretizedGraphonFeedbackPolicy):
        values_alpha = []

        for alpha in self.alphas:
            current_values = [0 for _ in range(mfg.agent_observation_space[1].n)]
            values = []

            for t in range(mfg.time_steps).__reversed__():
                Q_t_pi = []
                for x in range(mfg.agent_observation_space[1].n):
                    x = tuple([alpha, x])
                    Q_tx_pi = [mfg.reward(t, x, u, mu) + (1 - mfg.done(t, x, u, mu)) *
                               np.vdot(current_values, mfg.transition_probs(t, x, u, mu))
                               for u in range(mfg.agent_action_space.n)]
                    Q_t_pi.append(Q_tx_pi)

                current_values = [np.vdot(Q_t_pi[x], pi.pmf(t, tuple([alpha, x]))) for x in range(len(current_values))]
                values.append(current_values)

            values.reverse()
            values_alpha.append(values)

        eval_mean_returns_alpha = [np.vdot(mfg.initial_state_distribution.dist2(alpha).probs.numpy(),
                                           values_alpha[idx][0])
                                   for idx, alpha in zip(range(len(self.alphas)), self.alphas)]

        eval_mean_returns = np.mean(eval_mean_returns_alpha)

        return dict({
            # "eval_values_pi": values,
            "eval_mean_returns": eval_mean_returns,
            "eval_mean_returns_alpha": eval_mean_returns_alpha,
        })
