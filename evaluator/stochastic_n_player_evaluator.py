import numpy as np

from evaluator.base import PolicyEvaluator
from games.base import MeanFieldGame
from games.graphon_n_player_wrapper import GraphonNPlayerWrapper
from solver.policy.base import FeedbackPolicy


class StochasticNPlayerEvaluator(PolicyEvaluator):
    r"""
    Stochastic evaluation of returns in finite graph games.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def evaluate(self, mfg: MeanFieldGame, mfe_pi: FeedbackPolicy, pi: FeedbackPolicy, num_players=10, num_evals=100):
        returns = []
        for _ in range(num_evals):
            returns.append(self.run_once(mfg, mfe_pi, pi, num_players))

        return dict({
            "eval_mean_returns": np.mean(returns),
            "eval_max_returns": np.max(returns),
            "eval_min_returns": np.min(returns),
        })

    @staticmethod
    def run_once(mfg: MeanFieldGame, mfe_pi: FeedbackPolicy, policy: FeedbackPolicy, num_players):
        env = GraphonNPlayerWrapper(mfg, mfe_pi, num_players)
        done = 0
        reward_sum = 0
        observation = env.reset()
        while not done:
            observation, reward, done, _ = env.step(policy.act(observation[-1].item(), observation[0]))
            reward_sum = reward_sum + reward
        return reward_sum
