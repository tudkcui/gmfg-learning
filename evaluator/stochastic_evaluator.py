import numpy as np

from games.base import MeanFieldGame
from games.mfg_wrapper import MFGGymWrapper
from simulator.mean_fields.base import MeanField
from solver.policy.base import FeedbackPolicy


class StochasticEvaluator:
    """
    Stochastic evaluation of returns in MFGs.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def evaluate(self, mfg: MeanFieldGame, mu: MeanField, pi: FeedbackPolicy, num_evals=100):
        returns = []
        for _ in range(num_evals):
            returns.append(self.run_once(mfg, mu, pi))

        return dict({
            # "eval_episode_results": returns,
            "eval_mean_returns": np.mean(returns),
            "eval_max_returns": np.max(returns),
            "eval_min_returns": np.min(returns),
        })

    def run_once(self, mfg: MeanFieldGame, mu: MeanField, policy: FeedbackPolicy):
        env = MFGGymWrapper(mfg, mu, time_obs_augment=False)
        done = 0
        reward_sum = 0
        observation = env.reset()
        while not done:
            observation, reward, done, _ = env.step(policy.act(env.t, observation))
            reward_sum = reward_sum + reward
        return reward_sum
