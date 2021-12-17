import numpy as np
import ray
import torch
from ray.rllib.agents.ppo import ppo
from ray.rllib.models.preprocessors import get_preprocessor
from ray.tune import register_env

from games.base import MeanFieldGame
from games.mfg_wrapper import MFGGymWrapper
from simulator.mean_fields.base import MeanField
from solver.base import Solver
from solver.policy.finite_policy import FiniteFeedbackPolicy


class PPOSolver(Solver):
    """
    Approximate deterministic solutions using Rllib
    """

    def __init__(self, total_iterations=500, prior=None, eta=0, verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.prior_policy = prior
        self.eta = eta
        self.total_iterations = total_iterations
        self.verbose = verbose
        ray.init(ignore_reinit_error=True)

    def load_from_checkpoint(self, game: MeanFieldGame, checkpoint):
        def env_creator(env_config=None):
            return MFGGymWrapper(game, None, time_obs_augment=True)

        register_env("MFG-v0", env_creator)
        trainer = ppo.PPOTrainer(env="MFG-v0")
        trainer.load_checkpoint(checkpoint)

        class TrainerFeedbackPolicyPPO(FiniteFeedbackPolicy):
            def __init__(self, state_space, action_space, eta, prior_policy=None):
                super().__init__(state_space, action_space)
                self.trainer = trainer
                self.wrapper = MFGGymWrapper(game, None, time_obs_augment=True)
                self.maxq = (eta == 0)
                if not self.maxq:
                    self.tau = 1 / eta
                self.prior_policy = prior_policy
                obs_space = env_creator().observation_space
                self.prep = get_preprocessor(obs_space)(obs_space)

            def pmf(self, t, x):
                obs = self.wrapper.augment_obs(t, x)
                prepped_obs = self.prep.transform(obs)
                likelihoods = np.exp(np.array([trainer.get_policy().compute_log_likelihoods([i], torch.tensor(np.expand_dims(prepped_obs, axis=0)))
                                            for i in range(self.action_space.n)]))
                return np.squeeze(likelihoods)

        return TrainerFeedbackPolicyPPO(game.agent_observation_space, game.agent_action_space, self.eta,
                                     prior_policy=self.prior_policy), {"rllib_saved_chkpt": checkpoint}

    def solve(self, game: MeanFieldGame, mu: MeanField, **config):
        def env_creator(env_config=None):
            return MFGGymWrapper(game, mu, time_obs_augment=True)

        register_env("MFG-v0", env_creator)
        trainer = ppo.PPOTrainer(env="MFG-v0", config={
            'num_workers': 6,
            "gamma": 1,
            "entropy_coeff": 0.01,
            "clip_param": 0.2,
            "kl_target": 0.006,
        })

        logs = []
        for iteration in range(self.total_iterations):
            log = trainer.train()
            if self.verbose:
                print("Loop {} mean {} ent {}".format(iteration, log['episode_reward_mean'],
                                                      log['info']['learner']['default_policy']['entropy']))
            logs.append(log)
        checkpoint = trainer.save()
        trainer.load_checkpoint(checkpoint)

        class TrainerFeedbackPolicyPPO(FiniteFeedbackPolicy):
            def __init__(self, state_space, action_space, eta, prior_policy=None):
                super().__init__(state_space, action_space)
                self.trainer = trainer
                self.wrapper = MFGGymWrapper(game, mu, time_obs_augment=True)
                self.maxq = (eta == 0)
                if not self.maxq:
                    self.tau = 1 / eta
                self.prior_policy = prior_policy
                obs_space = env_creator().observation_space
                self.prep = get_preprocessor(obs_space)(obs_space)

            def pmf(self, t, x):
                obs = self.wrapper.augment_obs(t, x)
                prepped_obs = self.prep.transform(obs)
                likelihoods = np.exp(np.array([trainer.get_policy().compute_log_likelihoods([i], torch.tensor(np.expand_dims(prepped_obs, axis=0)))
                                            for i in range(self.action_space.n)]))
                return np.squeeze(likelihoods)

        return TrainerFeedbackPolicyPPO(game.agent_observation_space, game.agent_action_space, self.eta,
                                     prior_policy=self.prior_policy), {"rllib_saved_chkpt": checkpoint}
