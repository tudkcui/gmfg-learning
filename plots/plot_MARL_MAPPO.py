import pickle

import ray
import string
import torch
from ray.rllib.agents.ppo import ppo
from ray.rllib.models.preprocessors import get_preprocessor
from ray.tune import register_env

import matplotlib as mpl
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from games.finite.sis import SISGraphon
from games.graphon_n_player_multiagent_wrapper import GraphonNPlayerMultiAgentWrapper
from games.graphons import uniform_attachment_graphon

if __name__ == '__main__':
    N_agents = 50
    def env_creator(env_config=None):
        return GraphonNPlayerMultiAgentWrapper(SISGraphon(graphon=uniform_attachment_graphon), N_agents)
    env = env_creator()

    def gen_policy():
        config = {}
        return (None, env.observation_space, env.action_space, config)
    policies = {
        "default_policy": gen_policy()
    }

    ray.init()
    # ray.init(local_mode=True)
    register_env("Env-v0", env_creator)
    trainer = ppo.PPOTrainer(env="Env-v0", config={
        "num_gpus": 0,
        'num_workers': 6,
        "gamma": 0.99,
        # "entropy_coeff": 0.01,
        "clip_param": 0.3,
        "kl_target": 0.006,
        "no_done_at_end": False,
        "framework": 'torch',
         "multiagent": {
             "policies": policies,
             "policy_mapping_fn": (lambda i: "default_policy"),
         },
    })

    logs = []
    for iteration in range(250):
        log = trainer.train()
        print("Loop {} mean {} log {}".format(iteration, log['episode_reward_mean'], log))
        if iteration % 50 == 0:
            checkpoint = trainer.save()
        logs.append(log)

    checkpoint = trainer.save()
    trainer.load_checkpoint(checkpoint)

    obs_space = env_creator().observation_space
    prep = get_preprocessor(obs_space)(obs_space)

    def pmf(t, x):
        obs = env.augment_obs(t, x)
        prepped_obs = prep.transform(obs)
        likelihoods = np.exp(np.array(
            trainer.get_policy().compute_log_likelihoods(
                torch.tensor(list(range(2))), torch.tensor(np.expand_dims(prepped_obs, axis=0)))
        ))
        return np.squeeze(likelihoods)

    """ Plot """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 22,
        "font.sans-serif": ["Helvetica"]})
    cmap = pl.cm.plasma_r

    plt.subplot(1, 2, 1)
    plt.plot(range(len(logs)), [logs[i]['episode_reward_mean'] for i in range(len(logs))], label='_nolabel_', alpha=1)
    plt.grid('on')
    plt.xlabel(fr'Iterations', fontsize=22)
    plt.ylabel(fr'Sum of agent returns', fontsize=22)
    plt.xlim([0, len(logs)-1])

    plt.subplot(1, 2, 2)
    alphas = np.linspace(0, 1, N_agents)
    colors = cmap(np.linspace(0, 1, N_agents))
    eta = 0.01

    means_per_id = []

    means_alpha = []
    for alpha in alphas:
        means = []
        for t in range(50):
            mean = pmf(t, tuple([alpha, 0]))[1]
            means.append(mean)
        means_alpha.append(means)
    means_per_id.append(means_alpha)
    # else:
    # simulator = StochasticSimulator()
    # mu, _ = simulator.simulate(mfg, pi)
    #
    # means_alpha = []
    # for alpha in alphas:
    #     means = []
    #     for t in range(50):
    #         mean = mu.evaluate_integral(t, lambda dy: dy[1] * (np.abs(alpha - dy[0]) < 0.05)) / \
    #                (1e-10 + mu.evaluate_integral(t, lambda dy: (np.abs(alpha - dy[0]) < 0.05)))
    #         means.append(mean)
    #     means_alpha.append(means)
    # means_per_id.append(means_alpha)
    # pass

    for alpha_idx in range(len(alphas)):
        color = colors[alpha_idx]
        mean = np.mean(means_per_id, axis=0)[alpha_idx]
        plt.plot(range(50), mean, color=color, label='_nolabel_', alpha=0.85)

    # plt.legend()
    plt.grid('on')
    plt.xlabel(fr'$t$', fontsize=22)
    plt.ylabel(fr'$\pi^\alpha_t(I \mid x = 0)$', fontsize=22)
    plt.ylim([-0.05, 1.05])
    plt.xlim([0, 49])

    divider = make_axes_locatable(plt.gca())
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap, orientation='vertical')
    plt.gcf().add_axes(ax_cb)
    plt.title(r'$\alpha$')

    plt.gcf().set_size_inches(18, 6)
    plt.tight_layout()
    plt.savefig('./figures/MARL_IL_SIS_unif.pdf', bbox_inches='tight', transparent=True, pad_inches=0)
    plt.savefig('./figures/MARL_IL_SIS_unif.png', bbox_inches='tight', transparent=True, pad_inches=0)
    plt.show()

    with open('./figures/MARL_IL_SIS_unif_logs.pkl', 'wb') as f:
        pickle.dump(logs, f, 4)
    with open('./figures/MARL_IL_SIS_unif_checkpoint.pkl', 'wb') as f:
        pickle.dump(checkpoint, f, 4)
