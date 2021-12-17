import pickle
import string

import matplotlib as mpl
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Discrete
from mpl_toolkits.axes_grid1 import make_axes_locatable

from experiments import args_parser
from solver.policy.finite_policy import QMaxPolicy
from solver.policy.graphon_policy import DiscretizedGraphonFeedbackPolicy


def plot():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 22,
        "font.sans-serif": ["Helvetica"],
        # "axes.linewidth": 3,
    })
    cmap = pl.cm.viridis

    i = 1
    game = 'Investment-Graphon'
    for plot_mu in [False, True]:
        for graphon in ['unif-att', 'rank-att', 'er']:
            plt.subplot(2, 3, i)
            if i <= 3:
                plt.gca().text(-0.01, 1.06, '(' + string.ascii_lowercase[i-1] + ')', transform=plt.gca().transAxes,
                    size=22, weight='bold')
            i += 1

            solver = 'boltzmann'
            alpha_idxs = range(0, 101, 1)
            eta = 0.0 if graphon != 'er' else 0.05
            graphon_label = r'$W_\mathrm{unif}$' if graphon == 'unif-att' else \
                            r'$W_\mathrm{rank}$' if graphon == 'rank-att' else \
                            r'$W_\mathrm{er}$'

            args = args_parser.generate_config_from_kw(**{
                'game': game,
                'graphon': graphon,
                'solver': 'exact' if eta == 0 and solver in ['boltzmann'] else solver,
                'simulator': 'exact',
                'evaluator': 'exact',
                'eval_solver': 'exact',
                'iterations': 250,
                'total_iterations': 500,
                'eta': eta,
                'results_dir': None,
                'exp_name': None,
                'verbose': 0,
            })

            num_timesteps = 50
            with open(args['experiment_directory'] + 'logs.pkl', 'rb') as f:
                result = pickle.load(f)
                last_means = None
                for alpha_idx in alpha_idxs:
                    if not plot_mu:
                        """ Reconstruct policy"""
                        Q_alphas = result[-1]['solver']['Q']
                        alphas = np.linspace(0, 1, 101)
                        policy = DiscretizedGraphonFeedbackPolicy(Discrete(10),
                                                                  Discrete(2),
                                                                  [
                                                                      QMaxPolicy(Discrete(10),
                                                                                 Discrete(2),
                                                                                 Qs)
                                                                      for Qs, alpha in zip(Q_alphas, alphas)
                                                                  ], alphas)

                        means = []
                        for t in range(num_timesteps):
                            mean = []
                            for x in range(10):
                                mean.append(policy.pmf(t, tuple([alphas[alpha_idx], x]))[1])
                            means.append(mean)

                        for t in range(50):
                            color = cmap(1) if np.all(means[t]) else cmap((np.argmin(means[t])) / 10) if np.any(means[t]) else cmap(0)
                            plt.plot([t, t+1], 2*[alphas[alpha_idx]], color=color, label='_nolabel_', linewidth=2)

                    else:
                        cmap = pl.cm.plasma_r
                        colors = cmap(np.linspace(0, 1, 101))
                        means = []
                        for t in range(num_timesteps):
                            mean = np.sum([result[-1]['simulator']['mus'][t][alpha_idx][x] * x for x in range(10)])
                            means.append(mean)

                        color = colors[alpha_idx]

                        plt.plot(range(50), means, color=color, label='_nolabel_', linewidth=1)
                        if last_means is not None:
                            plt.fill_between(range(50), means, last_means, color=color)
                        last_means = means

            # plt.legend()
            plt.grid('on')
            if not plot_mu:
                plt.xlabel(fr'$t$', fontsize=22)
                plt.ylabel(fr'$\alpha$', fontsize=22)
                plt.ylim([0, 1])
                plt.xlim([0, 49])
                plt.title(graphon_label)

                divider = make_axes_locatable(plt.gca())
                ax_cb = divider.new_horizontal(size="5%", pad=0.05)
                cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=mpl.colors.Normalize(vmin=-1, vmax=9),
                                                orientation='vertical')
                cb1.set_ticks([-1, 4, 9], update_ticks=True)
                plt.gcf().add_axes(ax_cb)
                plt.title(r'$\hat x$')
            else:
                plt.xlabel(fr'$t$', fontsize=22)
                plt.ylabel(r'$\sum_{x \in \mathcal X} x \mu^\alpha_t(x)$', fontsize=22)
                plt.ylim([0, 8])
                plt.xlim([0, 49])

                divider = make_axes_locatable(plt.gca())
                ax_cb = divider.new_horizontal(size="5%", pad=0.05)
                cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap, orientation='vertical')
                plt.gcf().add_axes(ax_cb)
                plt.title(r'$\alpha$')

    plt.gcf().set_size_inches(18, 6)
    plt.tight_layout(h_pad=-0.7, w_pad=-0.02)
    plt.savefig('./figures/Investment_heatmap.pdf', bbox_inches='tight', transparent=True, pad_inches=0)
    plt.savefig('./figures/Investment_heatmap.png', bbox_inches='tight', transparent=True, pad_inches=0)
    plt.show()


if __name__ == '__main__':
    plot()
