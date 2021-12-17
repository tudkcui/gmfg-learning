import pickle
import string

import matplotlib as mpl
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from experiments import args_parser
from simulator.stochastic_simulator import StochasticSimulator


def plot():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 22,
        "font.sans-serif": ["Helvetica"]})
    cmap = pl.cm.plasma_r

    i = 1
    game = 'Investment-Graphon'
    for plot_mu in [False, True]:
        for graphon in ['unif-att', 'rank-att', 'er']:
            plt.subplot(2, 3, i)
            if i <= 3:
                plt.gca().text(-0.01, 1.06, '(' + string.ascii_lowercase[i-1] + ')', transform=plt.gca().transAxes,
                    size=22, weight='bold')
            i += 1

            alphas = np.linspace(0, 1, 250)
            colors = cmap(np.linspace(0, 1, 250))
            eta = 0.01
            graphon_label = r'$W_\mathrm{unif}$' if graphon == 'unif-att' else \
                            r'$W_\mathrm{rank}$' if graphon == 'rank-att' else \
                            r'$W_\mathrm{er}$'

            means_per_id = []
            args = args_parser.generate_config_from_kw(**{
                'id': 0,
                'game': game,
                'graphon': graphon,
                'solver': 'ppo',
                'simulator': 'stochastic',
                'evaluator': 'stochastic',
                'eval_solver': 'ppo',
                'iterations': 10,
                'total_iterations': 50,
                'eta': eta,
                'results_dir': None,
                'exp_name': None,
                'verbose': 0,
            })
            with open(args['experiment_directory'] + 'logs.pkl', 'rb') as f:
                result = pickle.load(f)

            """ Reconstruct policy"""
            mfg = args["game"](**args["game_config"])
            solver = args["solver"](**args["solver_config"])
            pi, _ = solver.load_from_checkpoint(mfg, result[-1]['solver']['rllib_saved_chkpt'])

            if not plot_mu:
                means_alpha = []
                for alpha in alphas:
                    means = []
                    for t in range(50):
                        mean = pi.pmf(t, tuple([alpha, 0]))[1]
                        means.append(mean)
                    means_alpha.append(means)
                means_per_id.append(means_alpha)
            else:
                simulator = StochasticSimulator()
                mu, _ = simulator.simulate(mfg, pi)

                means_alpha = []
                for alpha in alphas:
                    means = []
                    for t in range(50):
                        mean = mu.evaluate_integral(t, lambda dy: dy[1] * (np.abs(alpha - dy[0]) < 0.05)) / \
                               (1e-10 + mu.evaluate_integral(t, lambda dy: (np.abs(alpha - dy[0]) < 0.05)))
                        means.append(mean)
                    means_alpha.append(means)
                means_per_id.append(means_alpha)

            for alpha_idx in range(len(alphas)):
                color = colors[alpha_idx]
                mean = np.mean(means_per_id, axis=0)[alpha_idx]
                plt.plot(range(50), mean, color=color, label='_nolabel_', alpha=0.85)

            # plt.legend()
            plt.grid('on')
            if not plot_mu:
                plt.xlabel(fr'$t$', fontsize=22)
                plt.ylabel(fr'$\pi^\alpha_t(I \mid x = 0)$', fontsize=22)
                plt.ylim([-0.05, 1.05])
                plt.xlim([0, 49])
                plt.title(graphon_label)
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
    plt.savefig('./figures/Investment_PPO.pdf', bbox_inches='tight', transparent=True, pad_inches=0)
    plt.savefig('./figures/Investment_PPO.png', bbox_inches='tight', transparent=True, pad_inches=0)
    plt.show()


if __name__ == '__main__':
    plot()
