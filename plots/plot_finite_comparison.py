import itertools
import pickle
import string

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

from experiments import args_parser
from solver.policy.finite_policy import QSoftMaxPolicy, QMaxPolicy
from solver.policy.graphon_policy import DiscretizedGraphonFeedbackPolicy


def run_once(simulator, fixed_alphas):
    done = 0
    simulator.reset(fixed_alphas)
    alphas = [simulator.x[i][0] for i in range(len(simulator.x))]
    returns = np.zeros_like(alphas)
    while not done:
        _, rewards, done, _ = simulator.step()
        returns += np.array(rewards)
    return alphas, returns


def plot():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 22,
        "font.sans-serif": ["Helvetica"]})

    i = 1
    for game in ['SIS-Graphon', 'Investment-Graphon']:
    # for game in ['Investment-Graphon']:
        for graphon in ['unif-att', 'rank-att', 'er']:
        # for graphon in ['unif-att']:
            clist = itertools.cycle(cycler(color='rbgcmyk'))
            plt.subplot(2, 3, i)
            if i <= 3:
                plt.gca().text(-0.01, 1.06, '(' + string.ascii_lowercase[i-1] + ')', transform=plt.gca().transAxes,
                    size=22, weight='bold')
            i += 1

            for fixed_alpha in [False]:
                solver = 'boltzmann'
                if game == 'SIS-Graphon':
                    eta = 0.101 if graphon != 'rank-att' else 0.3
                else:
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
                    'average_policy': False,
                    'average_mf': False,
                    'eta': eta,
                    'results_dir': None,
                    'exp_name': None,
                    'verbose': 0,
                    'prior_iteration_c': None,
                    'prior_iterations': None,
                })

                with open(args['experiment_directory'] + 'logs.pkl', 'rb') as f:
                    result = pickle.load(f)

                """ Reconstruct policy """
                mfg = args["game"](**args["game_config"])
                Q_alphas = result[-1]['solver']['Q']
                alphas = np.linspace(0, 1, 101)
                policy = DiscretizedGraphonFeedbackPolicy(mfg.agent_observation_space[1],
                                                          mfg.agent_action_space,
                                                          [
                                                              QSoftMaxPolicy(mfg.agent_observation_space[1],
                                                                             mfg.agent_action_space,
                                                                             Qs,
                                                                             1 / eta) if eta > 0 else
                                                              QMaxPolicy(mfg.agent_observation_space[1],
                                                                         mfg.agent_action_space,
                                                                         Qs)
                                                              for Qs, alpha in zip(Q_alphas, alphas)
                                                          ], alphas)

                for run_id in range(0, 5):
                    num_players_points = range(4, 104, 4)

                    mean_deviations_in_return = []
                    max_deviations_in_return = []
                    for num_players in num_players_points:

                        try:
                            if fixed_alpha:
                                with open(args['experiment_directory'] + f'nagent_fixed_seeded_{run_id}_{num_players}.pkl', 'rb') as f:
                                    returns_and_alphas = pickle.load(f)
                            else:
                                with open(args['experiment_directory'] + f'nagent_seeded_{run_id}_{num_players}.pkl', 'rb') as f:
                                    returns_and_alphas = pickle.load(f)
                        except FileNotFoundError:
                            pass

                        mf_returns_of_each_agent = []
                        for alpha in returns_and_alphas[0][1]:
                            alpha_bin = (np.abs(policy.alphas - alpha)).argmin()
                            mf_returns_of_each_agent.append(result[-1]['eval_pi']['eval_mean_returns_alpha'][alpha_bin]) # J_alpha_i for alpha_i s

                        mean_returns_of_each_agent = np.mean([returns_and_alphas[i][0] for i in range(len(returns_and_alphas))], axis=0)

                        max_deviations_in_return.append(np.max(np.abs(mean_returns_of_each_agent - mf_returns_of_each_agent)))
                        mean_deviations_in_return.append(np.mean(np.abs(mean_returns_of_each_agent - mf_returns_of_each_agent)))

                    label = f'Equidistant sampling {run_id}' if fixed_alpha else f'W-random sampling {run_id}'
                    # label_mean = 'Mean deviation equidistant' if fixed_alpha else 'Mean deviation W-random'
                    color = clist.__next__()['color']
                    plt.plot(num_players_points, max_deviations_in_return, color=color, label=label, alpha=0.85)
                    # plt.plot(num_players_points, mean_deviations_in_return, '-.', color=color, label=label_mean, alpha=0.85)
                    plt.plot(num_players_points, [0] * len(num_players_points), '-.', color='white', label='__nolabel__', alpha=0.0)

                # plt.legend()
                plt.grid('on')
                plt.xlabel(fr'$N$', fontsize=22)
                plt.ylabel(r'$\max_{i }|J_i^N - J_{\alpha_i}|$', fontsize=22)
                plt.title(game + ', ' + graphon_label)
                # plt.ylim([0, 10])

            # plt.plot(range(1, 120), [40 / np.sqrt(i).item() for i in range(1, 120)])
            # plt.plot(range(1, 120), [25 / np.sqrt(i).item() for i in range(1, 120)])

    plt.gcf().set_size_inches(18, 6)
    plt.tight_layout(h_pad=-0.1, w_pad=-0.01)
    plt.savefig('./figures/nagent_compare_test.pdf', bbox_inches='tight', transparent=True, pad_inches=0)
    plt.show()


if __name__ == '__main__':
    plot()
