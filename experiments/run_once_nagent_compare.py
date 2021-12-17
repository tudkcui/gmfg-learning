import pickle

import numpy as np

from experiments import args_parser
from simulator.n_player_graphon_simulator import UniformGraphonNPlayerSimulator
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


def run_config(input):
    num_players = input[0]
    game = input[1]
    graphon = input[2]
    fixed_alpha = input[3]
    id_run = input[4]

    np.random.seed(id_run)
    print(f'Running {input}')

    solver = 'boltzmann'
    if game == 'SIS-Graphon':
        eta = 0.101 if graphon != 'rank-att' else 0.3
    else:
        eta = 0.0 if graphon != 'er' else 0.05

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
    mfg = args["game"](**args["game_config"])
    with open(args['experiment_directory'] + 'logs.pkl', 'rb') as f:
        result = pickle.load(f)

    """ Reconstruct policy """
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

    num_alpha_trials = 1
    num_return_trials = 10000

    simulator = UniformGraphonNPlayerSimulator(mfg, policy, num_players)

    returns_and_alphas = []
    for _ in range(num_alpha_trials):
        """ Simulate for N agents and some sampled alphas """
        simulator.reset()
        fixed_alphas = [simulator.x[i][0] for i in range(len(simulator.x))]
        if fixed_alpha:
            fixed_alphas = np.linspace(1/num_players, 1, num_players)

        for _ in range(num_return_trials):
            alphas, returns = run_once(simulator, fixed_alphas)
            returns_and_alphas.append((returns, alphas))

        mf_returns_of_each_agent = []
        for alpha in alphas:
            alpha_bin = (np.abs(policy.alphas - alpha)).argmin()
            mf_returns_of_each_agent.append(result[-1]['eval_pi']['eval_mean_returns_alpha'][alpha_bin])

        mean_returns_of_each_agent = np.mean([returns_and_alphas[i][0] for i in range(len(returns_and_alphas))], axis=0)

    print(f'{game} {graphon} {num_players}: {np.max(np.abs(mean_returns_of_each_agent - mf_returns_of_each_agent))}', flush=True)

    if fixed_alpha:
        with open(args['experiment_directory'] + f'nagent_fixed_seeded_{id_run}_{num_players}.pkl', 'wb') as f:
            pickle.dump(returns_and_alphas, f, 4)
    else:
        with open(args['experiment_directory'] + f'nagent_seeded_{id_run}_{num_players}.pkl', 'wb') as f:
            pickle.dump(returns_and_alphas, f, 4)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run evaluation of policies in finite agent game")
    parser.add_argument('--num_players_point', type=int)
    parser.add_argument('--game')
    parser.add_argument('--graphon')
    parser.add_argument('--fixed_alphas', type=int)
    parser.add_argument('--id', type=int)
    args = parser.parse_args()

    run_config((args.num_players_point,
                args.game,
                args.graphon,
                args.fixed_alphas,
                args.id))
