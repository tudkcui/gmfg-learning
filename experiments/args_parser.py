import argparse
import pathlib

from evaluator.graphon_evaluator import DiscretizedGraphonEvaluatorFinite
from evaluator.stochastic_evaluator import StochasticEvaluator
from games.graphons import uniform_attachment_graphon, er_graphon, ranked_attachment_graphon
from games.finite.investment import InvestmentGraphon
from games.finite.sis import SISGraphon
from simulator.graphon_simulator import DiscretizedGraphonExactSimulatorFinite
from simulator.stochastic_simulator import StochasticSimulator


def parse_args():
    parser = argparse.ArgumentParser(description="Approximate MFGs")
    parser.add_argument('--game', help='game setting')
    parser.add_argument('--graphon', help='graphon')
    parser.add_argument('--solver', help='solver', choices=['exact', 'boltzmann', 'ppo'])
    parser.add_argument('--simulator', help='simulator', choices=['exact', 'stochastic'])
    parser.add_argument('--evaluator', help='evaluator', choices=['exact', 'stochastic'])
    parser.add_argument('--eval_solver', help='eval solver', choices=['exact', 'ppo'], default='exact')

    parser.add_argument('--iterations', type=int, help='number of outer iterations', default=500)
    parser.add_argument('--total_iterations', type=int, help='number of inner solver iterations', default=5000)

    parser.add_argument('--eta', type=float, help='temperature parameter', default=1.)

    parser.add_argument('--id', type=int, help='experiment name', default=None)

    parser.add_argument('--results_dir', help='results directory')
    parser.add_argument('--exp_name', help='experiment name')
    parser.add_argument('--verbose', type=int, help='debug outputs', default=0)
    parser.add_argument('--num_alphas', type=int, help='number of discretization points', default=101)

    return parser.parse_args()


def generate_config(args):
    return generate_config_from_kw(**{
        'game': args.game,
        'graphon': args.graphon,
        'solver': args.solver,
        'simulator': args.simulator,
        'evaluator': args.evaluator,
        'eval_solver': args.eval_solver,
        'iterations': args.iterations,
        'total_iterations': args.total_iterations,
        'eta': args.eta,
        'results_dir': args.results_dir,
        'exp_name': args.exp_name,
        'id': args.id,
        'verbose': args.verbose,
        'num_alphas': args.num_alphas,
    })


def generate_config_from_kw(**kwargs):
    if kwargs['results_dir'] is None:
        kwargs['results_dir'] = "./results/"

    if kwargs['exp_name'] is None:
        kwargs['exp_name'] = "%s_%s_%s_%s_%s_0_0_%f_%d" % (
            kwargs['game'], kwargs['graphon'], kwargs['solver'], kwargs['simulator'], kwargs['evaluator'], kwargs['eta'], kwargs['num_alphas'])

    if 'id' in kwargs and kwargs['id'] is not None:
        kwargs['exp_name'] = kwargs['exp_name'] + "_%d" % (kwargs['id'])

    experiment_directory = kwargs['results_dir'] + kwargs['exp_name'] + "/"
    pathlib.Path(experiment_directory).mkdir(parents=True, exist_ok=True)

    if kwargs['game'] == 'SIS-Graphon':
        game = SISGraphon
    elif kwargs['game'] == 'Investment-Graphon':
        game = InvestmentGraphon
    else:
        raise NotImplementedError

    if kwargs['graphon'] == 'unif-att':
        graphon = uniform_attachment_graphon
    elif kwargs['graphon'] == 'rank-att':
        graphon = ranked_attachment_graphon
    elif kwargs['graphon'] == 'er':
        graphon = er_graphon
    else:
        raise NotImplementedError

    if kwargs['solver'] == 'exact' or kwargs['solver'] == 'boltzmann':
        from solver.graphon_solver import DiscretizedGraphonExactSolverFinite
        solver = DiscretizedGraphonExactSolverFinite
    elif kwargs['solver'] == 'ppo':
        from solver.ppo_solver import PPOSolver
        solver = PPOSolver
    else:
        raise NotImplementedError

    if kwargs['simulator'] == 'exact':
        simulator = DiscretizedGraphonExactSimulatorFinite
    elif kwargs['simulator'] == 'stochastic':
        simulator = StochasticSimulator
    else:
        raise NotImplementedError

    if kwargs['evaluator'] == 'exact':
        evaluator = DiscretizedGraphonEvaluatorFinite
    elif kwargs['evaluator'] == 'stochastic':
        evaluator = StochasticEvaluator
    else:
        raise NotImplementedError

    if kwargs['eval_solver'] == 'exact':
        from solver.graphon_solver import DiscretizedGraphonExactSolverFinite
        eval_solver = DiscretizedGraphonExactSolverFinite
    elif kwargs['eval_solver'] == 'ppo':
        from solver.ppo_solver import PPOSolver
        eval_solver = PPOSolver
    else:
        raise NotImplementedError

    return {
        # === Algorithm modules ===
        "game": game,
        "solver": solver,
        "simulator": simulator,
        "evaluator": evaluator,
        "eval_solver": eval_solver,

        # === General settings ===
        "iterations": kwargs['iterations'],

        # === Default module settings ===
        "game_config": {
            "graphon": graphon,
        },
        "solver_config": {
            "total_iterations": kwargs['total_iterations'],
            "eta": kwargs['eta'],
            'verbose': kwargs['verbose'],
            'num_alphas': kwargs['num_alphas'] if 'num_alphas' in kwargs else 101,
        },
        "eval_solver_config": {
            "total_iterations": kwargs['total_iterations'],
            "eta": 0,
            'verbose': kwargs['verbose'],
            'num_alphas': kwargs['num_alphas'] if 'num_alphas' in kwargs else 101,
        },
        "simulator_config": {
            'num_alphas': kwargs['num_alphas'] if 'num_alphas' in kwargs else 101,
        },
        "evaluator_config": {
            'num_alphas': kwargs['num_alphas'] if 'num_alphas' in kwargs else 101,
        },

        "experiment_directory": experiment_directory,
    }


def parse_config():
    args = parse_args()
    return generate_config(args)
