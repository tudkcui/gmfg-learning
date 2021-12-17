import time

from solver.policy.random_policy import RandomFinitePolicy


def run_experiment(**config):
    """ Initialize """
    game = config["game"](**config["game_config"])
    simulator = config["simulator"](**config["simulator_config"])
    evaluator = config["evaluator"](**config["evaluator_config"])
    solver = config["solver"](**config["solver_config"])
    eval_solver = config["eval_solver"](**config["eval_solver_config"])

    logs = []

    """ Initial mean field and policy """
    print("Initializing. ", flush=True)
    policy = RandomFinitePolicy(game.agent_observation_space, game.agent_action_space)
    mu, info = simulator.simulate(game, policy)
    print("Initialized. ", flush=True)

    """ Outer iterations """
    for i in range(config["iterations"]):

        """ New policy """
        log = {}
        t = time.time()
        print("Loop {}: {} Now solving for policy. ".format(i, time.time()-t), flush=True)
        policy, info = solver.solve(game, mu)
        if i >= config["iterations"]-3:
            log = {**log, "solver": info}

        """ New mean field """
        print("Loop {}: {} Now simulating mean field. ".format(i, time.time()-t), flush=True)
        mu, info = simulator.simulate(game, policy)
        if i >= config["iterations"]-3:
            log = {**log, "simulator": info}

        """ Evaluation of the policy under its induced mean field """
        print("Loop {}: {} Now solving for best response. ".format(i, time.time()-t), flush=True)
        best_response, info = eval_solver.solve(game, mu)
        if i >= config["iterations"]-3:
            log = {**log, "best_response": info}
        print("Loop {}: {} Now evaluating exploitability. ".format(i, time.time()-t), flush=True)
        eval_results_pi = evaluator.evaluate(game, mu, policy)
        eval_results_opt = evaluator.evaluate(game, mu, best_response)
        log = {**log, "eval_pi": eval_results_pi, "eval_opt": eval_results_opt}
        print("Loop {}: {} policy {} optimal {} ".format(i, time.time()-t,
                                                         eval_results_pi["eval_mean_returns"],
                                                         eval_results_opt["eval_mean_returns"]), flush=True)

        logs.append(log)

    return logs
