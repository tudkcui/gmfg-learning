import subprocess
import numpy as np

if __name__ == '__main__':
    child_processes = []

    import multiprocessing
    num_cores = multiprocessing.cpu_count()

    for graphon in ['unif-att']:
        for game in ['SIS-Graphon']:
            for solver in ['boltzmann']:
                for num_alphas in range(10, 300, 10):
                    if solver == 'exact':
                        etas = [0]
                    else:
                        etas = [0.101]

                    for eta in etas:
                        p = subprocess.Popen(['python',
                                              './experiments/run.py',
                                              '--game=' + game,
                                              '--solver=' + solver,
                                              '--simulator=exact',
                                              '--evaluator=exact',
                                              '--iterations=250',
                                              '--eta=' + '%f' % eta,
                                              '--graphon=' + graphon,
                                              '--num_alphas=%d' % num_alphas])
                        child_processes.append(p)
                        if len(child_processes) >= num_cores - 2:
                            for p in child_processes:
                                p.wait()
                            child_processes = []

    for p in child_processes:
        p.wait()
