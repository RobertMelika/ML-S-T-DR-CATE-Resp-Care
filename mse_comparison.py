from datasets import *
from models import *
import numpy as np
from s_t_dr_learners import learnGB, learnLinear, learnNeuralNetwork


# sample_sizes = [1000, 5000, 10000, 20000]
# sample_sizes = [1000, 5000, 10000, 15000]
sample_sizes = [1000, 5000, 10000]


def run_simulation(simulation, iterations, type):
    s_mse = []
    t_mse = []
    dr_mse = []

    for n_samples in sample_sizes:
        s_errors = []
        t_errors = []
        dr_errors = []

        for iteration in range(iterations):
            # np.random.seed(iteration * 10 + 2)
            print(f"Simulation {simulation.__name__} --- Iteration {iteration + 1}")

            s_mse_test, t_mse_test, dr_mse_test = type(simulation, n_samples)

            s_errors.append(s_mse_test)
            t_errors.append(t_mse_test)
            dr_errors.append(dr_mse_test)


        s_mse.append(np.mean(s_errors))
        t_mse.append(np.mean(t_errors))
        dr_mse.append(np.mean(dr_errors))

    return s_mse, t_mse, dr_mse
