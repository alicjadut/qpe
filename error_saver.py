import os
import sys
if len(sys.argv) < 9:
    raise(Exception(f"usage: python {sys.argv[0]} "
                    "<method> <n_phases> <eps> <final_error> <n_sim> <rep> <data_dir>"))

method = sys.argv[1]
num_phases = int(sys.argv[2])
eps = float(sys.argv[3])
final_error = float(sys.argv[4])
num_repetitions = int(sys.argv[5])
rep = int(sys.argv[6])
data_dir = sys.argv[7]

alpha = 2
gamma = 2.4
cutoff = 1/3/num_phases

from error_analysis_funs import *

rng = np.random.RandomState(42+rep)

costs, est_errors, failure_booleans = run_estimation_errors(
    [final_error], method, num_phases, eps, alpha, gamma, cutoff, num_repetitions, rng)


np.save(data_dir+f'/errors_costs_failures_{method}_phases_{num_phases}_eps_{eps}_final-error_{final_error}_rep_{rep}',
        (est_errors, costs, failure_booleans),
        allow_pickle = True)