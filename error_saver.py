import os
import sys
if len(sys.argv) < 9:
    raise(Exception(f"usage: python {sys.argv[0]} "
                    "<method> <n_phases> <eps> <eps0> <final_error> <n_sim> <rep> <data_dir>"))

method = sys.argv[1]
num_phases = int(sys.argv[2])
eps = float(sys.argv[3])
eps0 = float(sys.argv[4])
final_error = float(sys.argv[5])
num_repetitions = int(sys.argv[6])
rep = int(sys.argv[7])
data_dir = sys.argv[8]

alpha = 10
gamma = 1.5
cutoff = 1/3/num_phases

from error_analysis_funs import *

rng = np.random.RandomState(42+rep)

costs, est_errors, failure_booleans = run_estimation_errors(
    [final_error], method, num_phases, eps, eps0, alpha, gamma, cutoff, num_repetitions, rng)


np.save(data_dir+f'/errors_costs_failures_{method}_phases_{num_phases}_eps_{eps}_eps0_{eps0}_final-error_{final_error}_rep_{rep}',
        (est_errors, costs, failure_booleans),
        allow_pickle = True)