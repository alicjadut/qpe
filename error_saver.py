import os
import sys
if len(sys.argv) < 10:
    raise(Exception(f"usage: python {sys.argv[0]} "
                    "<method> <n_phases> <eps> <final_error> <alpha> <gamma> <n_sim> <rep> <data_dir>"))

method = sys.argv[1]
num_phases = int(sys.argv[2])
eps = float(sys.argv[3])
final_error = float(sys.argv[4])
alpha = float(sys.argv[5])
gamma = float(sys.argv[6])
num_repetitions = int(sys.argv[7])
rep = int(sys.argv[8])
data_dir = sys.argv[9]

cutoff = 1/3/num_phases

from error_analysis_funs import *

rng = np.random.RandomState(42)
phases = np.load('phases.npy')
all_phases = phases[rep]

costs, est_errors, failure_booleans = run_estimation_errors(
    all_phases,
    [final_error], method, eps, alpha, gamma, cutoff, num_repetitions, rng)


np.save(data_dir+
        f'/errors_costs_failures_{method}_phases_{num_phases}_eps_{eps}_final-error_{final_error}_alpha_{alpha}_gamma_{gamma}_rep_{rep}',
        (est_errors, costs, failure_booleans),
        allow_pickle = True)