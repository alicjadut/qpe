import numpy as np

n_batches = 10
n_rep = 5
n_phases = 4

rng = np.random.RandomState(42)
phases = rng.uniform(0, 2*np.pi, size=(n_batches,n_rep,n_phases))
np.save('phases', phases)