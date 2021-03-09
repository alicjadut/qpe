from error_analysis_funs import *

methods = ['qeep', 'qeep-sparse', 'pencil']

phases = [4.77144,    2.82579877, 4.88021636, 0.41070768, 3.06350016]
num_phases = len(phases)

deltas = [0.3]
max_order = 10
confidence_alpha = 0.8
confidence_beta = 0.1

cutoff = 1/ num_phases / 3

amplitudes = np.ones(num_phases)/num_phases

estimates = {}
costs = {}
for method in methods:
    estimates[method] = {}
    costs[method] = {}

for method in methods:
    for delta in deltas:
        
        print(method, delta)
        e,c  = multiorder_estimation(method,
                             phases, amplitudes,
                             delta, confidence_alpha, confidence_beta,
                             max_order, cutoff)
        estimates[method][delta] = e
        costs[method][delta] = c
        
plt.figure(figsize = (10, 6))
for i,delta in enumerate(deltas):
    plt.subplot(1,len(deltas),i+1)
    for method in methods:
        estimation_errors = get_estimation_errors(estimates[method][delta], phases)
        plt.scatter([c for c_vec in costs[method][delta] for c in c_vec ], [e for e_vec in estimation_errors for e in e_vec ], label = method)
    plt.yscale('log')
    plt.xscale('log')
    plt.title(f'$\delta = {delta}$')
    plt.legend()
plt.show()


plt.figure(figsize = (16, 12))
i = 0
for delta in deltas:
    for method in methods:
        i+=1
        plt.subplot(len(deltas), len(methods), i, projection='polar')
        plot_phase_estimates(phases, estimates[method][delta], max_order)
        plt.title(method+f', $\delta$ = {delta}')
plt.show()