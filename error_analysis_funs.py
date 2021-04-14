import queue

#Required imports

#import cirq
#from cirq.google.ops import SYC
import sympy
from scipy import linalg, optimize
import numpy as np
from matplotlib import pyplot as plt
#from openfermion import QubitOperator, eigenspectrum, get_sparse_operator
#from openfermion import prony
from prony import prony_phases
import time, datetime, json, os

import warnings
warnings.simplefilter("always")

from qeep_estimators import (
    qeep_solve, qeep_sparse_solve, get_signal_requirements,
    qeep_approximate_single_eigenvalues, get_phase_values,
    qeep_conservative_solve)
from sparse_qpe import(
    kappa_finder, match_phases, abs_phase_difference, _wn_diff)

from tqdm import tqdm


# # Definitions

# ## Functions to produce a phase function with the appropriate noise


def add_noise(prob, num_samples, rng = np.random.RandomState(42)):
    res = rng.binomial(num_samples, prob) / num_samples
    return res

def get_gk(signal_length, phases, amplitudes, num_samples, multiplier, rng = np.random.RandomState(42)):
    gk_clean = np.array([np.sum(
        np.array(amplitudes) * np.exp(1j * np.array(phases) * k * multiplier))
                     for k in range(signal_length+1)])
    if num_samples is None:
        gk_noisy = gk_clean
    else:
        pk_real_clean = 0.5 - 0.5 * np.real(gk_clean)
        pk_imag_clean = 0.5 - 0.5 * np.imag(gk_clean)
        pk_real_noisy = add_noise(pk_real_clean, num_samples, rng)
        pk_imag_noisy = add_noise(pk_imag_clean, num_samples, rng)

        gk_noisy = (1 - 2 * pk_real_noisy) + 1j *(1 - 2 * pk_imag_noisy)
    return gk_noisy


# ## Function to perform single-order estimation

# 3 methods to consider:
# 
# - `qeep` - find the spectral function with QEEP solver. 
# - `qeep-sparse` - find spectral function with QEEP solver. Return all bin centers that have amplitude > cutoff.
# - `pencil` - find the phases with matrix pencil method.


def estimate_phases(method, signal, cutoff, num_points):
   
    if(method == 'qeep-cons'):
        spectral_function = qeep_solve(signal, num_points)
        return qeep_conservative_solve(spectral_function, cutoff)
    
    if(method == 'qeep'):
        spectral_function = qeep_solve(signal, num_points)
        # Code to perform error estimation
        # Somma's method has a way to produce a 'better' phase estimation
        # --- with the cost that it might not estimate all phases.
        # But we should still use it in the final estimation.
        return qeep_approximate_single_eigenvalues(spectral_function, cutoff)
        
    if(method == 'qeep-sparse'):
        spectral_function = qeep_solve(signal, num_points)
        return qeep_sparse_solve(spectral_function, cutoff)
        
    if(method == 'pencil'):
        return prony_phases(signal, cutoff)
    
    raise ValueError(f'Wrong method: {method}')

def shift_value(phases, eps):
    
    #zeta is the number between 2 consecutive phases that are furtherst away
    phases = np.sort(phases)
    phase_differences = [
        abs_phase_difference(phases[j], phases[(j+1) % len(phases)]) for j in range(len(phases))
    ]
    ix = np.argmax(phase_differences)
    phase1 = phases[ix]
    phase2 = phases[(ix+1) % len(phases)]
    zeta = (phase1+phase2)/2+(1-_wn_diff(phase1, phase2))*np.pi
    if np.min(
        [abs_phase_difference(phase,zeta+np.pi) for phase in phases]
    ) > np.min(
        [abs_phase_difference(phase,zeta) for phase in phases]
    ):
        zeta+=np.pi
    d_zeta = np.min(
        [abs_phase_difference(phase,zeta) for phase in phases]
    )
    shift_val = zeta+d_zeta/2-8*eps
    return shift_val

# ## Function to perform multi-order estimation

def multiorder_estimation(method,
                             phases, amplitudes,
                             eps, alpha, gamma,
                             final_error, cutoff, rng = np.random.RandomState(42)):
    
    
    estimates = []
    costs = []

    # Generate zeroth order phase estimates
    multiplier = 1
    
    # Calculate the signal requirements at this order and the assoc. cost
    confidence = 1-np.exp(-alpha)*(multiplier*final_error/np.pi)**gamma
    print(confidence, alpha, multiplier, final_error,gamma)
    num_points, signal_length, num_samples = get_signal_requirements(confidence, eps)
    cost = sum([num_samples * 2 * k * multiplier for k in range(signal_length + 1)])
    
    # Get the new signal and estimate aliased phases from this.
    gk_noisy = get_gk(signal_length, phases, amplitudes, num_samples, multiplier, rng)
    phase_estimates = estimate_phases(method, gk_noisy, cutoff, num_points)
    error_estimates = [eps for phase in phase_estimates]
    
    # Add phase estimates and costs to data
    costs.append([cost for phase in phases])
    estimates.append(list(phase_estimates))
    
    # Shift the unitary
    shift_val = shift_value(phase_estimates, eps)
    phases = (phases - shift_val) % (2*np.pi)
    phase_estimates = (phase_estimates - shift_val) % (2*np.pi)
    
    d=1
    
    #Find the first multiplier
    try:
        multiplier = kappa_finder(phase_estimates, eps, multiplier)
    except ValueError:
        print(r'Couldnt find good $k_1$, exiting')
        return estimates, costs, ('kappa', d)  
    #The first multiplier has to be larger than 1/d_zeta
    if multiplier < 3*len(phases):
        print(r'Got $k_1 < 3n_\phi$, exiting')
        return estimates, costs, ('k1', d)  
    kappas = [multiplier]
    
    
    
    while(multiplier < 2*eps/final_error):
        print(multiplier)

        
        if(d>1):
            # Calculate the new best multiplier from the previous phase data.
            # If this doesn't work, fail gracefully.
            #(d = 1 is excluded, because we have extra assumptions for it)
            try:
                kappas.append(kappa_finder(phase_estimates, eps, multiplier))
            except ValueError:
                print('Couldnt find good kappa, exiting')
                return estimates, costs, ('kappa', d)            
            multiplier = np.prod(kappas)

        # Calculate the signal requirements at this order and the assoc. cost
        confidence = 1-np.exp(-alpha)*(multiplier*final_error/np.pi)**gamma
        num_points, signal_length, num_samples = get_signal_requirements(confidence, eps)
        cost += sum([num_samples * 2 * k * multiplier for k in range(signal_length + 1)])
        
        # Get the new signal and estimate aliased phases from this.
        gk_noisy = get_gk(signal_length, phases, amplitudes, num_samples, multiplier, rng)
        aliased_phase_estimates = estimate_phases(method, gk_noisy, cutoff, num_points)
        
        #If the new estimates are not close enough to the old estimates, exit
        if(
            np.max([
                np.min(
                    [abs_phase_difference(multiplier*phi, theta)
                     for phi in phase_estimates]
                ) for theta in aliased_phase_estimates]) > 2*eps*(1+kappas[-1])
            or
            np.max([
                np.min(
                    [abs_phase_difference(multiplier*phi, theta)
                     for theta in aliased_phase_estimates]
                ) for phi in phase_estimates]) > 2*eps*(1+kappas[-1])
        ):
            print('Cannot match new estimates to old estimates, exiting')
            return estimates, costs, ('match', d)  

        # Match phases --- generate new estimates of phases at each order
        phase_estimates = match_phases(
            phase_estimates,
            multiplier,
            aliased_phase_estimates)
        
        # If we have completely failed, do it gracefully
        if len(phase_estimates) == 0:
            print('No phases left, exiting')
            return estimates, costs, ('empty', d)  
        #If the estimates are outside of the allowed region, exit
        if(
            np.min(phase_estimates) < np.pi/multiplier
            or
            np.max(phase_estimates) > np.pi*(2*np.floor(multiplier)-1)/multiplier
        ):
            print('Got phase estimates outside of the allowed region, exiting')
            return estimates, costs, ('region', d)  
        
        # Add phase estimates errors and costs to data
        costs.append([cost for phase in phases])
        estimates.append((phase_estimates+shift_val) % (2*np.pi))
        
        d+=1
            
    return estimates, costs, ('success', d)    


def get_estimation_errors(all_phase_estimates, phases):
    phase_estimates = all_phase_estimates[-1]
    estimation_errors = [min([abs_phase_difference(phase_true, phase_est)
                              for phase_est in phase_estimates])
                         for phase_true in phases]
    return(estimation_errors)


def analyse_error_estimation(method,
                             phases, amplitudes,
                             eps, alpha, gamma,
                             final_error, cutoff,
                             rng = np.random.RandomState(42)):
    
    all_phase_estimates, costs, error_flag = multiorder_estimation(method,
                             phases, amplitudes,
                             eps, alpha, gamma,
                             final_error, cutoff, rng)
    
    est_errors = get_estimation_errors(all_phase_estimates, phases)
    
    fail = (error_flag[0]!='success')
    print(error_flag)

    return est_errors, costs[-1], fail


# ## Running script on some test data


def run_estimation_errors(
    final_errors, method, num_phases, eps, alpha, gamma, cutoff, num_repetitions, rng = np.random.RandomState(42)):
    
    est_errors_big = []
    costs_big = []
    failures_big = []
    
    for final_error in final_errors:
        print('Processing final error:', final_error)
        est_errors = []
        costs = []
        failures = []
        for rep in range(num_repetitions):

            start = datetime.datetime.now()
            print(rep, 'Started at:', start)

            phases = rng.uniform(0, 2*np.pi, num_phases)
            print(phases)
            amplitudes = np.ones(num_phases)
            amplitudes = amplitudes / np.sum(amplitudes)
            estimation_errors, cost, failure = analyse_error_estimation(
                method,
                phases, amplitudes,
                eps, alpha, gamma,
                final_error, cutoff, rng)
            est_errors.append(estimation_errors)
            costs.append(cost)
            failures.append(failure)

            end = datetime.datetime.now()
            print('Executed in:', end-start)
        
        print(f'Proportion of simulations exited before last order:{np.sum(failures)*100/num_repetitions}%')
    
        est_errors_big.append(est_errors)
        costs_big.append(costs)
        failures_big.append(failures)
    
    return(costs_big, est_errors_big, failures_big)


# ## Binning results and rejecting things far outside our confidence interval


def plot_estimation_errors(costs_big, est_errors_big, color = 'black'):
    midpoints = np.sort(np.kron(10**np.arange(
        start = np.floor(np.log10(np.min([np.min(c) for c in costs_big]))),
        stop = 1+np.ceil(np.log10(np.max([np.max(c) for c in costs_big])))
    ), [1, 3]))
    log_midpoints = np.log(midpoints)
    log_edges = 0.5 * (log_midpoints[:-1] + log_midpoints[1:])
    edges = np.exp(log_edges)
    right_edge = max(edges)
    num_edges = len(edges)
    bin_xvals = [[] for x in midpoints]
    bin_yvals = [[] for x in midpoints]
    for costs, est_errors in zip(costs_big, est_errors_big):
        for cvec, errorvec in zip(costs, est_errors):
            for c, error in zip(cvec, errorvec):
                if c > right_edge:
                    bin_xvals[-1].append(c)
                    bin_yvals[-1].append(error)
                else:
                    index = min([j for j in range(num_edges) if edges[j] > c])
                    bin_xvals[index].append(c)
                    bin_yvals[index].append(error)
    for n in range(num_edges, -1, -1):
        if len(bin_xvals[n]) == 0:
            del bin_xvals[n]
            del bin_yvals[n]
            
    binx_means = [np.mean(b) for b in bin_xvals]
    binx_err = [np.std(b) / np.sqrt(len(b)) * 2 for b in bin_xvals]
    biny_means = [np.mean(b) for b in bin_yvals]
    biny_err = [np.std(b) / np.sqrt(len(b)) * 2 for b in bin_yvals]
    
    factor = np.exp(np.polyfit(np.zeros(len(binx_means)), np.log(binx_means)+np.log(biny_means), 0))
    print('Factor:',factor)
    xvec_temp = np.linspace(min(binx_means)/1e1, max(binx_means)*1e1)
    plt.plot(xvec_temp, factor/xvec_temp, '--', color = 'k')
    
    plt.plot([x for b in bin_xvals for x in b], [y for b in bin_yvals for y in b],
             '.', markersize = 1, color = color)
    #plt.plot(binx_means, biny_means, 'r+', markersize=20, markeredgewidth=3)
    plt.plot(binx_means, biny_means, 'o', markersize=5, color = color)
    plt.errorbar(binx_means, biny_means, yerr=biny_err, xerr=binx_err, fmt='.',
                 color=color, capsize=8, capthick=3, linewidth=3)
    plt.xscale('log')
    plt.yscale('log')
    #plt.plot(midpoints, 5000/midpoints, 'b--', label=r'$y\sim 1/x$')
    plt.xlabel(r'Total quantum cost $T$')
    plt.ylabel(r'Estimator error $\delta$')
    
def plot_phase_estimates(true_phases, phase_estimates, max_order):

    min_rad = 5

    plt.polar()
    for phase in true_phases:
        plt.plot([0, phase], [0, max_order + min_rad], color = 'black')
    for j,est in enumerate(phase_estimates):
        plt.scatter(est, [5+j]*len(est), color = 'red', marker = 'x', s = 20)
    plt.gca().axes.xaxis.set_ticklabels([])
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.gca().axes.yaxis.set_ticks(range(min_rad,max_order+min_rad))

    plt.xlabel('Radius = order')
    plt.plot([], [], color = 'black', label = 'True phases')
    plt.scatter([],[], color = 'red', marker = 'x', s = 20, label = 'Estimates')
    plt.legend()