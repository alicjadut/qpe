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
    qeep_approximate_single_eigenvalues, get_phase_values)
from sparse_qpe import(
    kappa_finder, match_phases, abs_phase_difference)

from tqdm import tqdm

rng = np.random.RandomState(42)


# # Definitions

# ## Functions to produce a phase function with the appropriate noise


def add_noise(prob, num_samples):
    res = rng.binomial(num_samples, prob) / num_samples
    return res

def get_gk(signal_length, phases, amplitudes, num_samples, multiplier):
    gk_clean = np.array([np.sum(
        np.array(amplitudes) * np.exp(1j * np.array(phases) * k * multiplier))
                     for k in range(signal_length+1)])
    if num_samples is None:
        gk_noisy = gk_clean
    else:
        pk_real_clean = 0.5 - 0.5 * np.real(gk_clean)
        pk_imag_clean = 0.5 - 0.5 * np.imag(gk_clean)
        pk_real_noisy = add_noise(pk_real_clean, num_samples)
        pk_imag_noisy = add_noise(pk_imag_clean, num_samples)

        gk_noisy = (1 - 2 * pk_real_noisy) + 1j *(1 - 2 * pk_imag_noisy)
    return gk_noisy


# ## Function to perform single-order estimation

# 3 methods to consider:
# 
# - `qeep` - find the spectral function with QEEP solver. 
# - `qeep-sparse` - find spectral function with QEEP solver. Return all bin centers that have amplitude > cutoff.
# - `pencil` - find the phases with matrix pencil method.


def estimate_phases(method, signal, cutoff, num_points):
   
    
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

def shift_value(phases):
    
    #zeta is the number between 2 consecutive phases that are furtherst away
    phases = np.sort(phases)
    phase_differences = [
        abs_phase_difference(phases[j], phases[(j+1) % len(phases)]) for j in range(len(phases))
    ]
    ix = np.argmax(phase_differences)
    zeta = (phases[(ix+1) % len(phases)]+
            phases[ix])/2
    #d_zeta is half of this largest distance
    d_zeta = np.max(phase_differences)/2
    shift_val = zeta+d_zeta/2
    return shift_val

# ## Function to perform multi-order estimation

def multiorder_estimation(method,
                             phases, amplitudes,
                             eps, eps0, alpha, gamma,
                             final_error, cutoff):
    
    max_order = np.ceil(np.log2(2*eps/final_error)).astype('int')
    
    estimates = []
    costs = []

    # Generate zeroth order phase estimates
    multiplier = 1
    
    # Calculate the signal requirements at this order and the assoc. cost
    confidence = 1-np.exp(-alpha-gamma*max_order)
    num_points, signal_length, num_samples = get_signal_requirements(confidence, eps0)
    cost = sum([num_samples * 2 * k * multiplier for k in range(signal_length + 1)])
    
    # Get the new signal and estimate aliased phases from this.
    gk_noisy = get_gk(signal_length, phases, amplitudes, num_samples, multiplier)
    phase_estimates = estimate_phases(method, gk_noisy, cutoff, num_points)
    error_estimates = [eps for phase in phase_estimates]
    
    # Add phase estimates and costs to data
    costs.append([cost for phase in phases])
    estimates.append(list(phase_estimates))
    
    # Shift the unitary
    shift_val = shift_value(phase_estimates)
    phases = phases - shift_val
    phase_estimates = phase_estimates - shift_val
    
    
    #Find the first multiplier
    try:
        multiplier = kappa_finder(phase_estimates, 2*eps0, multiplier, np.pi / (2 * eps0))
    except ValueError:
        print(r'Couldnt find good $k_1$, exiting')
        return  
    kappas = [multiplier]
    
    d=1

    while(multiplier < 2*eps/final_error and d<max_order+1):

        
        if(d>1):
            # Calculate the new best multiplier from the previous phase data.
            # If this doesn't work, fail gracefully.
            #(d = 1 is excluded, because we want to use eps0 for it)
            try:
                kappas.append(kappa_finder(phase_estimates, 2*eps, multiplier, np.pi / (2 * eps)))
            except ValueError:
                print('Couldnt find good kappa, exiting')
                break          
            multiplier = np.prod(kappas)

        # Calculate the signal requirements at this order and the assoc. cost
        confidence = 1 - np.exp(-alpha-gamma*(max_order-d))
        num_points, signal_length, num_samples = get_signal_requirements(confidence, eps)
        cost += sum([num_samples * 2 * k * multiplier for k in range(signal_length + 1)])
        
        # Get the new signal and estimate aliased phases from this.
        gk_noisy = get_gk(signal_length, phases, amplitudes, num_samples, multiplier)
        aliased_phase_estimates = estimate_phases(method, gk_noisy, cutoff, num_points)
        aliased_error_estimates = [eps for phase in aliased_phase_estimates]
        
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
            return estimates, costs

        # Match phases --- generate new estimates of phases at each order
        phase_estimates, error_estimates = match_phases(
            phase_estimates, error_estimates, multiplier,
            aliased_phase_estimates, aliased_error_estimates)
        
        # If we have completely failed, do it gracefully
        if len(phase_estimates) == 0:
            print('No phases left, exiting')
            break
        
        # Add phase estimates errors and costs to data
        costs.append([cost for phase in phases])
        estimates.append(phase_estimates+shift_val)
        
        d+=1
            
    return estimates, costs    


def get_estimation_errors(all_phase_estimates, phases):
    estimation_errors = []
    for phase_estimates in all_phase_estimates:
        estimation_errors.append([min([abs_phase_difference(phase_true, phase_est)
                                   for phase_est in phase_estimates])
                              for phase_true in phases])
    return(estimation_errors)

def get_estimation_failures(all_phase_estimates, phases, eps, max_order):
    # This is my current definition of 'failure' --- if we either see
    # a spurious phase or don't pick up a given eigenvalue.
    # Haven't looked into this too much though.
        failure_booleans = []
        for phase_estimates in all_phase_estimates:
            spurious_phases = [phase_est for phase_est in phase_estimates
                       if min([abs_phase_difference(phase_est, phase_true)
                               for phase_true in phases]) > eps]
            errors = get_estimation_errors([phase_estimates], phases)
            if len(spurious_phases) == 0 and np.max(errors) < eps:
                failure_booleans.append([False for phase in phases])
            else:
                failure_booleans.append([True for phase in phases])
        if(len(failure_booleans) < max_order):
            failure_booleans.append(True)
        return(failure_booleans)

def analyse_error_estimation(method,
                             phases, amplitudes,
                             eps, alpha, gamma,
                             max_order, cutoff):
    
    all_phase_estimates, costs = multiorder_estimation(method,
                             phases, amplitudes,
                             eps, alpha, gamma,
                             max_order, cutoff)
    
    errors = get_estimation_errors(all_phase_estimates, phases)
    failure_booleans = get_estimation_failures(all_phase_estimates, phases, eps, max_order)

    return errors, failure_booleans, costs


# ## Running script on some test data


def run_estimation_errors(method, num_phases, max_order, eps, alpha, gamma, cutoff, num_repetitions):
    est_errors_big = []
    failure_booleans_big = []
    costs_big = []

    #rng = np.random.RandomState(42)
    
    for rep in range(num_repetitions):
        
        start = datetime.datetime.now()
        print(rep, 'Started at:', start)
        
        phases = rng.uniform(0, 2*np.pi, num_phases)
        amplitudes = np.ones(num_phases)
        amplitudes = amplitudes / np.sum(amplitudes)
        estimation_errors, failure_booleans, costs = analyse_error_estimation(
            method,
            phases, amplitudes,
            eps, alpha, gamma,
            max_order, cutoff)
        est_errors_big.append(estimation_errors)
        failure_booleans_big.append(failure_booleans)
        costs_big.append(costs)
        
        end = datetime.datetime.now()
        print('Executed in:', end-start)
        
    print(f'Proportion of simulations exited before last order:{np.sum([len(f)<max_order+1 for f in failure_booleans_big])*100/num_repetitions}%')
        
    return(costs_big, est_errors_big, failure_booleans_big)


# ## Binning results and rejecting things far outside our confidence interval


def plot_estimation_errors(costs_big, est_errors_big):
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
    num_rejections = 0
    for costs, est_errors in zip(costs_big, est_errors_big):
        for cvec, errorvec in zip(costs, est_errors):
            for c, error in zip(cvec, errorvec):
                if error > 0.5:
                    num_rejections += 1
                    continue
                if c > right_edge:
                    bin_xvals[-1].append(c)
                    bin_yvals[-1].append(error)
                else:
                    index = min([j for j in range(num_edges) if edges[j] > c])
                    bin_xvals[index].append(c)
                    bin_yvals[index].append(error)
    print('Proportion of rejected samples: {}'.format(num_rejections / sum([len(b) for b in bin_xvals])))
    for n in range(num_edges, -1, -1):
        if len(bin_xvals[n]) == 0:
            del bin_xvals[n]
            del bin_yvals[n]
            
    binx_means = [np.mean(b) for b in bin_xvals]
    binx_err = [np.std(b) / np.sqrt(len(b)) * 2 for b in bin_xvals]
    biny_means = [np.median(b) for b in bin_yvals]
    biny_max = [np.percentile(b, 75) for b in bin_yvals]
    biny_min  = [np.percentile(b, 25) for b in bin_yvals]
    
    plt.plot([x for b in bin_xvals for x in b], [y for b in bin_yvals for y in b], 'k.', markersize=1, label='Data points')
    #plt.plot(binx_means, biny_means, 'r+', markersize=20, markeredgewidth=3)
    plt.plot(binx_means, biny_means, 'ro', markersize=5, label = 'Binned means')
    plt.errorbar(binx_means, biny_means, yerr=(biny_min, biny_max), xerr=binx_err, fmt='r.',
                 color='red', capsize=8, capthick=3, linewidth=3)
    plt.xscale('log')
    plt.yscale('log')
    #plt.plot(midpoints, 5000/midpoints, 'b--', label=r'$y\sim 1/x$')
    plt.legend()#fontsize=22)
    plt.xlabel('Total number of unitary applications')
    plt.ylabel('Estimator error')
    
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