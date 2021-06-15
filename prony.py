#Prony code without fixed L
import scipy.linalg
import numpy as np


def prolong(signal):
    '''
    Take [g(0), g(1), ... g(K)] and return [g(-K), g(-K+1), ... g(K)]
    '''
    signal = np.concatenate((
        np.flip(np.conj(signal[1:]),0),
        [signal[0]],
        signal[1:]
    ))
    return(signal)

def prony(signal, num_phases, prolong_signal = False):
    
    if prolong_signal:
        long_signal = prolong(signal)
    else:
        long_signal = signal
    
    hankel0 = scipy.linalg.hankel(
        c=long_signal[:num_phases], r=long_signal[num_phases-1:-1])
    hankel1 = scipy.linalg.hankel(
        c=long_signal[1:num_phases+1], r=long_signal[num_phases:])

    shift_matrix = scipy.linalg.lstsq(hankel0.T, hankel1.T)[0]
    
    eigvals = np.linalg.eigvals(shift_matrix.T)

    generation_matrix = np.array([
        [val**k for val in eigvals]
        for k in range(len(signal))])
    amplitudes = scipy.linalg.lstsq(generation_matrix, signal)[0]

    amplitudes, eigvals = zip(
            *sorted(zip(amplitudes, eigvals),
                    key=lambda x: np.abs(x[0]), reverse=True))

    return np.array(amplitudes), np.array(eigvals)

def prony_phases(signal, cutoff, delta = None):
    '''
    Get phases estimates mod 2pi with amplitude > cutoff
    If delta is given, the estimates have to be at least delta apart.
    '''
  
    amp_estimates, eigval_estimates = prony(signal, len(signal)//2, True)

    indices = np.where(np.abs(amp_estimates) < cutoff)[0]
    if len(indices) > 0:
        cutoff_index = np.min(indices)
    else:
        cutoff_index = len(amp_estimates)
        
    amp_estimates = amp_estimates[:cutoff_index]
    eigval_estimates = eigval_estimates[:cutoff_index]

    phase_estimates = np.angle(eigval_estimates) % (2*np.pi)
    
    #if(delta is not None):
    #    for i in range(len(phase_estimates)):
            
    
    return phase_estimates

#def get_signal_requirements(num_phases, confidence, error):
#    signal_length = 2 * num_phases
#    
#    return(signal_length, num_samples)