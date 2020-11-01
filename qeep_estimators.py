#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""An implementation of Rolando Somma's method for estimating phases from
single ancilla QPE, taken from ArXiv:1907.11748
"""

import numpy
from scipy import integrate, optimize

# Calculated with scipy.integrate.quad with 1e-14 tolerance
BUMP_NORMALIZATION = 2.25228362104358


def get_signal_requirements(confidence, error):
    '''Calculates requirements on the input signal for the qeep solver

    Returns the number of points g(k) must be sampled at,
    and the number of samples required at each point, in order
    to obtain an approximation to the spectral function on
    [0, 2*pi] with a specified confidence that
    the 1-norm error is less than a fixed amount

    Arguments:
        confidence [float] -- confidence of result being within 1-norm
        error [float] -- bound on 1-norm and point spacing of spectral
            function approximation

    Returns:
        num_points [int] -- number of points in spectral function
            approximation
        num_points_samples [int] -- number of points that the phase function
            g(k) needs to be sampled at
        num_samples [int] -- number of samples of the phase function needed
            at each point (equiv. 1/std_dev^2 where std_dev is the standard
            deviation on each g(k))
    '''
    num_points = int(numpy.ceil((2 * numpy.pi) / error))
    # Prefactors from ArXiv:1907.11748
    num_points_sampled = int(
        numpy.ceil(0.1 * numpy.log(num_points)**2 * num_points))
    num_samples = int(
        numpy.ceil(
            numpy.abs(
                numpy.log(1 - confidence**(1 / num_points_sampled)) / error**4)))

    return num_points, num_points_sampled, num_samples


def get_phase_values(spectral_function):
    '''Returns the x values associated to a spectral function

    (Separated from qeep_sparse_solve to make inspection easier.)

    Arguments:
        spectral_function [array-like] -- the spectral function
    '''
    num_points = len(spectral_function)
    phase_values = numpy.linspace(0, 2 * numpy.pi * (1 - 1 / num_points),
                                  num_points)
    return phase_values


def qeep_sparse_solve(spectral_function, cutoff):
    '''Solver for the sparse quantum eigenvalue estimation problem

    Solves the qeep, but returns a set of phases with corresponding
    signals above some cutoff amplitude A instead of the spectral
    function approximation.

    Arguments:
        spectral_function [numpy array] -- the spectral function obtained from
            qeep_solve
        cutoff [float] -- minimum amplitude to count as a signal
            in the spectral function
    '''
    phase_values = get_phase_values(spectral_function)
    phase_estimates = [
        v for v, p in zip(phase_values, spectral_function) if abs(p) > cutoff
    ]
    return phase_estimates


def qeep_approximate_single_eigenvalues(spectral_function, cutoff):
    '''Calculates approximate QEEP eigenvalues, assuming separation

    Attempts to find single eigenvalues within bins of the QEEP,
    assuming that all eigenvalues of the problem are well separated.

    Arguments:
        spectral_function [array-like] -- the output of qeep_solve
        cutoff [float] -- cutoff value at which point a signal is declared
            not to exist
    '''
    num_points = len(spectral_function)
    spectral_function = numpy.abs(spectral_function)
    epsilon = 2 * numpy.pi / num_points
    eigenvalues = []
    for index, amplitude in enumerate(spectral_function):

        # Pass if less than cutoff
        if amplitude < cutoff:
            continue
        amp_left = spectral_function[(index - 1) % num_points]
        amp_right = spectral_function[(index + 1) % num_points]
        # To prevent duplicate frequencies, pass if not the maximum eigenvalue
        if max(amp_left, amp_right) > amplitude:
            continue
        if amp_left > amp_right:
            other_amp = amp_left
            side = -1
        else:
            other_amp = amp_right
            side = 1
        rescaled_amp = amplitude / (amplitude + other_amp)
        left_bound = min(index * epsilon, (index + side) * epsilon)
        right_bound = max(index * epsilon, (index + side) * epsilon)
        eigenvalue = optimize.minimize_scalar(
            lambda x: abs(weight_function(x, index, epsilon) - rescaled_amp),
            bounds=(left_bound, right_bound),
            method='Bounded')
        eigenvalues.append(eigenvalue['x'])
    return eigenvalues


def qeep_solve(signal, num_points):
    '''solver for the quantum eigenvalue estimation problem

    Arguments:
        signal [list] -- the input signal for time-series analysis
        num_points [int] -- frequency discretization of output

    Returns:
        list -- approximation to the spectral function on [-pi, pi]
            with discretization num_points.
    '''
    epsilon = 2 * numpy.pi / num_points
    return numpy.array([
        _weight_ft(j + 0.5, 0, epsilon) * signal[0] + numpy.sum([
            _weight_ft(j + 0.5, k, epsilon) * numpy.conj(signal[k]) +
            _weight_ft(j + 0.5, -k, epsilon) * signal[k]
            for k in range(1, len(signal))
        ]) for j in range(num_points // 2, -num_points // 2, -1)
    ])


def _bump_function(x):
    if x <= -1 or x >= 1:
        return 0
    return BUMP_NORMALIZATION * numpy.exp(-1 / (1 - x**2))


def _bump_fourier_transform(k):
    '''Takes the fourier transform of the bump function. As this
    is an even function we only need the real part. We further split
    the interval of integration into its different nodes, as we know
    these in advance, to improve convergence.
    '''
    nodes = [(2 * n + 1) * numpy.pi / (2 * k)
             for n in range(int(numpy.ceil(-k / numpy.pi - 1 / 2)),
                            int(numpy.floor(k / numpy.pi - 1 / 2)) + 1)]
    nodes = [-1] + nodes + [1]

    res = numpy.sum([
        integrate.quad(lambda x: _bump_function(x) * numpy.cos(k * x),
                       left_limit, right_limit)[0] / numpy.sqrt(2 * numpy.pi)
        for left_limit, right_limit in zip(nodes[:-1], nodes[1:])
    ])
    return res


def weight_function(x, k, epsilon):
    '''Generates the weight function used in ArXiv:1907.11748

    Arguments:
        x [float] -- Point in real space for evaluation
        k [int] -- Function index
        epsilon [float] -- Function width

    Returns:
        [float] -- Function evaluation
    '''
    left_limit = (k - 0.5) * epsilon
    right_limit = (k + 0.5) * epsilon
    res = integrate.quad(lambda xp: _bump_function(2 * (xp - x) / epsilon),
                         left_limit, right_limit)[0] * 2 / epsilon
    return res


def _weight_ft(j, k, epsilon):
    # Implements Equation 16 of ArXiv:1907.11748
    weight_fn = (_bump_fourier_transform(k * epsilon / 2) *
                 numpy.exp(-1j * (-numpy.pi + j * epsilon) * k) *
                 numpy.sqrt(2 / numpy.pi))
    if k == 0:
        weight_fn *= epsilon / 2
    else:
        weight_fn *= numpy.sin(k * epsilon / 2) / k
    return weight_fn
