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
"""Sparse QPE estimators"""

import warnings
import queue
import numpy


def match_phases(prev_phases, multiplier, new_phases):
    '''
    Simple minimal distance matching
    '''
    matched_phases = []
    for new_phase in new_phases:
        phase_ix = numpy.argmin([abs_phase_difference(prev_phase*multiplier, new_phase) for prev_phase in prev_phases])
        prev_phase = prev_phases[phase_ix]
        winding_number = numpy.argmin([
            abs_phase_difference(prev_phase, (new_phase+2*numpy.pi*n)/multiplier)
            for n in numpy.arange(numpy.floor(multiplier)+1)])
        matched_phases.append((new_phase+2*numpy.pi*winding_number)/multiplier)
        
    return matched_phases


def abs_phase_difference(angle1, angle2):
    '''Calculates the difference between two angles, including checking if
    they wrap around the circle.

    Args:
        angle1, angle2 (floats): the two angles to take the difference of

    Returns:
        angle1 - angle2: the difference **taken around the circle**
    '''
    diff = (angle1 - angle2) % (2 * numpy.pi)
    if diff > numpy.pi:
        return 2 * numpy.pi - diff
    return diff

#TO DO: after Appendix C is fixed this needs to be checked
def kappa_finder(phases, error, prev_multiplier, max_kappa=None):
    '''Finds the largest possible multiplier for unambiguous phase estimation

    Single-ancilla QPE requires that one estimates a signal at
    different orders separated by multipliers (i.e. g(k*k_d) for
    k_d = prod_d kappa_d. Individual kappa_d need to be chosen to ensure
    that the phase matching at subsequent orders is unambiguous. This
    function determines the best choice of kappa_d to make for these purposes.

    Args:
        phases [list of floats] -- The phases obtained at the previous
            order to an accuracy of error / prev_multiplier
        error [float] -- error bars on individual phases (i.e. half the
            size of the confidence interval) (=2\epsilon)
        prev_multiplier [float] -- The previous multiplier used.
    '''
    if max_kappa is None:
        max_kappa = numpy.pi / error

    phase_differences = [
        abs(phase1 - phase2) for j, phase1 in enumerate(phases)
        for phase2 in phases[j + 1:]
    ]

    if not phase_differences:
        return max_kappa

    forbidden_region_alias_numbers = [
        _alias_number_left_side(prev_multiplier, max_kappa, error, phase_difference)
        for phase_difference in phase_differences
    ]

    # We want to put these entires in a priority queue, but we want
    # to select the greatest value of kappa first, so we flip the sign.
    forbidden_region_lhs = [
        (-_alias_region_left_side(alias_number, prev_multiplier, error,
                                  phase_difference), phase_difference,
         alias_number) for phase_difference, alias_number in zip(
             phase_differences, forbidden_region_alias_numbers)
    ]

    lhs_queue = queue.PriorityQueue()
    rhs_queue = queue.PriorityQueue()

    for lhs in forbidden_region_lhs:
        if _alias_region_unnecessary(lhs[1], prev_multiplier, lhs[0], error):
            continue
        lhs_queue.put(lhs)
    next_lhs = lhs_queue.get()
    next_rhs = None
    while True:

        if -next_lhs[0] <= 1:
            raise ValueError(
                'No available multiplier without '
                'the possibility of matching. Please provide a '
                'better estimate at the current multiplier before '
                'continuing.')
        if -next_lhs[0] < 2:
            warnings.warn('New multiplier is between 1 and 2, this '
                          'may be inefficient.')

        # If the next side we find is the-left hand side of a region,
        # find the right-hand side of the next region and insert
        if next_rhs is None or next_lhs[0] < next_rhs[0]:
            # If the lhs queue is empty, we are done
            if lhs_queue.empty():
                if next_rhs is None:
                    best_kappa = -next_lhs[0]
                else:
                    best_kappa = -(next_lhs[0] + next_rhs[0]) / 2
                return best_kappa

            # Get the RHS of the next aliasing region
            this_rhs_kappa = _alias_region_right_side(
                next_lhs[2] - 1, prev_multiplier, error, next_lhs[1])

            # If the next region doesn't cause an aliasing
            # event (i.e. Eq.55 in ArXiv:XXXXX is not satisfied),
            # discount it.
            if _alias_region_unnecessary(next_lhs[1], prev_multiplier,
                                         this_rhs_kappa, error):
                next_lhs = lhs_queue.get()
                continue

            # Make queue entry and insert
            this_rhs = (-this_rhs_kappa, next_lhs[1], next_lhs[2] - 1)
            if next_rhs is None:
                next_rhs = this_rhs
            elif this_rhs[0] < next_rhs[0]:
                rhs_queue.put(next_rhs)
                next_rhs = this_rhs
            else:
                rhs_queue.put(this_rhs)

            # Get next lhs
            next_lhs = lhs_queue.get()
            continue
        # Otherwise we have hit the right-hand-side of a region,
        # find the left hand side of the same region and insert
        this_lhs = (-_alias_region_left_side(next_rhs[2], prev_multiplier,
                                             error, next_rhs[1]), next_rhs[1],
                    next_rhs[2])
        # next_lhs is never None or we would have already exited
        if this_lhs[0] < next_lhs[0]:
            lhs_queue.put(next_lhs)
            next_lhs = this_lhs
        else:
            lhs_queue.put(this_lhs)
        # Get the next rhs if it exists
        if rhs_queue.empty():
            next_rhs = None
        else:
            next_rhs = rhs_queue.get()


def _wn_diff(old_phase, new_phase):
    if abs(old_phase - new_phase) > numpy.pi:
        if old_phase > new_phase:
            return 1
        return -1
    return 0


def _alias_region_unnecessary(phase_difference, prev_multiplier, kappa, error):
    '''Checks whether a point in an alias region is unnecessary.

    Checks whether two phase estimates that could in principle both be
    matched to a third would not give two different results. This condition
    is given in ArXiv:XXXXX, Eq.55.

    Arguments:
        phase_difference [float] -- difference between two phases
        prev_multiplier [float] -- the multiplier at the last point
        kappa [float] -- multiplier to be tested
        error [float] -- error in estimation
    '''
    if phase_difference < (
            numpy.pi - error * (1 + kappa)) / (kappa * prev_multiplier):
        return True
    return False


def _alias_number_left_side(prev_multiplier, this_multiplier, error,
                            phase_difference):
    '''Finds the alias number for the first LHS before a given kappa

    Args:
        prev_multiplier [float] -- Multiplier up to this point
        this_multiplier [float] -- Multiplier at this order
        error [float] -- Estimation error at each order
            (0.5 * width of the confidence interval)
        phase_difference [float] -- Difference between phases
    '''
    return numpy.floor((this_multiplier * (
        prev_multiplier * phase_difference + 2 * error) + 2 * error) / (
        2 * numpy.pi))


def _alias_number_right_side(prev_multiplier, this_multiplier, error,
                             phase_difference):
    '''Finds the alias number for the first RHS before a given kappa

    Args:
        prev_multiplier [float] -- Multiplier up to this point
        this_multiplier [float] -- Multiplier at this order
        error [float] -- Estimation error at each order
            (0.5 * width of the confidence interval)
        phase_difference [float] -- Difference between phases
    '''
    return numpy.floor((this_multiplier * (
        prev_multiplier * phase_difference - 2 * error) - 2 * error) / (
        2 * numpy.pi))


def _alias_region_left_side(alias_number, prev_multiplier, error,
                            phase_difference):
    '''Counts the number of aliasing events that occur before the current kappa value

    Args:
        alias_number [float] -- Number of times wrapped around the circle.
        prev_multiplier [float] -- Multiplier up to this point
        error [float] -- Estimation error at each order
            (half-width of the confidence interval)
        phase_difference [float] -- Difference between phases
    '''
    return (2 * numpy.pi * alias_number - 2 * error) / (
        prev_multiplier * phase_difference + 2 * error)


def _alias_region_right_side(alias_number, prev_multiplier, error,
                             phase_difference):
    '''Counts the number of aliasing events that occur before the current kappa value

    Args:
        alias_number [float] -- Number of times wrapped around the circle.
        prev_multiplier [float] -- Multiplier up to this point
        error [float] -- Estimation error at each order
            (width of the confidence interval)
        phase_difference [float] -- Difference between phases
    '''
    return (2 * numpy.pi * alias_number + 2 * error) / (
        prev_multiplier * phase_difference - 2 * error)
