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


def match_phases(prev_phases, prev_errors, multiplier, new_phases, new_errors):
    '''Matches a set of already-obtained signals accurate to within
    1/multiplier to a new set of signals, under the assumption that
    the multiplier found prevents aliasing.

    [description]

    Arguments:
        prev_signals {[type]} -- [description]
        multiplier {[type]} -- [description]
        new_signals {[type]} -- [description]
    '''
    winding_numbers = [(phase * multiplier) // (2 * numpy.pi)
                       for phase in prev_phases]
    residual_phases = [(phase * multiplier) % (2 * numpy.pi)
                       for phase in prev_phases]
    magnified_errors = [error * multiplier for error in prev_errors]
    matched_phases = []
    matched_errors = []
    for phase, error in zip(new_phases, new_errors):
        possible_matchings = [
            (old_phase, wn + _wn_diff(old_phase, phase), old_error)
            for old_phase, old_error, wn in zip(
                residual_phases, magnified_errors, winding_numbers)
            if abs_phase_difference(old_phase, phase) < (error + old_error) / 2
        ]
        if len(possible_matchings) == 0:
            warnings.warn('No matched phase found, could be spurious'
                          ' detection at this order.')
            continue
        if len(set([match[1] for match in possible_matchings])) > 1:
            warnings.warn('Alias detected, matching possibly ambiguous,'
                          'taking closest signal. Did you choose the correct'
                          ' multiplier?')
            print('Printing possible matchings:')
            print('New phase: {}, old_phases: {}'.format(
                phase, possible_matchings))
            print()
            possible_matchings = sorted(
                possible_matchings,
                key=lambda x: abs_phase_difference(x[0], phase))

        found_wn = possible_matchings[0][1]
        full_phase = (found_wn * 2 * numpy.pi + phase) / multiplier
        full_error = error / multiplier
        matched_phases.append(full_phase)
        matched_errors.append(full_error)
    return matched_phases, matched_errors


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


def beta_finder(phases, delta, prev_multiplier, max_beta=None):
    '''Finds the largest possible multiplier for unambiguous phase estimation

    Single-ancilla QPE requires that one estimates a signal at
    different orders separated by multipliers (i.e. g(k*B_d) for
    B_d = prod_d beta_d. Individual beta_d need to be chosen to ensure
    that the phase matching at subsequent orders is unambiguous. This
    function determines the best choice of beta_d to make for these purposes.

    Args:
        phases [list of floats] -- The phases obtained at the previous
            order to an accuracy of epsilon / prev_multiplier
        error [float] -- Width of the confidence interval on the obtained phases.
            I.e. 2 times the error bars on the phases. We assume
            that any phases within error/prev_multiplier of each
            other do not need to be resolved.
        prev_multiplier [float] -- The previous multiplier used.
    '''
    
    error = delta
    if max_beta is None:
        max_beta = numpy.pi / error - 1
        
        
    diff_bound = delta/prev_multiplier
    phase_differences = [
        abs(phase1 - phase2) for j, phase1 in enumerate(phases)
        for phase2 in phases[j:]
        if abs(phase1 - phase2) > diff_bound
    ]
    if not phase_differences:
        return max_beta
    forbidden_region_alias_numbers = [
        _next_alias_number(prev_multiplier, max_beta, error, phase_difference)
        for phase_difference in phase_differences
    ]

    # We want to put these entires in a priority queue, but we want
    # to select the greatest value of beta first, so we flip the sign.
    forbidden_region_lhs = [
        (-_alias_region_left_side(alias_number, prev_multiplier, error,
                                  phase_difference), phase_difference,
         alias_number) for phase_difference, alias_number in zip(
             phase_differences, forbidden_region_alias_numbers)
        if alias_number > 0
    ]
    
    if not  forbidden_region_lhs:
        return(max_beta)

    # forbidden_region_rhs = [
    #     (-_alias_region_right_side(
    #         alias_number, prev_multiplier, error, phase_difference),
    #      phase_difference, alias_number)
    #     for phase_difference, alias_number in zip(
    #         phase_differences, forbidden_region_alias_numbers)]

    lhs_queue = queue.PriorityQueue()
    rhs_queue = queue.PriorityQueue()

    for lhs in forbidden_region_lhs:
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
                best_beta = -next_lhs[0]
                return best_beta

            # Make queue entry and insert
            this_rhs = (-_alias_region_right_side(
                next_lhs[2] - 1, prev_multiplier, error, next_lhs[1]),
                        next_lhs[1], next_lhs[2] - 1)
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


def _next_alias_number(prev_multiplier, this_multiplier, error,
                       phase_difference):
    '''Counts the number of aliasing events that occur before the current beta value

    Args:
        prev_multiplier [float] -- Multiplier up to this point
        this_multiplier [float] -- Multiplier at this order
        error [float] -- Estimation error at each order
            (width of the confidence interval)
        phase_difference [float] -- Difference between phases
    '''
    return numpy.floor(
        (this_multiplier * prev_multiplier * abs(phase_difference) + error *
         (1 + this_multiplier)) / (2 * numpy.pi))


def _alias_region_left_side(alias_number, prev_multiplier, error,
                            phase_difference):
    '''Counts the number of aliasing events that occur before the current beta value

    Args:
        alias_number [float] -- Number of times wrapped around the circle.
        prev_multiplier [float] -- Multiplier up to this point
        error [float] -- Estimation error at each order
            (width of the confidence interval)
        phase_difference [float] -- Difference between phases
    '''
    return (2 * numpy.pi * alias_number -
            error) / (prev_multiplier * phase_difference + error)


def _alias_region_right_side(alias_number, prev_multiplier, error,
                             phase_difference):
    '''Counts the number of aliasing events that occur before the current beta value

    Args:
        alias_number [float] -- Number of times wrapped around the circle.
        prev_multiplier [float] -- Multiplier up to this point
        error [float] -- Estimation error at each order
            (width of the confidence interval)
        phase_difference [float] -- Difference between phases
    '''
    return (2 * numpy.pi * alias_number +
            error) / (prev_multiplier * phase_difference - error)
