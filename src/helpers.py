import itertools
import math
import warnings

import numpy as np
import scipy.signal
from scipy.special import logsumexp


def check_rows_sum_to_one(matrix):
    row_sums = np.sum(matrix[:-1], axis=1)
    return np.allclose(row_sums, 1)

def generate_full_mch(parameters, number_of_states):
    matrix = np.zeros((number_of_states, number_of_states))
    counter = 0
    for row in range(number_of_states - 1):
        sum = np.exp(0)
        matrix_row = np.zeros(number_of_states)
        matrix_row[0] = np.exp(0)
        for column in range(1, number_of_states):
            parameter = np.exp(parameters[counter])
            sum += parameter
            matrix_row[column] = parameter
            counter += 1
        matrix[row] = matrix_row / sum
    if (check_rows_sum_to_one(matrix)) :
        return matrix
    else:
        print("smthing went wrong")

def generate_matrix(parameters, architecture, number_of_states, k=None, l=None):
    if architecture == "chain":
        array = np.zeros((number_of_states,number_of_states))
        for i in range(number_of_states - 1):
            x = parameters[i]
            prob_of_self_looping = np.exp(x) / (np.exp(x) + np.exp(0))
            array[i][i] = prob_of_self_looping # set the diagonal whis is slucka to param[i]
            array[i][i + 1] = np.exp(0) / (np.exp(x) + np.exp(0))  # set the transition from i to i+1
        #array[number_of_states][number_of_states] = 0 # from escape state we aint gonna go anywhere
        matrix = array
    elif architecture == "escape_chain":
        matrix = np.zeros((number_of_states,number_of_states))
        for i in range(number_of_states - 1):
            x = parameters[i]
            prob_of_self_looping = np.exp(x) / (np.exp(x) + np.exp(0))
            matrix[i][number_of_states - 1] = prob_of_self_looping
            matrix[i][i + 1] = np.exp(0) / (np.exp(x) + np.exp(0))
    elif architecture == "combined":
        matrix = np.zeros((number_of_states,number_of_states))
        for i in range(number_of_states - 1):
            if i != number_of_states - 2:
                transition_prob_parameter = parameters[2*i]
                escaping_prob_parameter = parameters[2*i + 1]
                self_looping_parameter = 0
                sum = np.exp(transition_prob_parameter) + np.exp(escaping_prob_parameter) + np.exp(self_looping_parameter)
                prob_self_looping = np.exp(self_looping_parameter) / sum
                prob_trasnition = np.exp(transition_prob_parameter) / sum
                prob_escaping = np.exp(escaping_prob_parameter) / sum
                matrix[i][i] = prob_self_looping
                matrix[i][number_of_states - 1] = prob_escaping
                matrix[i][i + 1] = prob_trasnition
            else:
                transition_prob_parameter = parameters[2*i]
                self_looping_parameter = 0
                sum = np.exp(self_looping_parameter) + np.exp(transition_prob_parameter)
                prob_self_looping = np.exp(self_looping_parameter) / sum
                prob_of_transitioning = np.exp(transition_prob_parameter) / sum
                matrix[i][i] = prob_self_looping
                matrix[i][i + 1] = prob_of_transitioning
    elif architecture == "full":
        matrix = generate_full_mch(parameters=parameters, number_of_states=number_of_states)
    elif architecture == "cyclic":
        escape_state = number_of_states - 1
        matrix = np.zeros((number_of_states,number_of_states))
        for i in range(1, number_of_states - 2):
            x = parameters[i]
            prob_of_self_looping = np.exp(x) / (np.exp(x) + np.exp(0))
            matrix[i][i] = prob_of_self_looping # set the diagonal whis is slucka to param[i]
            matrix[i][i + 1] = np.exp(0) / (np.exp(x) + np.exp(0))
        #do first and last state separately: davaj dost pozor ako tam mas upratane tie parametre
        i = number_of_states - 3
        # transition_prob_parameter = parameters[0]
        # escaping_prob_parameter = parameters[i + 1]
        # self_looping_parameter = 0
        transition_prob_parameter = parameters[0]
        escaping_prob_parameter = 0
        #sum = np.exp(transition_prob_parameter) + np.exp(escaping_prob_parameter) + np.exp(self_looping_parameter)
        sum = np.exp(transition_prob_parameter) + np.exp(escaping_prob_parameter)
        #prob_self_looping = np.exp(self_looping_parameter) / sum
        prob_trasnition = np.exp(transition_prob_parameter) / sum
        prob_escaping = np.exp(escaping_prob_parameter) / sum
        #matrix[0][0] = prob_self_looping
        matrix[0][1] = prob_trasnition
        matrix[0][escape_state] = prob_escaping
        # last state
        last_transient_st = number_of_states - 2
        #x = parameters[i + 2]
        x = parameters[i+1]
        prob_of_self_looping = np.exp(x) / (np.exp(x) + np.exp(0))
        matrix[last_transient_st][last_transient_st] = prob_of_self_looping # set the diagonal whis is slucka to param[i]
        matrix[last_transient_st][0] = np.exp(0) / (np.exp(x) + np.exp(0))
    elif architecture == "k-jumps":
       matrix = np.zeros((number_of_states,number_of_states))
       param_counter = 0
       for i in range(number_of_states - 2):
        if i % k != 0 or i == 0 or i < l :
            transition_prob_parameter = parameters[param_counter]
            escaping_prob_parameter = parameters[param_counter + 1]
            self_looping_parameter = 0
            sum = np.exp(transition_prob_parameter) + np.exp(escaping_prob_parameter) + np.exp(self_looping_parameter)
            prob_self_looping = np.exp(self_looping_parameter) / sum
            prob_trasnition = np.exp(transition_prob_parameter) / sum
            prob_escaping = np.exp(escaping_prob_parameter) / sum
            matrix[i][i] = prob_self_looping
            matrix[i][number_of_states - 1] = prob_escaping
            matrix[i][i + 1] = prob_trasnition
            param_counter += 2
        else:
            transition_prob_parameter = parameters[param_counter]
            escaping_prob_parameter = parameters[param_counter + 1]
            self_looping_parameter = 0
            back_loop_param = parameters[param_counter + 2]
            sum = np.exp(transition_prob_parameter) + np.exp(escaping_prob_parameter) + np.exp(self_looping_parameter) + np.exp(back_loop_param)
            prob_self_looping = np.exp(self_looping_parameter) / sum
            prob_trasnition = np.exp(transition_prob_parameter) / sum
            prob_escaping = np.exp(escaping_prob_parameter) / sum
            prob_back_loop = np.exp(back_loop_param) / sum
            matrix[i][i] = prob_self_looping
            matrix[i][number_of_states - 1] = prob_escaping
            matrix[i][i + 1] = prob_trasnition
            matrix[i][i - l] = prob_back_loop
            param_counter += 3
       if i % k != 0 or i == 0 or i < l :
            transition_prob_parameter = parameters[param_counter]
            self_looping_parameter = 0
            sum = np.exp(transition_prob_parameter) + np.exp(self_looping_parameter)
            prob_self_looping = np.exp(self_looping_parameter) / sum
            prob_trasnition = np.exp(transition_prob_parameter) / sum
            matrix[number_of_states - 2][number_of_states - 2] = prob_self_looping
            matrix[number_of_states - 2][number_of_states - 1] = prob_trasnition
       else:
            transition_prob_parameter = parameters[param_counter]
            self_looping_parameter = 0
            back_loop_param = parameters[param_counter + 1]
            sum = np.exp(transition_prob_parameter)  + np.exp(self_looping_parameter) + np.exp(back_loop_param)
            prob_self_looping = np.exp(self_looping_parameter) / sum
            prob_trasnition = np.exp(transition_prob_parameter) / sum
            prob_back_loop = np.exp(back_loop_param) / sum
            matrix[number_of_states - 2][number_of_states - 2] = prob_self_looping
            matrix[number_of_states - 2][number_of_states - 1] = prob_trasnition
            matrix[number_of_states - 2][number_of_states - 2 - l] = prob_back_loop

       
    else:
        raise ValueError("Invalid architecture")

    return matrix


def load_trained_model(dir_path):
    # Read the text file
    with open(f'{dir_path}', 'r') as file:
        content = file.readlines()

    # Initialize variables to store parameters
    architecture = None
    number_of_states = None
    parameters = []

    # Iterate through each line of the content
    for idx, line in enumerate(content):
        # Split the line by ':'
        if ':' in line:
            key, value = line.strip().split(':')
            # Remove whitespace from key and value
            key = key.strip()
            value = value.strip()

            # Check the key and assign values accordingly
            if key == 'Architecture':
                architecture = value
            elif key == 'Number of States':
                number_of_states = int(value)
            elif key == 'Parameters':
                # Extract parameters
                for param_line in content[idx+1:]:
                    param = param_line.strip()
                    if param:  # Check if param is not an empty string
                        parameters.append(float(param))
                    else:
                        break  # Stop if an empty line is encountered
    matrix = generate_matrix(parameters, architecture, number_of_states)
    return matrix


def load_intervals(file, is_closed=False):
    result = []
    for lnum, line in enumerate(file):
        if len(line.strip()) == 0:
            # skip empty lines
            continue
        elements = line.strip().split("\t")
        if len(elements) != 3:
            raise ValueError(f"Incorrect number of columns! Line #{lnum}: {line}")
        chr_name, b, e = elements
        b = int(b)
        e = int(e)

        if is_closed:
            b -= 1

        if b < 0 or e < 0:
            raise ValueError(f"Coordinates should be non-negative! Line #{lnum}: {line}")
        if not (b < e):
            raise ValueError(f"Begin should be less that end! Line #{lnum}: {line}")
        result.append((chr_name, b, e))
    return sorted(result)


def load_chr_sizes(file):
    result = []
    for lnum, line in enumerate(file):
        if len(line.strip()) == 0:
            # skip empty lines
            continue
        elements = line.strip().split("\t")
        if len(elements) != 2:
            raise ValueError(f"Incorrect number of columns! Line #{lnum}: {line}")
        chr_name, length = elements
        length = int(length)
        if length <= 0:
            raise ValueError(f"Length should be positive! Line #{lnum}: {line}")
        result.append((chr_name, length))
    return result


def count_overlaps(r, q):
    chr_names = sorted(set(s[0] for s in itertools.chain(r, q)))
    r_sorted = sorted(r)
    q_sorted = sorted(q)
    r_next, q_next = 0, 0

    total_overlap_count = 0
    for chr_name in chr_names:
        while r_next < len(r_sorted) and r_sorted[r_next][0] < chr_name:
            r_next += 1
        while q_next < len(q_sorted) and q_sorted[q_next][0] < chr_name:
            q_next += 1
        r_sub = []
        while r_next < len(r_sorted) and r_sorted[r_next][0] == chr_name:
            r_sub.append(r_sorted[r_next][1:])
            r_next += 1
        q_sub = []
        while q_next < len(q_sorted) and q_sorted[q_next][0] == chr_name:
            q_sub.append(q_sorted[q_next][1:])
            q_next += 1
        if len(r_sub) == 0 or len(q_sub) == 0:
            continue
        overlap_count = count_overlaps_single_chromosome(r_sub, q_sub)
        total_overlap_count += overlap_count

    return total_overlap_count


def count_overlaps_single_chromosome(r, q):
    """Assumes that the input arrays are sorted."""
    ends = []
    for b, e in r:
        ends.append((b, 0, 0))
        ends.append((e, 0, 1))
    for b, e in q:
        ends.append((b, 1, 0))
        ends.append((e, 1, 1))
    ends.append((math.inf, 1, 0))  # to count down the last possible overlap

    count = 0
    is_ref_interval_open = False
    is_query_interval_open = False
    is_current_ref_interval_counted = False
    last_pos = -1
    for pos, t, e in sorted(ends):
        if last_pos < pos:
            last_pos = pos
            if is_ref_interval_open \
                    and is_query_interval_open \
                    and not is_current_ref_interval_counted:
                count += 1
                is_current_ref_interval_counted = True
        # a new reference interval starts
        if t == 0 and e == 0:
            is_current_ref_interval_counted = False
            is_ref_interval_open = True
        # reference interval ends
        if t == 0 and e == 1:
            is_ref_interval_open = False
        # query interval starts
        if t == 1 and e == 0:
            is_query_interval_open = True
        # query interval ends
        if t == 1 and e == 1:
            is_query_interval_open = False
    return count


def joint_pvalue(probs_by_level, overlap_count):
    """Calculates joint p-value for a given `overlap_count`.
`pvalues_by_level` should contain log-values."""
    if overlap_count < 0:
        return 1
    logprobs = joint_logprobs(probs_by_level)
    if overlap_count >= len(logprobs):
        return 0
    result = np.exp(logsumexp(logprobs[overlap_count:]))
    return result


def joint_logprobs(probs_by_level):
    if len(probs_by_level) == 0:
        raise ValueError(f"p-values should have at least one level!")

    # check for empty levels
    if any(len(level) == 0 for level in probs_by_level):
        raise ValueError(f"Layers should be non-empty!")

    # if only one level, no need to do complex computations
    if len(probs_by_level) == 1:
        return probs_by_level[0]

    # now we have to do full DP
    # we want to preserve the memory, so instead of the whole DP table
    # we only store two levels
    max_k = sum(len(level) - 1 for level in probs_by_level)

    prev_row = np.array([-np.inf for _ in range(max_k+1)], dtype=np.longdouble)
    for pos, value in enumerate(probs_by_level[0]):
        prev_row[pos] = value

    current_row = np.array([-np.inf for _ in range(max_k+1)], dtype=np.longdouble)
    accum = []
    for level in probs_by_level[1:]:
        for k in range(max_k+1):
            for j in range(min(k+1, len(level))):
                accum.append(level[j] + prev_row[k-j])
            current_row[k] = logsumexp(accum)
            accum.clear()
        prev_row = current_row.copy()  # sic! without a copy it would just pass the reference
    return current_row


def select_intervals_by_chr_name(intervals, chr_name):
    result = []
    for name, b, e in intervals:
        if name == chr_name:
            result.append((b, e))
    return result


def merge_nondisjoint_intervals(intervals):
    intervals = filter(lambda interval: interval[1] < interval[2], intervals)
    intervals = sorted(intervals)
    if len(intervals) < 2:
        return intervals
    result = []
    c, b, e = intervals[0]
    for c2, b2, e2 in intervals[1:]:
        if c2 != c:
            result.append((c, b, e))
            c, b, e = c2, b2, e2
            continue
        if e < b2:
            result.append((c, b, e))
            c, b, e = c2, b2, e2
            continue
        e = max(e, e2)
    result.append((c, b, e))
    return result


def filter_intervals_by_chr_name(intervals, chr_names):
    result = [(c, b, e) for c, b, e in intervals if c in chr_names]
    return result


def filter_empty_intervals(intervals):
    results = [(c, b, e) for c, b, e in intervals if b < e]
    return results
