import matplotlib.pyplot as plt
from numpy import logaddexp
import numpy as np
from scipy.special import logsumexp

def log_matrix_multipl(A, B):
    size = A.shape[0]
    result = np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            elem = A[i,:] + B[:,j]
            result[i][j] = logsumexp(elem)
    return result


def vector_matrix_log(vector, matrix):
    size = len(vector)
    result = []
    for i in range(size):
        elem_result = vector[0] + matrix[0][i]
        for j in range(1,size):
            elem = vector[j] + matrix[j][i]
            elem_result = logaddexp(elem_result, elem)
        result.append(elem_result)
    return result

# Generate a larger vector with smaller numbers
vector = np.random.uniform(0.01, 0.1, size=10)

# Generate a larger matrix with smaller numbers
A = np.random.uniform(0.01, 0.1, size=(10, 10))
B = np.random.uniform(0.01, 0.1, size=(10, 10))

# Convert to log space
A_log = np.log(A)
B_log = np.log(B)

# Standard matrix multiplication
dot_product = A.dot(B)

# Matrix multiplication in log space
log_result = log_matrix_multipl(A_log, B_log)

# Convert back to normal space from log space
exp_log_result = np.exp(log_result)

# Assertions for comparison
assert np.allclose(dot_product, exp_log_result), "Matrix multiplication in normal space and log space results do not match"
assert np.allclose(np.log(dot_product), log_result), "Log of dot product and log space matrix multiplication results do not match"

# # Calculate using the custom function
# result = vector_matrix_log(np.log(vector), np.log(matrix))
# print("Result from custom function:")
# print(np.exp(result))


# Function to read the data from the file
def read_data(filename):
    k_values = []
    p_values = []
    
    with open(filename, 'r') as file:
        next(file)  # Skip the header line
        for line in file:
            parts = line.split()
            k_values.append(int(parts[0]))
            p_values.append(float(parts[1]))
    
    return k_values, p_values

# Function to compute the original values from suffix sums
def compute_original_values(p_values):
    original_values = []
    for i in range(len(p_values) - 1):
        original_values.append(p_values[i] - p_values[i + 1])
    original_values.append(p_values[-1])  # The last value remains as it is
    return original_values

# Function to plot the data
def plot_data(k_values, p_values):
    plt.plot(k_values, p_values, marker='o', linestyle='-', color='b')
    plt.xlabel('k')
    plt.ylabel('pvalue')
    plt.title('Plot of k vs pvalue')
    plt.grid(True)
    plt.show()

plt.figure(figsize=(10 * 0.6, 8 * 0.5))
# Main part of the script
filename = '../resources/example_github/pvalues_Gains.txt'
k_values, p_values = read_data(filename)
p_values = compute_original_values(p_values)
filename = '../resources/example_github/askar_Gains.txt'
k_values_1, p_values_1 = read_data(filename)
p_values_1 = compute_original_values(p_values_1)
filename = '../resources/example_github/orig2.txt'
k_values_2, p_values_2 = read_data(filename)
p_values_2 = compute_original_values(p_values_2)
plt.plot(k_values, p_values, label="extended MCDP")
plt.plot(k_values_1, p_values_1, label="MCDP", linestyle="--" ,color="black")
plt.plot(k_values_2, p_values_2, label="direct sampling", alpha=0.6)
plt.legend()
plt.xlabel('k')
plt.ylabel('Pr[K(R,Q) = k]')
plt.xlim(450,800)
plt.title('Gains inc')
plt.grid(True)
plt.savefig("p_value_Gains.svg", format='svg')
plt.show()

plt.figure(figsize=(10 * 0.5, 8 * 0.5))
# Main part of the script
filename = '../resources/example_github/pvalues_hirt.txt'
k_values, p_values = read_data(filename)
p_values = compute_original_values(p_values)
filename = '../../pokusik/mc-overlaps-master (1)/mc-overlaps-master/resources/example_github/pvalues_direct.txt'
k_values_1, p_values_1 = read_data(filename)
p_values_1 = compute_original_values(p_values_1)
filename = '../resources/example_github/orig1.txt'
k_values_2, p_values_2 = read_data(filename)
p_values_2 = compute_original_values(p_values_2)
plt.plot(k_values, p_values, label="extended MCDP")
plt.plot(k_values_1, p_values_1, label="MCDP",  linestyle="--" ,color="black")
plt.plot(k_values_2, p_values_2, label="direct sampling", alpha=0.6)
plt.legend()
plt.xlabel('k')
plt.ylabel('Pr[K(R,Q) = k]')
plt.xlim(0,80)
plt.title('hirt')
plt.grid(True)
plt.savefig("p_value_hirt.svg", format='svg')
plt.show()

import numpy as np

gap_matrix = np.array([[1,2,3],[4,5,6],[0,0,0]])
interval_matrix = np.array([[8,8,8],[8,8,8],[0,0,0]])

def connect_mch_mod(gap_matrix, interval_matrix):
    # Get the sizes of the adjacency matrices
    size1 = gap_matrix.shape[0]
    size2 = interval_matrix.shape[0]

    # Create a new adjacency matrix to accommodate the combined graphs
    combined_size = size1 + size2 - 2
    combined_adjacency_matrix = np.zeros((combined_size, combined_size), dtype=np.longdouble)
    
    # Copy the adjacency matrix of the first graph into the upper-left corner
    combined_adjacency_matrix[:size1 -1, :size1 -1] = gap_matrix[:-1, :-1]
    combined_adjacency_matrix[(size1 - 1):, :1] = interval_matrix[:-1, -1:]
    return combined_adjacency_matrix

def connect_mch(gap_matrix, interval_matrix):
        # Get the sizes of the adjacency matrices
        size1 = gap_matrix.shape[0]
        size2 = interval_matrix.shape[0]
    
        # Create a new adjacency matrix to accommodate the combined graphs
        combined_size = size1 + size2 - 2
        combined_adjacency_matrix = np.zeros((combined_size, combined_size), dtype=np.longdouble)
        #print(combined_adjacency_matrix)
        # Copy the adjacency matrix of the first graph into the upper-left corner
        combined_adjacency_matrix[:size1, :size1] = gap_matrix
        #print(combined_adjacency_matrix)
        # Copy the adjacency matrix of the second graph into the lower-right corner
        combined_adjacency_matrix[(size1 - 1):, (size1 - 1):] = interval_matrix[:-1, :-1]
        #print(combined_adjacency_matrix)
        combined_adjacency_matrix[(size1 - 1):, :1] = interval_matrix[:-1, -1:]
        #print(combined_adjacency_matrix)
        return combined_adjacency_matrix

# print(connect_mch(gap_matrix, interval_matrix))
# print(connect_mch_mod(gap_matrix, interval_matrix))
# connect_mch(gap_matrix, interval_matrix)