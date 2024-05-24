import numpy as np
def connect_mch(gap_matrix, interval_matrix):
        # Get the sizes of the adjacency matrices
        size1 = gap_matrix.shape[0]
        size2 = interval_matrix.shape[0]
    
        # Create a new adjacency matrix to accommodate the combined graphs
        combined_size = size1 + size2 - 2
        combined_adjacency_matrix = np.zeros((combined_size, combined_size), dtype=np.longdouble)
        print(combined_adjacency_matrix)
        
        # Copy the adjacency matrix of the first graph into the upper-left corner
        combined_adjacency_matrix[:size1, :size1] = gap_matrix
        print(combined_adjacency_matrix[:size1, :size1])
        print(gap_matrix)
        
        # Copy the adjacency matrix of the second graph into the lower-right corner
        # print(combined_adjacency_matrix[(size1 - 1):, (size1 - 1):] )
        # print(interval_matrix[:-1, :-1])
        combined_adjacency_matrix[(size1 - 1):, (size1 - 1):] = interval_matrix[:-1, :-1]
        combined_adjacency_matrix[(size1 - 1):, :1] = interval_matrix[:-1, -1:]
        
        return combined_adjacency_matrix

a = np.array([[0.25, 0.25, 0.75],
    [0.25, 0.25, 0.75],
    [0,0,0]])

# #b = np.array([[0.5, 0, 0, 0.5],
#     [0.25, 0.25, 0.25, 0.25],
#     [0, 0.3, 0.3, 0.4],
#     [0,0,0,0]])

b = adjacency_matrix = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]
)

a = np.array([
    [0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 3, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 3, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]
)

print(connect_mch(a,b))