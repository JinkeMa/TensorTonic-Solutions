import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    # Write code here
    M,N = len(A), len(A[0])

    o = np.empty((N,M),dtype=np.float32)
    for i in range(M):
        o[:,i] = np.array(A[i])

    return o

