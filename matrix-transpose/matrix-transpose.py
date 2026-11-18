import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    # Write code here

    M,N = len(A), len(A[0])
    o = np.empty((N,M),dtype=np.float64)
    for i in range(M):
         o[:,i] = np.array(A[i])
    return o

    # much slower when using meshgrid
    # X = np.asarray(A,dtype=np.float64)
    # M,N = X.shape
    # x = np.arange(M)
    # y = np.arange(N)
    # i,j = np.meshgrid(x,y,indexing='xy')        
    # return X[i,j]

