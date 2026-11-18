import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    N = len(seqs)
    if N == 0:
        return np.empty((0,0),dtype=np.int32)
    T_max = max([len(seq) for seq in seqs])
    L = max_len if max_len is not None and max_len > 0 else T_max
    
    o = np.full(shape = (N,L), fill_value = pad_value)
    for i in range(N):
        l = len(seqs[i]) if len(seqs[i]) <= L else L
        o[i,:l] = np.array(seqs[i][:l])
    return o