import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    # Write code here
    f = base ** (np.arange(stop=d_model,step = 2) / d_model)
    pos = np.arange(seq_len)
    freq = np.expand_dims(pos,axis=1) * 1 / f
    o = np.empty((seq_len,d_model),dtype=np.float64)
    sin, cos = np.sin(freq), np.cos(freq)
    o[:,0::2] = sin
    o[:,1::2] = cos if d_model % 2== 0 else cos[:,:-1]
    return o