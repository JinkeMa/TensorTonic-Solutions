import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    """
    # Write code here
    w_ = np.asarray(w,dtype=np.float64)
    g_ = np.asarray(g,dtype=np.float64)
    s_ = np.asarray(s,dtype=np.float64)
    
    s_ = beta * s_ + (1-beta) * g_**2
    w_ = w_ - lr * g_ / (np.sqrt(s_ + eps))
    w = w_.tolist()
    s = s_.tolist()
    return w,s