import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    # Write code here
    X = np.asarray(x,dtype=np.float64)
    p_ = rng.random(X.shape)
    mask = p_ <= (1-p)
    pattern = mask / (1-p)
    X = X * pattern
    return X,pattern