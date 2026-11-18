import numpy as np

def apply_homogeneous_transform(T: list, points: list) -> np.ndarray:
    """
    Returns transformed points with shape (3,) or (N, 3).
    """
    # Write code here
    pts = np.asarray(points,dtype=np.float64)
    pts = pts[None,:] if len(pts.shape) == 1 else pts

    suplem = np.ones((len(pts)),dtype=np.float64)[:,None]
    hp = np.concat((pts,suplem),axis=-1)

    T_ = np.asarray(T,dtype=np.float64)
    o = T_ @ hp.T
    oT = o.T[:,:-1]
    oT = oT.squeeze() if pts.shape[0] == 1 else oT
    return oT
    
    