def generate_anchors(feature_size: int, image_size: float, scales: list[float], aspect_ratios: list[float]) -> list[list[float]]:
    """
    Returns a list of [x1, y1, x2, y2] anchor boxes.
    """
    import numpy as np
    # Write code here
    stride = image_size // feature_size
    x,y = np.arange(feature_size), np.arange(feature_size)
    i,j = np.meshgrid(y,x,indexing='ij')
    cy = (i + 0.5) * stride
    cx = (j + 0.5) * stride
    cy = cy[...,None]
    cx = cx[...,None]
    # w,h
    scales_ = np.asarray(scales,dtype=np.float64)
    ar = np.asarray(aspect_ratios,dtype=np.float64)
    s = np.expand_dims(scales_,axis=-1)
    w,h = s * np.sqrt(ar), s / np.sqrt(ar)

    # cx,cy, (st,st)
    # w,h, (sc,ar)
    w,h = w.flatten()[None,None,:], h.flatten()[None,None,:]
    tly,tlx = cy - h/2, cx - w/2
    bry,brx = cy + h/2, cx + w/2
    o = np.stack((tlx,tly,brx,bry),axis=-1).reshape((-1,4))
    
    return o.tolist()