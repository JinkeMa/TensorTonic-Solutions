import torch

def rmsnorm(x: torch.Tensor, g: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Returns the RMS-normalized tensor with the same shape and dtype as x.
    """
    d = x.size()[-1]
    # (d,) * (n,d)
    return g[None,:] * x / torch.sqrt(x.square().sum(dim=-1,keepdim=True) / d + epsilon)
