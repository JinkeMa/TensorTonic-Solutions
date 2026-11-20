import torch

def attention_scores(q: torch.Tensor, k: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Returns scores of shape (batch, heads, query_length, key_length).
    """
    b,sq,dim = q.size()
    _,sk,_ = k.size()
    head_dim = dim //num_heads
    q_ = q.reshape((b,sq,num_heads,-1)).transpose(1,2)
    k_ = k.reshape((b,sk,num_heads,-1)).transpose(1,2)
    return torch.einsum("bnsd,bnld -> bnsl",q_,k_)/torch.sqrt(torch.tensor(head_dim))
