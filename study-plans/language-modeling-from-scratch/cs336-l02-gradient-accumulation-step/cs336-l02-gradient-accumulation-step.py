import torch

def gradient_accumulation_step(
    param: torch.Tensor, microbatch_inputs: list[torch.Tensor],
    microbatch_targets: list[torch.Tensor], lr: float,
) -> dict:
    """
    Returns new_param and full_grad tensors in a dictionary.
    """
    D = param.size()
    N = sum(len(s) for s in microbatch_inputs)
    steps = len(microbatch_inputs)
    # forward
    hiddens = [(param * input).sum(dim=-1) for input in microbatch_inputs]
    # backward
    microbatch_grads = torch.zeros_like(param)
    for i in range(steps):
        grad_h = 2 * (hiddens[i] - microbatch_targets[i]) / N
        # microbatch_grads.append(gradh)
        grad_w = grad_h[:,None] * microbatch_inputs[i] #  (b,1) * (b,D)
        microbatch_grads +=grad_w.sum(dim=0)
    new_param = param - lr * microbatch_grads
    return {
        "new_param":new_param,
        "full_grad":microbatch_grads
    }
    