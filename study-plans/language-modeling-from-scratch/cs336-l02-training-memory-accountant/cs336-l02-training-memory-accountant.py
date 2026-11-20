def memory_accountant(
    param_shapes: list[list[int]], param_bytes_per_element: int,
    grad_bytes_per_element: int, activation_shapes: list[list[int]],
    activation_bytes_per_element: int, optimizer: str,
    optimizer_bytes_per_element: int,
) -> dict:
    """
    Returns integer byte counts for parameters, gradients, activations, optimizer_state, and total.
    """
    import math
    param_num = sum([math.prod(s) for s in param_shapes])
    
    param_bytes = param_num * param_bytes_per_element
    grad_bytes = param_num * grad_bytes_per_element
    act_num = sum([math.prod(s) for s in activation_shapes])
    activation_bytes = act_num * activation_bytes_per_element
    if optimizer == 'sgd':
        optimizer_bytes = 0 * optimizer_bytes_per_element
    elif optimizer == 'adagrad':
        optimizer_bytes = param_num * optimizer_bytes_per_element
    elif optimizer == 'adam':
        optimizer_bytes = 2 * param_num * optimizer_bytes_per_element
    
    return {
        "parameters":param_bytes,
        "gradients":grad_bytes,
        "activations":activation_bytes,
        "optimizer_state":optimizer_bytes,
        "total":param_bytes + grad_bytes + activation_bytes + optimizer_bytes
    }
