import torch

def train_tiny_micrograd_mlp(inputs: torch.Tensor, targets: torch.Tensor, weights: list[torch.Tensor], biases: list[torch.Tensor], learning_rate: float, steps: int) -> tuple:
    """
    Returns final predictions, final loss, trained weights, trained biases, and pre-update loss history.
    """
    from copy import deepcopy

    # tanh(x) = [exp(x) - exp(-x)] / [exp(x) + exp(-x)]
    # tanh'(x) = 1 - tanh(x) ^ 2

    nlayer = len(weights)
        
    # precision
    o = deepcopy(inputs)
    targets_ = targets
    weights_ = deepcopy([w.T for w in weights])
    biases_ = deepcopy(biases)
    
    
    #
    acts = []
    loss_history = []
    loss = 0

    # h: (bs,idim)
    # w: (idim,odim)
    # o = h@w, 
    # o: (bs,odim), grad_o: (bs,odim), grad_w: (idim,odim), grad_h: (bs, idim)
    # grad_h = grad_o @ w.T, grad_w = h.T @ grad_o, grad_b = 1*grad_o

    def predict(pred, train=False):
        # forward
        for i in range(nlayer):
            pred = pred@weights_[i] + biases_[i]
            # tanh
            pred = torch.tanh(pred) # (torch.exp(pred) - torch.exp(-pred)) / (torch.exp(pred) + torch.exp(-pred))
            if i != nlayer -1 and train:
                acts.append(pred)
        return pred.squeeze(-1)
    
    def compute_loss(pred:torch.Tensor, targets:torch.Tensor):
        return torch.sum((pred.squeeze(-1) - targets)**2)
        
    for step in range(steps):
        
        acts.clear()
        acts.append(o)
        pred = deepcopy(o)

        # forward
        pred = predict(pred, True)
        
        # loss
        loss = compute_loss(pred, targets_) # torch.sum((pred - targets)**2)
        loss_history.append(loss)
        # (bs,odim)
        grad_p = 2 * (pred - targets_)
        grad_p = (1-pred**2) * grad_p
        grad_p = grad_p.unsqueeze(-1)
        # backward
        for i in range(nlayer):
            indx = -(i+1)
            # tanh'(x)
            
            gw = acts[indx].T@grad_p
            gb = grad_p.sum(dim=0)
            grad_p = grad_p@weights_[indx].T
            grad_p = (1-acts[indx]**2) * grad_p
            
            weights_[indx] -= learning_rate * gw
            biases_[indx]  -= learning_rate * gb
    
    # predictions
    predictions = deepcopy(o)
    predictions = predict(predictions,False)
    final_loss = compute_loss(predictions,targets_)
    # 
    # weights
    trained_weights = [w.T for w in weights_]
    trained_biases = biases_
    
    #
    return predictions,final_loss,trained_weights,trained_biases,loss_history
