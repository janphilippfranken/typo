import torch
import torch.nn.functional as F


def pragmatic_loss_no_labels_no_reference(
    logprobs: torch.FloatTensor, 
    max_iter: int = 100,
    epsilon: float = 1e-10,
) -> torch.FloatTensor:
    """Compute the pragmatic loss for a batch of response log probabilities.
    
    Args:
        logprobs: The log probabilities of the responses. Shape: (constitution_batch_size, constitution_batch_size).
        max_iter: The maximum number of iterations for the pragmatic recursion.
        epsilon: The convergence threshold for the pragmatic recursion.
        
    Returns:
        pragmatic_loss: The pragmatic loss for the batch of responses. 
    """        
    # constitutions compete for responses
    probs = torch.softmax(logprobs, dim=0) 
    
    for _ in range(max_iter):

        # row normalization
        probs = probs / probs.sum(dim=1, keepdim=True)
        
        # check convergence
        col_sums = probs.sum(dim=0)
        if torch.max(col_sums) - torch.min(col_sums) < epsilon:
            break
        
        # column normalization
        probs = probs / probs.sum(dim=0, keepdim=True)
        
    # final row normalization to make sure rows sum to one as these are the class probs
    probs = probs / probs.sum(dim=1, keepdim=True)
    
    # use probs as class probabilities to compute the loss
    loss = F.cross_entropy(logprobs, probs, reduction="mean")
 
    return probs, loss 


def pragmatic_loss_with_reference(
    logprobs_policy: torch.FloatTensor, 
    logprobs_reference: torch.FloatTensor, 
) -> torch.FloatTensor:
    """Compute the pragmatic loss for a batch of response log probabilities.
    
    Args:
        logprobs_policy: The log probabilities of the responses from the policy. Shape: (constitution_batch_size, constitution_batch_size).
        logprobs_reference: The log probabilities of the responses from the reference model without constitutions. Shape: (constitution_batch_size, constitution_batch_size).

    Returns:
        pragmatic_loss: The pragmatic loss for the batch of responses. 
    """        
    # logits 
    logits = logprobs_policy - logprobs_reference
    
    # labels 
    labels = torch.eye(logits.shape[0]).to(logits.device) 

    # loss 
    loss = F.cross_entropy(logits, labels, reduction="mean")
   
    return loss


def pragmatic_loss_no_reference(
    logprobs: torch.FloatTensor,
) -> torch.FloatTensor:
    """Compute the pragmatic loss for a batch of response log probabilities.
    
    Args:
        logprobs: The log probabilities of the responses from the policy. Shape: (batch_size, batch_size).

    Returns:
        pragmatic_loss: The pragmatic loss for the batch of responses.
    """
    # normalization constant 
    logsumexp = torch.logsumexp(logprobs, dim=0, keepdim=True)
    
    # logits 
    logits = logprobs - logsumexp
    
    # labels 
    labels = torch.arange(logits.size(0)).long().to(logits.device)

    # cross entropy
    loss = F.cross_entropy(logits, labels, reduction="mean")
   
    return loss