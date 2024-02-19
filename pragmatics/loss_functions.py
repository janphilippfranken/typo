import torch
import torch.nn.functional as F


def pragmatic_loss(
    logprobs: torch.FloatTensor, 
    max_iter: int = 100,
    epsilon: float = 1e-10,
) -> torch.FloatTensor:
    """Compute the pragmatic loss for a batch of response log probabilities.
    
    Args:
        logprobs: The log probabilities of the responses. Shape: (constitution_batch_size, constitution_batch_size * response_batch_size).
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

    # use probs as class probabilities to compute the loss
    loss = F.cross_entropy(logprobs, probs, reduction="mean")
 
    return probs, loss 


def kl_divergence(
    probs_policy: torch.FloatTensor, 
    probs_reference: torch.FloatTensor, 
    epsilon=1e-32,
) -> torch.FloatTensor:
    """Compute the KL divergence between the policy and the reference distributions. 
    
    Args:
        probs_policy: Shape: (constitution_batch_size * response_batch_size, response_batch_size).
        probs_reference: Shape: (constitution_batch_size * response_batch_size, response_batch_size).
        epsilon: For numerical stability.
        
    Returns:
        kl_divergence: KL divergence between the policy and reference distributions.
    """
    log_probs_policy = (probs_policy + epsilon).log()
    log_probs_reference = (probs_reference + epsilon).log()

    log_diff = log_probs_policy - log_probs_reference

    kl = (probs_policy * log_diff).sum()
    
    return kl

