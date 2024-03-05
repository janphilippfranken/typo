import torch
import torch.nn.functional as F


def pragmatic_loss_no_labels_no_reference(
    logprobs: torch.FloatTensor, 
    max_iter: int = 100,
    epsilon: float = 1e-10,
) -> torch.FloatTensor:
    """Compute the pragmatic loss for a batch of response log probabilities without labels or reference.
    
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


def pragmatic_clip_loss(
    logprobs: torch.FloatTensor,
) -> torch.FloatTensor:
    """Compute the pragmatic loss for a batch of response log probabilities.
    
    Args:
        logprobs: The log probabilities of the responses from the policy. Shape: (batch_size, batch_size).

    Returns:
        pragmatic_loss: The pragmatic loss for the batch of responses.
    """
    # normalization constants
    logsumexp = torch.logsumexp(logprobs, dim=0, keepdim=True)
    
    # logits
    logits = logprobs - logsumexp
    logits_t = logits.t()
    
    labels = torch.arange(logprobs.size(0)).long().to(logprobs.device)
    
    loss_row = F.cross_entropy(logits, labels, reduction="mean")
    loss_col = F.cross_entropy(logits_t, labels, reduction="mean")
    
    return (loss_row + loss_col) / 2


def pragmatic_token_loss(
    c0r0: torch.FloatTensor,
    c1r0: torch.FloatTensor,
    c0r1: torch.FloatTensor,
    c1r1: torch.FloatTensor,
) -> torch.FloatTensor:
    """Compute the pragmatic loss for each token."""
    # concatenate 
    r0 = torch.stack((c0r0, c1r0), dim=0)
    r1 = torch.stack((c0r1, c1r1), dim=0)
   
    # normalization constants
    r0logsumexp = torch.logsumexp(r0, dim=0, keepdim=True)
    r1logsumexp = torch.logsumexp(r1, dim=0, keepdim=True)
    
    # logits
    r0logits = r0 - r0logsumexp
    r1logits = r1 - r1logsumexp
    
    # labels
    r0labels = torch.FloatTensor([1, 0]).repeat(r0logits.size(1), 1).to(r0logits.device)
    r1labels = torch.FloatTensor([0, 1]).repeat(r1logits.size(1), 1).to(r1logits.device)
    
    # loss
    r0loss = F.cross_entropy(r0logits.t(), r0labels, reduction="mean")
    r1loss = F.cross_entropy(r1logits.t(), r1labels, reduction="mean")
    
    return (r0loss +  r1loss) / 2
