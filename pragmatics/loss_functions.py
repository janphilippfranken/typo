import torch
import torch.nn.functional as F


def pragmatic_loss(
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


def kl_divergence(
    probs_policy: torch.FloatTensor, 
    probs_reference: torch.FloatTensor, 
    epsilon=1e-32,
) -> torch.FloatTensor:
    """Compute the KL divergence between the policy and the reference distributions. 
    
    Args:
        probs_policy: Shape: (constitution_batch_size, constitution_batch_size).
        probs_reference: Shape:(constitution_batch_size, constitution_batch_size).
        epsilon: For numerical stability.
        
    Returns:
        kl_divergence: KL divergence between the policy and reference distributions.
    """
    log_probs_policy = (probs_policy + epsilon).log()
    log_probs_reference = (probs_reference + epsilon).log()

    log_diff = log_probs_policy - log_probs_reference

    kl = (probs_policy * log_diff).sum(dim=1).mean()
    
    return kl


def kl_divergence_from_logits(logprobs_policy_logits: torch.FloatTensor, 
                              logprobs_reference_logits: torch.FloatTensor, 
                              epsilon: float = 1e-32) -> torch.FloatTensor:
    """
    Compute the KL divergence between the policy and the reference distributions using logits.
    
    Args:
        logprobs_policy_logits: Logits from the policy distribution. Shape: (batch_size, num_classes).
        logprobs_reference_logits: Logits from the reference distribution. Shape: (batch_size, num_classes).
        epsilon: Small value added for numerical stability when converting probabilities to log probabilities.
        
    Returns:
        kl_divergence: Mean KL divergence between the policy and reference distributions across the batch.
    """
    probs_policy = F.softmax(logprobs_policy_logits, dim=1)
    probs_reference = F.softmax(logprobs_reference_logits, dim=1)

    log_probs_policy = probs_policy.log() + epsilon
    log_probs_reference = probs_reference.log() + epsilon

    kl_divergence = (probs_policy * (log_probs_policy - log_probs_reference)).sum(dim=1).mean()
    
    return kl_divergence


def pragmatic_loss_with_labels(
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
    logits = logprobs_policy - (logprobs_policy.sum(dim=0) / logprobs_policy.shape[0])
    
    # labels 
    labels = torch.eye(logits.shape[0]).to(logits.device)

    # loss 
    loss = F.cross_entropy(logits, labels, reduction="mean")
   
    return loss


def kl_divergence_from_logits_per_token(
    logprobs_policy_logits: torch.FloatTensor, 
    logprobs_reference_logits: torch.FloatTensor, 
    loss_mask_policy: torch.BoolTensor,
    loss_mask_reference: torch.BoolTensor,
) -> torch.FloatTensor:
    """
    Compute the KL divergence between the policy and reference distributions per token in a numerically stable way.
    
    Args:
        logprobs_policy_logits: Logits from the policy distribution. Shape: (batch_size, sequence_length).
        logprobs_reference_logits: Logits from the reference distribution. Shape: (batch_size, sequence_length).
        loss_mask_policy: Boolean mask for policy logits.
        loss_mask_reference: Boolean mask for reference logits.
        
    Returns:
        kl_divergences.mean(): Mean of KL divergence for each batch entry.
    """
    kl_divergences = torch.zeros(logprobs_policy_logits.shape[0])

    for i in range(logprobs_policy_logits.shape[0]):
        
        # mask
        policy_logits = logprobs_policy_logits[i][~loss_mask_policy[i]]
        reference_logits = logprobs_reference_logits[i][~loss_mask_reference[i]]

        # logsumexp
        max_policy_logits = policy_logits.max()
        max_reference_logits = reference_logits.max()
        
        log_probs_policy = policy_logits - (max_policy_logits + (policy_logits - max_policy_logits).exp().sum().log())
        log_probs_reference = reference_logits - (max_reference_logits + (reference_logits - max_reference_logits).exp().sum().log())

        # exp 
        probs_policy = log_probs_policy.exp()

        # kl 
        kl_divergence = (probs_policy * (log_probs_policy - log_probs_reference)).sum()
        kl_divergences[i] = kl_divergence

    return kl_divergences.mean()