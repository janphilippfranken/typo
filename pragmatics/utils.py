import torch.distributed as dist



def rank0_print(
    *args, 
    **kwargs
) -> None:
    """Print, but only on rank 0."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)