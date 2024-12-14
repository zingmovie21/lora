import torch
import numpy as np
from typing import List, Optional, Tuple, Union

def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
) -> float:
    """
    Calculate the shift value for image sequence processing.
    
    Args:
        image_seq_len: Length of the image sequence
        base_seq_len: Base sequence length for calculation
        max_seq_len: Maximum sequence length
        base_shift: Base shift value
        max_shift: Maximum shift value
    
    Returns:
        Calculated shift value
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, int]:
    """
    Retrieve timesteps for the diffusion process.
    
    Args:
        scheduler: Diffusion scheduler
        num_inference_steps: Number of inference steps
        device: Device to use for computation
        timesteps: Optional explicit timesteps
        sigmas: Optional sigma values
        **kwargs: Additional arguments for scheduler
    
    Returns:
        Tuple of (timesteps tensor, number of inference steps)
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed")

    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

    return timesteps, num_inference_steps
