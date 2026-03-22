import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Literal


def nearest_divisor(n: int, target: int) -> int:
    """
    Find the divisor of n closest to target.
    
    If two divisors are equidistant, returns the smaller one.
    Useful for Kronecker factorization to find factors close to sqrt(n).
    
    Args:
        n: The number to factorize
        target: The target divisor value
        
    Returns:
        The divisor of n closest to target
    """
    if target <= 0:
        target = 1
    if target >= n:
        target = n
    
    # Search outward from target
    best = 1
    best_dist = abs(target - 1)
    
    # Check divisors up to sqrt(n) and their complements
    i = 1
    while i * i <= n:
        if n % i == 0:
            # i is a divisor
            dist_i = abs(target - i)
            if dist_i < best_dist:
                best = i
                best_dist = dist_i
            # n // i is also a divisor
            complement = n // i
            dist_c = abs(target - complement)
            if dist_c < best_dist:
                best = complement
                best_dist = dist_c
        i += 1
    
    return best
    
# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _dense_target_std(fan_in: int, fan_out: int) -> float:
    """Xavier/Glorot target std for a dense matrix."""
    return math.sqrt(2.0 / (fan_in + fan_out))


def _scale_tensor_std(tensor: torch.Tensor, target_std: float, eps: float = 1e-8) -> torch.Tensor:
    """
    Per-sample normalization: rescale each sample in the batch so its
    flattened std matches target_std. Avoids batch-level mixing.
    """
    if tensor.dim() < 2:
        raise ValueError("Tensors must be at least 2D for per-sample std scaling.")
    flat_std = tensor.flatten(1).std(dim=1, keepdim=True, unbiased=False)
    scale = target_std / (flat_std + eps)
    reshape_dims = [tensor.shape[0]] + [1] * (tensor.dim() - 1)
    return tensor * scale.view(*reshape_dims)


def _kron_apply(A: torch.Tensor, B: torch.Tensor, x: torch.Tensor, p: int, q: int) -> torch.Tensor:
    """
    Apply Kronecker product (A ⊗ B) to x without materializing the full matrix.
    
    x: (batch, dim) where dim = p * q
    A: (batch, p, p)  
    B: (batch, q, q)
    
    Equivalent to x @ kron(A, B) but O(dim*(p+q)) instead of O(dim^2).
    Reshapes x to (batch, p, q), applies B @ x @ A^T, flattens back.
    """
    batch = x.shape[0]
    # x: (batch, p, q)
    h = x.view(batch, p, q)
    # Apply B on the right: (batch, p, q) @ (batch, q, q)^T -> (batch, p, q)
    h = torch.bmm(h, B.transpose(-2, -1))
    # Apply A on the left: (batch, p, p)^T @ (batch, p, q) -> (batch, p, q)
    h = torch.bmm(A, h)
    return h.reshape(batch, p * q)


def _make_bit_reversal_permutation(n: int) -> torch.Tensor:
    """Create a bit-reversal permutation for dimension n (must be power of 2)."""
    bits = int(math.log2(n))
    indices = []
    for i in range(n):
        reversed_i = int(bin(i)[2:].zfill(bits)[::-1], 2)
        indices.append(reversed_i)
    return torch.tensor(indices, dtype=torch.long)


def _make_stride_permutation(n: int, stride: int) -> torch.Tensor:
    """Stride permutation: index i -> (i * stride) % n, with handling for non-coprime cases."""
    return torch.tensor([(i * stride) % n for i in range(n)], dtype=torch.long)

