"""
Dynamic Transform Layers for Neural Networks

A collection of input-conditioned transformation layers exploring structured 
hypernetwork designs. Each layer generates data-dependent transforms using 
parameter-efficient structures (, low-rank, block-diagonal, spectral).

Design principles:
- Static base + dynamic delta where possible (guaranteed >= baseline)
- Careful initialization so dynamic component starts near zero
- Per-sample std normalization for stable training
- Support for both per-token (variant 1) and per-sequence (variant 2) conditioning
- Residual connections to preserve representation rank

Main experiments:
2. SpectralModulation      - lightweight FFT-based layer
3. ButterflyTransform      - input-conditioned fast transform via 2x2 blocks
4. CompositeDynamicFFN     - combines multiple structured transforms

Attention-generated structured transforms (parameter-efficient generation):
6. AttnGenButterfly        - cross-attention generates butterfly block params
7. AttnGenSpectralFFN      - cross-attention generates spectral mask + FFN delta
8. AttnGenSwiGLU          - cross-attention generates SwiGLU gate conditioning
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Literal

from utils import nearest_divisor, _dense_target_std, _make_bit_reversal_permutation, _make_stride_permutation


# ---------------------------------------------------------------------------
# Experiment 1: Low-Rank Hyper Gate (generalized SwiGLU)
# ---------------------------------------------------------------------------

class LowRankHyperGate(nn.Module):
    """
    SwiGLU with data-dependent low-rank/ perturbations on weights.
    
    Standard SwiGLU: out = (xW1 ⊙ swish(xV)) W2
    
    This layer generates sparse/structured deltas for SwiGLU weights:
        W1_eff = W1 + δ_w1(x)   (linear path)
        V_eff  = V  + δ_v(x)    (gate path)
        W2_eff = W2 + δ_w2(h)   (output projection)
    
    Factor types (factor_type):
        "lowrank":       δ(x) = x @ U @ V^T, U/V are (in_dim, rank) and (out_dim, rank)
        "multiplicative": W_eff = W ⊙ (row ⊗ col^T), applied as col * ((x * row) @ W)
                         Rank-1 multiplicative modulation, extremely parameter efficient
    
    Perturbation modes (delta_target):
        Input-space (dim → dim):
            "gate_input":    V_eff = V, input to V gets delta
            "linear_input":  W1_eff = W1, input to W1 gets delta  
            "both_input":    both W1 and V see input delta
        
        Output-space (generates actual weight deltas):
            "gate_output":   δ_v: dim→r→ff_dim
            "linear_output": δ_w1: dim→r→ff_dim
            "both_output":   δ_v and δ_w1 both get deltas
            "w2_output":     δ_w2: ff_dim→r→dim
            "full":          all three (δ_w1, δ_v, δ_w2)
    
    Args:
        dim: model dimension
        ff_mult: feedforward expansion factor
        rank: rank of the dynamic perturbation (for lowrank)
        bottleneck_dim: dimension of the generation bottleneck
        delta_target: where to apply the perturbation (see above)
        factor_type: "lowrank", or "multiplicative"
        use_bmm: if True, use bmm/matmul (better TF32); if False, use einsum
        pre_norm: if True, apply RMSNorm at start (disable if TransformerBlock already norms)
        act_fn: activation function for gating
    """

    def __init__(
        self,
        dim: int,
        ff_mult: int = 4,
        rank: int = 16,
        bottleneck_dim: int = 128,
        delta_target: Literal["gate_input", "linear_input", "both_input", 
                              "gate_output", "linear_output", "both_output",
                              "w2_output", "full"] = "gate_input",
        factor_type: Literal["lowrank", "multiplicative"] = "lowrank",
        use_bmm: bool = True,
        pre_norm: bool = False,
        act_fn: nn.Module = nn.SiLU,
    ):
        super().__init__()
        self.rank = rank
        self.dim = dim
        self.ff_dim = dim * ff_mult
        self.delta_target = delta_target
        self.factor_type = factor_type
        self.use_bmm = use_bmm
        self.pre_norm = pre_norm

        # Static SwiGLU components
        self.w1 = nn.Linear(dim, self.ff_dim, bias=False)
        self.w2 = nn.Linear(self.ff_dim, dim, bias=False)
        self.v_base = nn.Linear(dim, self.ff_dim, bias=False)

        # Dynamic path: bottleneck → factors
        self.hyper_down = nn.Linear(dim, bottleneck_dim, bias=False)
        self.hyper_act = nn.GELU()
        
        if factor_type == "lowrank":
            # Determine how many factor pairs we need based on delta_target
            if delta_target in ("gate_input", "linear_input", "both_input"):
                # Single dim→dim delta
                factor_size = 2 * dim * rank
            elif delta_target in ("gate_output", "linear_output"):
                # Single dim→ff_dim delta  
                factor_size = (dim + self.ff_dim) * rank
            elif delta_target == "both_output":
                # Two dim→ff_dim deltas (for w1 and v)
                factor_size = 2 * (dim + self.ff_dim) * rank
            elif delta_target == "w2_output":
                # Single ff_dim→dim delta
                factor_size = (self.ff_dim + dim) * rank
            elif delta_target == "full":
                # Three deltas: w1 (dim→ff), v (dim→ff), w2 (ff→dim)
                factor_size = 2 * (dim + self.ff_dim) * rank + (self.ff_dim + dim) * rank
            else:
                raise ValueError(f"Unknown delta_target: {delta_target}")
            
            self.hyper_to_factors = nn.Linear(bottleneck_dim, factor_size, bias=False)
            nn.init.normal_(self.hyper_to_factors.weight, std=0.01)

        
        elif factor_type == "multiplicative":
            # Multiplicative rank-1: generate row (scales input) and col (scales output)
            # For W: (in_dim, out_dim), we generate row: (in_dim,), col: (out_dim,)
            # Applied as: col * ((x * row) @ W) = x @ (W ⊙ (row ⊗ col^T))
            if delta_target in ("gate_input", "linear_input", "both_input"):
                # dim → dim: row and col both have size dim
                mult_size = 2 * dim
            elif delta_target in ("gate_output", "linear_output"):
                # dim → ff_dim: row is dim, col is ff_dim
                mult_size = dim + self.ff_dim
            elif delta_target == "both_output":
                # Two (dim → ff_dim): 2 * (dim + ff_dim)
                mult_size = 2 * (dim + self.ff_dim)
            elif delta_target == "w2_output":
                # ff_dim → dim: row is ff_dim, col is dim
                mult_size = self.ff_dim + dim
            elif delta_target == "full":
                # w1, v: dim → ff_dim, w2: ff_dim → dim
                mult_size = 2 * (dim + self.ff_dim) + (self.ff_dim + dim)
            else:
                raise ValueError(f"Unknown delta_target: {delta_target}")
            
            self.hyper_to_mult = nn.Linear(bottleneck_dim, mult_size, bias=True)
            nn.init.normal_(self.hyper_to_mult.weight, std=0.01)
            nn.init.ones_(self.hyper_to_mult.bias)

        self.delta_scale = nn.Parameter(torch.zeros(1))
        self.act = act_fn()

    def _apply_lowrank_delta(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Apply low-rank delta: x + scale * (x @ U @ V^T)
        x: (B, S, in_dim)
        u: (B, S, in_dim, rank)
        v: (B, S, out_dim, rank)
        Returns: (B, S, out_dim)
        """
        scaled_x_delta = self._compute_lowrank_delta(x, u, v)
        return x + scaled_x_delta

    def _compute_lowrank_delta(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute just the delta (without adding to x): scale * (x @ U @ V^T)
        """
        B, S = x.shape[:2]
        if self.use_bmm:
            x_flat = x.reshape(B * S, 1, -1)
            u_flat = u.reshape(B * S, u.shape[-2], self.rank)
            v_flat = v.reshape(B * S, v.shape[-2], self.rank)
            x_low = torch.bmm(x_flat, u_flat).squeeze(1)
            x_delta = torch.bmm(x_low.unsqueeze(1), v_flat.transpose(1, 2)).squeeze(1)
            x_delta = x_delta.view(B, S, -1)
        else:
            x_low = torch.einsum('bsi,bsir->bsr', x, u)
            x_delta = torch.einsum('bsr,bsor->bso', x_low, v)
        return self.delta_scale * x_delta

    def _apply_multiplicative(self, x: torch.Tensor, weight: nn.Linear, 
                               row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        """
        Apply multiplicative rank-1 modulation: col * ((x * row) @ W)
        
        This is equivalent to: x @ (W ⊙ (row ⊗ col^T))
        but computed efficiently without materializing the outer product.
        
        The modulation is: W_eff = W * (1 + scale * (row ⊗ col^T))
        At init (scale=0): W_eff = W (identity)
        
        x: (B, S, in_dim)
        row: (B, S, in_dim) - scales input features
        col: (B, S, out_dim) - scales output features
        Returns: (B, S, out_dim)
        """
        # Compute base output
        base_out = weight(x)  # (B, S, out_dim)
        # Compute modulated output: col * ((x * row) @ W)
        modulated = weight(x * row) * col
        # Blend: base + scale * (modulated - base) = (1-scale)*base + scale*modulated
        # When scale=0: returns base. When scale=1: returns modulated.
        return base_out + self.delta_scale * (modulated - base_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape

        # Generate factors
        bottleneck = self.hyper_act(self.hyper_down(x))  # (B, S, bottleneck)
        
        if self.factor_type == "multiplicative":
            # Multiplicative rank-1 path
            mult_params = self.hyper_to_mult(bottleneck)
            
            if self.delta_target in ("gate_input", "linear_input", "both_input"):
                # dim → dim: row and col both have size dim
                row, col = mult_params.chunk(2, dim=-1)
                
                if self.delta_target == "gate_input":
                    h_linear = self.w1(x)
                    h_gate = self._apply_multiplicative(x, self.v_base, row, col)
                elif self.delta_target == "linear_input":
                    h_linear = self._apply_multiplicative(x, self.w1, row, col)
                    h_gate = self.v_base(x)
                else:  # both_input
                    h_linear = self._apply_multiplicative(x, self.w1, row, col)
                    h_gate = self._apply_multiplicative(x, self.v_base, row, col)
            
            elif self.delta_target in ("gate_output", "linear_output"):
                # dim → ff_dim: row is dim, col is ff_dim
                row, col = mult_params.split([D, self.ff_dim], dim=-1)
                
                if self.delta_target == "gate_output":
                    h_linear = self.w1(x)
                    h_gate = self._apply_multiplicative(x, self.v_base, row, col)
                else:  # linear_output
                    h_linear = self._apply_multiplicative(x, self.w1, row, col)
                    h_gate = self.v_base(x)
            
            elif self.delta_target == "both_output":
                # Two (dim → ff_dim): separate row/col for w1 and v
                row_w1, col_w1, row_v, col_v = mult_params.split(
                    [D, self.ff_dim, D, self.ff_dim], dim=-1)
                
                h_linear = self._apply_multiplicative(x, self.w1, row_w1, col_w1)
                h_gate = self._apply_multiplicative(x, self.v_base, row_v, col_v)
            
            elif self.delta_target == "w2_output":
                # ff_dim → dim: row is ff_dim, col is dim
                row, col = mult_params.split([self.ff_dim, D], dim=-1)
                
                h_linear = self.w1(x)
                h_gate = self.v_base(x)
                h = h_linear * self.act(h_gate)
                return self._apply_multiplicative(h, self.w2, row, col)
            
            elif self.delta_target == "full":
                # w1, v: dim → ff_dim, w2: ff_dim → dim
                row_w1, col_w1, row_v, col_v, row_w2, col_w2 = mult_params.split(
                    [D, self.ff_dim, D, self.ff_dim, self.ff_dim, D], dim=-1)
                
                h_linear = self._apply_multiplicative(x, self.w1, row_w1, col_w1)
                h_gate = self._apply_multiplicative(x, self.v_base, row_v, col_v)
                h = h_linear * self.act(h_gate)
                return self._apply_multiplicative(h, self.w2, row_w2, col_w2)
            
            h = h_linear * self.act(h_gate)
            return self.w2(h)
        
        else:
            # Low-rank path
            factors = self.hyper_to_factors(bottleneck)
            
            if self.delta_target in ("gate_input", "linear_input", "both_input"):
                # Input-space delta: dim → dim
                u, v = factors.chunk(2, dim=-1)
                u = u.view(B, S, D, self.rank)
                v = v.view(B, S, D, self.rank)
                x_eff = self._apply_lowrank_delta(x, u, v)
                
                if self.delta_target == "gate_input":
                    h_linear = self.w1(x)
                    h_gate = self.v_base(x_eff)
                elif self.delta_target == "linear_input":
                    h_linear = self.w1(x_eff)
                    h_gate = self.v_base(x)
                else:  # both_input
                    h_linear = self.w1(x_eff)
                    h_gate = self.v_base(x_eff)
            
            elif self.delta_target in ("gate_output", "linear_output"):
                # Output-space delta for one of w1/v: dim → ff_dim
                u_size = D * self.rank
                v_size = self.ff_dim * self.rank
                u, v = factors.split([u_size, v_size], dim=-1)
                u = u.view(B, S, D, self.rank)
                v = v.view(B, S, self.ff_dim, self.rank)
                
                delta = self._compute_lowrank_delta(x, u, v)
                
                if self.delta_target == "gate_output":
                    h_linear = self.w1(x)
                    h_gate = self.v_base(x) + delta
                else:  # linear_output
                    h_linear = self.w1(x) + delta
                    h_gate = self.v_base(x)
            
            elif self.delta_target == "both_output":
                # Output-space deltas for both w1 and v: 2 × (dim → ff_dim)
                u_size = D * self.rank
                v_size = self.ff_dim * self.rank
                u_w1, v_w1, u_v, v_v = factors.split([u_size, v_size, u_size, v_size], dim=-1)
                
                u_w1 = u_w1.view(B, S, D, self.rank)
                v_w1 = v_w1.view(B, S, self.ff_dim, self.rank)
                u_v = u_v.view(B, S, D, self.rank)
                v_v = v_v.view(B, S, self.ff_dim, self.rank)
                
                delta_w1 = self._compute_lowrank_delta(x, u_w1, v_w1)
                delta_v = self._compute_lowrank_delta(x, u_v, v_v)
                
                h_linear = self.w1(x) + delta_w1
                h_gate = self.v_base(x) + delta_v
            
            elif self.delta_target == "w2_output":
                # Output-space delta for w2: ff_dim → dim
                u_size = self.ff_dim * self.rank
                v_size = D * self.rank
                u, v = factors.split([u_size, v_size], dim=-1)
                u = u.view(B, S, self.ff_dim, self.rank)
                v = v.view(B, S, D, self.rank)
                
                h_linear = self.w1(x)
                h_gate = self.v_base(x)
                h = h_linear * self.act(h_gate)
                
                delta_w2 = self._compute_lowrank_delta(h, u, v)
                return self.w2(h) + delta_w2
            
            elif self.delta_target == "full":
                # All three deltas: w1, v (dim→ff), w2 (ff→dim)
                u_up_size = D * self.rank
                v_up_size = self.ff_dim * self.rank
                u_down_size = self.ff_dim * self.rank
                v_down_size = D * self.rank
                
                u_w1, v_w1, u_v, v_v, u_w2, v_w2 = factors.split(
                    [u_up_size, v_up_size, u_up_size, v_up_size, u_down_size, v_down_size], dim=-1
                )
                
                u_w1 = u_w1.view(B, S, D, self.rank)
                v_w1 = v_w1.view(B, S, self.ff_dim, self.rank)
                u_v = u_v.view(B, S, D, self.rank)
                v_v = v_v.view(B, S, self.ff_dim, self.rank)
                u_w2 = u_w2.view(B, S, self.ff_dim, self.rank)
                v_w2 = v_w2.view(B, S, D, self.rank)
                
                delta_w1 = self._compute_lowrank_delta(x, u_w1, v_w1)
                delta_v = self._compute_lowrank_delta(x, u_v, v_v)
                
                h_linear = self.w1(x) + delta_w1
                h_gate = self.v_base(x) + delta_v
                h = h_linear * self.act(h_gate)
                
                delta_w2 = self._compute_lowrank_delta(h, u_w2, v_w2)
                return self.w2(h) + delta_w2
            
            # SwiGLU combination (for targets that don't return early)
            h = h_linear * self.act(h_gate)
            return self.w2(h)




def parse_base_weight_mode(mode: str) -> tuple[str, Optional[int]]:
    """Parse base_weight_mode string into (type, rank).
    
    Returns:
        ("regular", None) for "regular"
        ("none", None) for "none"
        ("lowrank", N) for "lowrank_N"
    """
    if mode == "regular":
        return ("regular", None)
    elif mode == "none":
        return ("none", None)
    elif mode.startswith("lowrank_"):
        try:
            rank = int(mode.split("_")[1])
            return ("lowrank", rank)
        except (IndexError, ValueError):
            raise ValueError(f"Invalid base_weight_mode: {mode}. Expected 'lowrank_N' where N is an integer.")
    else:
        raise ValueError(f"Unknown base_weight_mode: {mode}. Expected 'regular', 'none', or 'lowrank_N'.")


class LowRankHyperFFN(nn.Module):
    """
    Simple FFN with data-dependent low-rank perturbations on weights.
    
    Standard FFN: out = act(x @ W1) @ W2
    
    This layer generates sparse/structured deltas for FFN weights:
        W1_eff = W1 + δ_w1(x)   (up projection)
        W2_eff = W2 + δ_w2(h)   (down projection)
    
    Factor types (factor_type):
        "lowrank":       δ(x) = x @ U @ V^T, U/V are (in_dim, rank) and (out_dim, rank)
        "multiplicative": W_eff = W ⊙ (row ⊗ col^T), applied as col * ((x * row) @ W)
                         Rank-1 multiplicative modulation, extremely parameter efficient
        "circulant":     δ(x) = circ_convolve(x, c) via FFT. Generates one vector c.
                         O(d) params, O(d log d) compute. Translation-equivariant.
        "basis":         δ(x) = Σ_k α_k(x) * B_k where B_k are frozen random matrices.
                         Only n_bases params generated per token! Very cheap.
        "fastfood":      δ(x) = S H G Π H B x (Le et al. 2013)
                         Generates diagonals S,G,B dynamically. O(d) params, O(d log d) compute.
                         Only supports square transforms (w1_input, w2_input).
    
    Perturbation modes (delta_target):
        "w1_input":   δ on W1 input (dim → dim), x_eff = x + δ(x), then W1(x_eff)
        "w1_output":  δ on W1 output (dim → ff_dim), h = W1(x) + δ(x)
        "w2_input":   δ on W2 input (ff_dim → ff_dim), h_eff = h + δ(h)
        "w2_output":  δ on W2 output (ff_dim → dim), out = W2(h) + δ(h)
        "full":       δ on both W1 output and W2 output
    
    Base weight modes (base_weight_mode):
        "regular":   Standard nn.Linear weights (default)
        "lowrank_N": Base weights formed from low-rank matrices of rank N
        "none":      No base weights (pure dynamic transform)
    
    Args:
        dim: model dimension
        ff_mult: feedforward expansion factor
        rank: rank of the dynamic perturbation (for lowrank), or n_bases for basis
        bottleneck_dim: dimension of the generation bottleneck
        delta_target: where to apply the perturbation (see above)
        factor_type: "lowrank", "multiplicative", "circulant", "basis", or "fastfood"
        use_bmm: if True, use bmm/matmul (better TF32); if False, use einsum
        act_fn: activation function
        base_weight_mode: how to parameterize base weights (see above)
    """

    def __init__(
        self,
        dim: int,
        ff_mult: int = 4,
        rank: int = 16,
        bottleneck_dim: int = 128,
        delta_target: Literal["w1_input", "w1_output", "w2_input", "w2_output", "full"] = "w1_output",
        factor_type: Literal["lowrank", "multiplicative", "circulant", "basis", "fastfood"] = "lowrank",
        use_bmm: bool = True,
        act_fn: nn.Module = nn.SiLU,
        base_weight_mode: str = "regular",
    ):
        super().__init__()
        self.rank = rank
        self.dim = dim
        self.ff_dim = dim * ff_mult
        self.delta_target = delta_target
        self.factor_type = factor_type
        self.use_bmm = use_bmm
        
        # Parse base weight mode
        base_type, base_rank = parse_base_weight_mode(base_weight_mode)
        self.base_weight_type = base_type
        self.base_rank = base_rank

        # Static FFN components based on base_weight_mode
        if base_type == "regular":
            self.w1 = nn.Linear(dim, self.ff_dim, bias=True)
            # zero bias
            nn.init.zeros_(self.w1.bias)
            self.w2 = nn.Linear(self.ff_dim, dim, bias=True)
            # zero bias
            nn.init.zeros_(self.w2.bias)
        elif base_type == "lowrank":
            # W1 = w1_u @ w1_v.T, W2 = w2_u @ w2_v.T
            self.w1_u = nn.Parameter(torch.randn(dim, base_rank) * (1.0 / math.sqrt(dim)))
            self.w1_v = nn.Parameter(torch.randn(self.ff_dim, base_rank) * (1.0 / math.sqrt(base_rank)))
            self.w2_u = nn.Parameter(torch.randn(self.ff_dim, base_rank) * (1.0 / math.sqrt(self.ff_dim)))
            self.w2_v = nn.Parameter(torch.randn(dim, base_rank) * (1.0 / math.sqrt(base_rank)))
        # else: base_type == "none" - no base weights

        # Dynamic path: bottleneck → factors
        self.hyper_down = nn.Linear(dim, bottleneck_dim, bias=True)
        # zero bias
        nn.init.zeros_(self.hyper_down.bias)
        self.hyper_act = nn.GELU()
        
        if factor_type == "lowrank":
            if delta_target == "w1_input":
                # dim → dim
                factor_size = 2 * dim * rank
            elif delta_target == "w1_output":
                # dim → ff_dim
                factor_size = (dim + self.ff_dim) * rank
            elif delta_target == "w2_input":
                # ff_dim → ff_dim
                factor_size = 2 * self.ff_dim * rank
            elif delta_target == "w2_output":
                # ff_dim → dim
                factor_size = (self.ff_dim + dim) * rank
            elif delta_target == "full":
                # w1: dim → ff_dim, w2: ff_dim → dim
                factor_size = (dim + self.ff_dim) * rank + (self.ff_dim + dim) * rank
            else:
                raise ValueError(f"Unknown delta_target: {delta_target}")
            
            self.hyper_to_factors = nn.Linear(bottleneck_dim, factor_size, bias=False)
            nn.init.normal_(self.hyper_to_factors.weight, std=0.01)

        
        elif factor_type == "multiplicative":
            # Multiplicative rank-1: generate row (scales input) and col (scales output)
            if delta_target == "w1_input":
                # dim → dim: row and col both have size dim
                mult_size = 2 * dim
            elif delta_target == "w1_output":
                # dim → ff_dim: row is dim, col is ff_dim
                mult_size = dim + self.ff_dim
            elif delta_target == "w2_input":
                # ff_dim → ff_dim: row and col both have size ff_dim
                mult_size = 2 * self.ff_dim
            elif delta_target == "w2_output":
                # ff_dim → dim: row is ff_dim, col is dim
                mult_size = self.ff_dim + dim
            elif delta_target == "full":
                # w1: dim → ff_dim, w2: ff_dim → dim
                mult_size = (dim + self.ff_dim) + (self.ff_dim + dim)
            else:
                raise ValueError(f"Unknown delta_target: {delta_target}")
            
            self.hyper_to_mult = nn.Linear(bottleneck_dim, mult_size, bias=True)
            nn.init.normal_(self.hyper_to_mult.weight, std=0.01)
            nn.init.ones_(self.hyper_to_mult.bias)
        
        elif factor_type == "circulant":
            # Circulant: generate one vector that defines the circulant matrix
            # Applied via FFT: y = ifft(fft(c) * fft(x))
            if delta_target == "w1_input":
                circ_size = dim
            elif delta_target == "w1_output":
                # For rectangular, we apply circulant to input (dim), then project
                circ_size = dim
            elif delta_target == "w2_input":
                circ_size = self.ff_dim
            elif delta_target == "w2_output":
                circ_size = self.ff_dim
            elif delta_target == "full":
                circ_size = dim + self.ff_dim  # one for each stage
            else:
                raise ValueError(f"Unknown delta_target: {delta_target}")
            
            self.hyper_to_circ = nn.Linear(bottleneck_dim, circ_size, bias=True)
            nn.init.normal_(self.hyper_to_circ.weight, std=0.01)
            nn.init.zeros_(self.hyper_to_circ.bias)
        
        elif factor_type == "basis":
            # Basis: generate K coefficients, apply weighted sum of frozen basis matrices
            # rank parameter is repurposed as n_bases
            self.n_bases = rank
            
            # Determine transform dimensions based on delta_target
            if delta_target == "w1_input":
                in_d, out_d = dim, dim
            elif delta_target == "w1_output":
                in_d, out_d = dim, self.ff_dim
            elif delta_target == "w2_input":
                in_d, out_d = self.ff_dim, self.ff_dim
            elif delta_target == "w2_output":
                in_d, out_d = self.ff_dim, dim
            elif delta_target == "full":
                # Two sets of bases
                in_d, out_d = dim, self.ff_dim  # for w1
                in_d2, out_d2 = self.ff_dim, dim  # for w2
            else:
                raise ValueError(f"Unknown delta_target: {delta_target}")
            
            # Create frozen random low-rank basis matrices for efficiency
            # Each basis is a rank-1 matrix: B_k = u_k @ v_k^T
            if delta_target != "full":
                # (n_bases, in_d) and (n_bases, out_d)
                self.register_buffer('basis_u', torch.randn(self.n_bases, in_d) / math.sqrt(in_d))
                self.register_buffer('basis_v', torch.randn(self.n_bases, out_d) / math.sqrt(out_d))
            else:
                self.register_buffer('basis_u1', torch.randn(self.n_bases, in_d) / math.sqrt(in_d))
                self.register_buffer('basis_v1', torch.randn(self.n_bases, out_d) / math.sqrt(out_d))
                self.register_buffer('basis_u2', torch.randn(self.n_bases, in_d2) / math.sqrt(in_d2))
                self.register_buffer('basis_v2', torch.randn(self.n_bases, out_d2) / math.sqrt(out_d2))
            
            # Generate just n_bases coefficients (or 2*n_bases for full)
            coeff_size = self.n_bases if delta_target != "full" else 2 * self.n_bases
            self.hyper_to_coeffs = nn.Linear(bottleneck_dim, coeff_size, bias=True)
            nn.init.normal_(self.hyper_to_coeffs.weight, std=0.01)
            nn.init.zeros_(self.hyper_to_coeffs.bias)
        
        elif factor_type == "fastfood":
            # Fastfood: W ≈ S H G Π H B where S,G,B are diagonals, H is Hadamard, Π is permutation
            # Only works for square transforms
            if delta_target not in ("w1_input", "w2_input"):
                raise ValueError(f"Fastfood only supports square transforms (w1_input, w2_input), got {delta_target}")
            
            transform_dim = dim if delta_target == "w1_input" else self.ff_dim
            
            # Check power of 2 for Hadamard
            assert transform_dim > 0 and (transform_dim & (transform_dim - 1)) == 0, \
                f"Fastfood requires power-of-2 dim, got {transform_dim}"
            
            self.fastfood_dim = transform_dim
            
            # Fixed random permutation
            self.register_buffer('fastfood_perm', torch.randperm(transform_dim))
            
            # Fixed random signs for B (binary ±1)
            self.register_buffer('fastfood_B_signs', torch.randint(0, 2, (transform_dim,)) * 2 - 1)
            
            # Generate 3 diagonal vectors: S, G, B_scale (we'll multiply B_signs by B_scale)
            # Total: 3 * transform_dim
            self.hyper_to_fastfood = nn.Linear(bottleneck_dim, 3 * transform_dim, bias=True)
            nn.init.normal_(self.hyper_to_fastfood.weight, std=0.01)
            # Initialize to produce identity-ish transform: S=1, G=1, B_scale=1
            nn.init.ones_(self.hyper_to_fastfood.bias)

        self.delta_scale = nn.Parameter(torch.zeros(1))
        self.act = act_fn()

    def _apply_w1(self, x: torch.Tensor) -> torch.Tensor:
        """Apply W1 based on base_weight_type."""
        if self.base_weight_type == "regular":
            return self.w1(x)
        elif self.base_weight_type == "lowrank":
            # W1 = w1_u @ w1_v.T, so x @ W1 = x @ w1_u @ w1_v.T
            return torch.matmul(torch.matmul(x, self.w1_u), self.w1_v.T)
        else:  # none
            return torch.zeros(x.shape[0], x.shape[1], self.ff_dim, device=x.device, dtype=x.dtype)

    def _apply_w2(self, h: torch.Tensor) -> torch.Tensor:
        """Apply W2 based on base_weight_type."""
        if self.base_weight_type == "regular":
            return self.w2(h)
        elif self.base_weight_type == "lowrank":
            # W2 = w2_u @ w2_v.T, so h @ W2 = h @ w2_u @ w2_v.T
            return torch.matmul(torch.matmul(h, self.w2_u), self.w2_v.T)
        else:  # none
            return torch.zeros(h.shape[0], h.shape[1], self.dim, device=h.device, dtype=h.dtype)

    def _compute_lowrank_delta(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute low-rank delta: scale * (x @ U @ V^T)"""
        B, S = x.shape[:2]
        if self.use_bmm:
            x_flat = x.reshape(B * S, 1, -1)
            u_flat = u.reshape(B * S, u.shape[-2], self.rank)
            v_flat = v.reshape(B * S, v.shape[-2], self.rank)
            x_low = torch.bmm(x_flat, u_flat).squeeze(1)
            x_delta = torch.bmm(x_low.unsqueeze(1), v_flat.transpose(1, 2)).squeeze(1)
            x_delta = x_delta.view(B, S, -1)
        else:
            x_low = torch.einsum('bsi,bsir->bsr', x, u)
            x_delta = torch.einsum('bsr,bsor->bso', x_low, v)
        return self.delta_scale * x_delta

    def _apply_multiplicative_w1(self, x: torch.Tensor, row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        """
        Apply multiplicative rank-1 modulation to W1: col * ((x * row) @ W1)
        
        x: (B, S, dim)
        row: (B, S, dim) - scales input features
        col: (B, S, ff_dim) - scales output features
        Returns: (B, S, ff_dim)
        """
        base_out = self._apply_w1(x)
        modulated = self._apply_w1(x * row) * col
        return base_out + self.delta_scale * (modulated - base_out)

    def _apply_multiplicative_w2(self, h: torch.Tensor, row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        """
        Apply multiplicative rank-1 modulation to W2: col * ((h * row) @ W2)
        
        h: (B, S, ff_dim)
        row: (B, S, ff_dim) - scales input features
        col: (B, S, dim) - scales output features
        Returns: (B, S, dim)
        """
        base_out = self._apply_w2(h)
        modulated = self._apply_w2(h * row) * col
        return base_out + self.delta_scale * (modulated - base_out)

    def _apply_circulant(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Apply circulant convolution via FFT: y = ifft(fft(c) * fft(x))
        
        x: (B, S, d)
        c: (B, S, d) - circulant-defining vector
        Returns: (B, S, d)
        """
        # FFT-based circular convolution
        x_fft = torch.fft.rfft(x.float(), dim=-1)
        c_fft = torch.fft.rfft(c.float(), dim=-1)
        y_fft = x_fft * c_fft
        y = torch.fft.irfft(y_fft, n=x.shape[-1], dim=-1)
        return self.delta_scale * y.to(x.dtype)

    def _apply_basis(self, x: torch.Tensor, coeffs: torch.Tensor, 
                     basis_u: torch.Tensor, basis_v: torch.Tensor) -> torch.Tensor:
        """
        Apply basis-weighted transform: δ(x) = Σ_k α_k * (x @ u_k) * v_k
        
        x: (B, S, in_d)
        coeffs: (B, S, n_bases)
        basis_u: (n_bases, in_d)
        basis_v: (n_bases, out_d)
        Returns: (B, S, out_d)
        """
        # x @ u_k for all k: (B, S, in_d) @ (n_bases, in_d).T -> (B, S, n_bases)
        x_proj = torch.matmul(x, basis_u.T)  # (B, S, n_bases)
        # Weight by coefficients: (B, S, n_bases) * (B, S, n_bases) -> (B, S, n_bases)
        weighted = x_proj * coeffs
        # Project to output: (B, S, n_bases) @ (n_bases, out_d) -> (B, S, out_d)
        delta = torch.matmul(weighted, basis_v)
        return self.delta_scale * delta

    def _apply_fastfood(self, x: torch.Tensor, S_diag: torch.Tensor, 
                        G_diag: torch.Tensor, B_scale: torch.Tensor) -> torch.Tensor:
        """
        Apply Fastfood transform: y = S H G Π H B x
        
        x: (B, S, d)
        S_diag, G_diag, B_scale: (B, S, d) - diagonal scaling factors
        Returns: (B, S, d)
        """
        d = x.shape[-1]
        
        # B: multiply by random signs scaled by B_scale
        # B_signs is (d,), B_scale is (B, S, d)
        y = x * (self.fastfood_B_signs.float() * B_scale)
        
        # First H: Hadamard transform (normalized)
        y = _hadamard_transform(y.float())
        
        # Π: fixed permutation
        y = y[..., self.fastfood_perm]
        
        # G: diagonal Gaussian-like scaling
        y = y * G_diag
        
        # Second H: Hadamard transform
        y = _hadamard_transform(y)
        
        # S: final diagonal scaling
        y = y * S_diag
        
        return self.delta_scale * y.to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape

        # Generate factors from input
        bottleneck = self.hyper_act(self.hyper_down(x))  # (B, S, bottleneck)
        
        if self.factor_type == "lowrank":
            factors = self.hyper_to_factors(bottleneck)
            
            if self.delta_target == "w1_input":
                # dim → dim delta on input
                u, v = factors.chunk(2, dim=-1)
                u = u.view(B, S, D, self.rank)
                v = v.view(B, S, D, self.rank)
                delta = self._compute_lowrank_delta(x, u, v)
                x_eff = x + delta
                h = self.act(self._apply_w1(x_eff))
                return self._apply_w2(h)
            
            elif self.delta_target == "w1_output":
                # dim → ff_dim delta
                u_size = D * self.rank
                v_size = self.ff_dim * self.rank
                u, v = factors.split([u_size, v_size], dim=-1)
                u = u.view(B, S, D, self.rank)
                v = v.view(B, S, self.ff_dim, self.rank)
                delta = self._compute_lowrank_delta(x, u, v)
                h = self.act(self._apply_w1(x) + delta)
                return self._apply_w2(h)
            
            elif self.delta_target == "w2_input":
                # ff_dim → ff_dim delta
                h = self.act(self._apply_w1(x))
                u, v = factors.chunk(2, dim=-1)
                u = u.view(B, S, self.ff_dim, self.rank)
                v = v.view(B, S, self.ff_dim, self.rank)
                delta = self._compute_lowrank_delta(h, u, v)
                return self._apply_w2(h + delta)
            
            elif self.delta_target == "w2_output":
                # ff_dim → dim delta
                h = self.act(self._apply_w1(x))
                u_size = self.ff_dim * self.rank
                v_size = D * self.rank
                u, v = factors.split([u_size, v_size], dim=-1)
                u = u.view(B, S, self.ff_dim, self.rank)
                v = v.view(B, S, D, self.rank)
                delta = self._compute_lowrank_delta(h, u, v)
                return self._apply_w2(h) + delta
            
            elif self.delta_target == "full":
                # w1: dim → ff_dim, w2: ff_dim → dim
                u1_size = D * self.rank
                v1_size = self.ff_dim * self.rank
                u2_size = self.ff_dim * self.rank
                v2_size = D * self.rank
                u1, v1, u2, v2 = factors.split([u1_size, v1_size, u2_size, v2_size], dim=-1)
                
                u1 = u1.view(B, S, D, self.rank)
                v1 = v1.view(B, S, self.ff_dim, self.rank)
                u2 = u2.view(B, S, self.ff_dim, self.rank)
                v2 = v2.view(B, S, D, self.rank)
                
                delta_w1 = self._compute_lowrank_delta(x, u1, v1)
                h = self.act(self._apply_w1(x) + delta_w1)
                delta_w2 = self._compute_lowrank_delta(h, u2, v2)
                return self._apply_w2(h) + delta_w2
        
        elif self.factor_type == "multiplicative":
            # Multiplicative rank-1 path
            mult_params = self.hyper_to_mult(bottleneck)
            
            if self.delta_target == "w1_input":
                # dim → dim: row and col both have size dim
                row, col = mult_params.chunk(2, dim=-1)
                h = self.act(self._apply_multiplicative_w1(x, row, col))
                return self._apply_w2(h)
            
            elif self.delta_target == "w1_output":
                # dim → ff_dim: row is dim, col is ff_dim
                row, col = mult_params.split([D, self.ff_dim], dim=-1)
                h = self.act(self._apply_multiplicative_w1(x, row, col))
                return self._apply_w2(h)
            
            elif self.delta_target == "w2_input":
                # ff_dim → ff_dim: row and col both have size ff_dim
                h = self.act(self._apply_w1(x))
                row, col = mult_params.chunk(2, dim=-1)
                h_mod = h + self.delta_scale * ((h * row) * col - h)
                return self._apply_w2(h_mod)
            
            elif self.delta_target == "w2_output":
                # ff_dim → dim: row is ff_dim, col is dim
                h = self.act(self._apply_w1(x))
                row, col = mult_params.split([self.ff_dim, D], dim=-1)
                return self._apply_multiplicative_w2(h, row, col)
            
            elif self.delta_target == "full":
                # w1: dim → ff_dim, w2: ff_dim → dim
                row_w1, col_w1, row_w2, col_w2 = mult_params.split(
                    [D, self.ff_dim, self.ff_dim, D], dim=-1)
                
                h = self.act(self._apply_multiplicative_w1(x, row_w1, col_w1))
                return self._apply_multiplicative_w2(h, row_w2, col_w2)
        
        elif self.factor_type == "circulant":
            # Circulant convolution path
            circ_params = self.hyper_to_circ(bottleneck)
            
            if self.delta_target == "w1_input":
                # Circulant on input (dim → dim)
                delta = self._apply_circulant(x, circ_params)
                h = self.act(self._apply_w1(x + delta))
                return self._apply_w2(h)
            
            elif self.delta_target == "w1_output":
                # Circulant on input, then project (dim → ff_dim via circulant + W1)
                delta = self._apply_circulant(x, circ_params)
                h = self.act(self._apply_w1(x) + self._apply_w1(delta))
                return self._apply_w2(h)
            
            elif self.delta_target == "w2_input":
                # Circulant on hidden (ff_dim → ff_dim)
                h = self.act(self._apply_w1(x))
                delta = self._apply_circulant(h, circ_params)
                return self._apply_w2(h + delta)
            
            elif self.delta_target == "w2_output":
                # Circulant on hidden, then project (ff_dim → dim via circulant + W2)
                h = self.act(self._apply_w1(x))
                delta = self._apply_circulant(h, circ_params)
                return self._apply_w2(h) + self._apply_w2(delta)
            
            elif self.delta_target == "full":
                # Two circulants: one for input, one for hidden
                c1, c2 = circ_params.split([D, self.ff_dim], dim=-1)
                delta1 = self._apply_circulant(x, c1)
                h = self.act(self._apply_w1(x) + self._apply_w1(delta1))
                delta2 = self._apply_circulant(h, c2)
                return self._apply_w2(h) + self._apply_w2(delta2)
        
        elif self.factor_type == "basis":
            # Basis coefficients path
            coeffs = self.hyper_to_coeffs(bottleneck)
            
            if self.delta_target == "w1_input":
                delta = self._apply_basis(x, coeffs, self.basis_u, self.basis_v)
                h = self.act(self._apply_w1(x + delta))
                return self._apply_w2(h)
            
            elif self.delta_target == "w1_output":
                delta = self._apply_basis(x, coeffs, self.basis_u, self.basis_v)
                h = self.act(self._apply_w1(x) + delta)
                return self._apply_w2(h)
            
            elif self.delta_target == "w2_input":
                h = self.act(self._apply_w1(x))
                delta = self._apply_basis(h, coeffs, self.basis_u, self.basis_v)
                return self._apply_w2(h + delta)
            
            elif self.delta_target == "w2_output":
                h = self.act(self._apply_w1(x))
                delta = self._apply_basis(h, coeffs, self.basis_u, self.basis_v)
                return self._apply_w2(h) + delta
            
            elif self.delta_target == "full":
                coeffs1, coeffs2 = coeffs.split([self.n_bases, self.n_bases], dim=-1)
                delta1 = self._apply_basis(x, coeffs1, self.basis_u1, self.basis_v1)
                h = self.act(self._apply_w1(x) + delta1)
                delta2 = self._apply_basis(h, coeffs2, self.basis_u2, self.basis_v2)
                return self._apply_w2(h) + delta2
        
        elif self.factor_type == "fastfood":
            # Fastfood path (only square transforms)
            ff_params = self.hyper_to_fastfood(bottleneck)
            d = self.fastfood_dim
            S_diag, G_diag, B_scale = ff_params.split([d, d, d], dim=-1)
            
            if self.delta_target == "w1_input":
                delta = self._apply_fastfood(x, S_diag, G_diag, B_scale)
                h = self.act(self._apply_w1(x + delta))
                return self._apply_w2(h)
            
            elif self.delta_target == "w2_input":
                h = self.act(self._apply_w1(x))
                delta = self._apply_fastfood(h, S_diag, G_diag, B_scale)
                return self._apply_w2(h + delta)


# ---------------------------------------------------------------------------
# Experiment 4: Spectral Modulation Layer
# ---------------------------------------------------------------------------

def _dct_type2(x: torch.Tensor) -> torch.Tensor:
    """DCT-II along last dimension. Input and output are real.
    
    Uses the FFT-based algorithm for O(n log n) complexity.
    """
    N = x.shape[-1]
    # Reorder: [x0, x1, ..., x_{N-1}] -> [x0, x2, x4, ..., x_{N-1}, ..., x3, x1]
    v = torch.cat([x[..., ::2], x[..., 1::2].flip(-1)], dim=-1)
    # FFT
    V = torch.fft.fft(v)
    # Multiply by phase factors: exp(-i * pi * k / (2N))
    k = torch.arange(N, device=x.device, dtype=torch.float32)
    phase = torch.exp(-1j * math.pi * k / (2 * N))
    return (V * phase).real * 2


def _idct_type2(X: torch.Tensor) -> torch.Tensor:
    """Inverse DCT-II (DCT-III) along last dimension. Input and output are real.
    
    Uses the FFT-based algorithm for O(n log n) complexity.
    """
    N = X.shape[-1]
    # Scale: divide by 2, with first element divided by 4
    scale = torch.ones(N, device=X.device, dtype=X.dtype) * 0.5
    scale[0] = 0.25
    X_scaled = X * scale
    
    # Multiply by conjugate phase factors: exp(i * pi * k / (2N))
    k = torch.arange(N, device=X.device, dtype=torch.float32)
    phase = torch.exp(1j * math.pi * k / (2 * N))
    V = X_scaled.to(torch.complex64) * phase
    
    # IFFT
    v = torch.fft.ifft(V).real * N
    
    # Reorder back: interleave even indices from first half, odd from reversed second half
    half = (N + 1) // 2
    even_part = v[..., :half]
    odd_part = v[..., half:].flip(-1)
    
    # Interleave: result[0,2,4,...] = even_part, result[1,3,5,...] = odd_part
    result = torch.empty_like(v)
    result[..., ::2] = even_part
    result[..., 1::2] = odd_part
    return result


def _hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard transform along last dimension (vectorized).
    
    Requires dim to be a power of 2. Input and output are real.
    This is its own inverse (up to scaling).
    
    Uses the standard butterfly structure with tensor reshaping for efficiency.
    """
    *batch_dims, N = x.shape
    assert N > 0 and (N & (N - 1)) == 0, f"Hadamard requires power-of-2 dim, got {N}"
    
    # Flatten batch dims for easier manipulation
    x = x.reshape(-1, N)
    B = x.shape[0]
    
    # Butterfly rounds: log2(N) stages
    h = 1
    while h < N:
        # Reshape to expose pairs: (B, num_blocks, 2, block_size)
        # where num_blocks = N // (2*h), block_size = h
        x = x.reshape(B, N // (2 * h), 2, h)
        
        # Butterfly operation: (a, b) -> (a+b, a-b)
        a = x[:, :, 0, :]  # (B, num_blocks, h)
        b = x[:, :, 1, :]  # (B, num_blocks, h)
        x = torch.stack([a + b, a - b], dim=2)  # (B, num_blocks, 2, h)
        
        # Flatten back
        x = x.reshape(B, N)
        h *= 2
    
    # Restore batch dims and normalize
    return x.reshape(*batch_dims, N) / math.sqrt(N)


class SpectralModulation(nn.Module):
    """
    Lightweight layer that modulates features in the frequency/transform domain.
    Each token generates a mask, features are transformed, modulated, and transformed back.
    
    Transform options:
        "fft":      Real FFT - magnitude scaling only (dim//2+1 mask values)
                    Complex internally but scales both real/imag equally.
                    Pros: Well-understood frequency semantics
                    Cons: Complex ops, bf16 issues, only magnitude (not phase)
        
        "fft_complex": Real FFT with both magnitude AND phase control
                    Generates 2x params: magnitude scale + phase shift
                    z' = z * mag_scale * exp(i * phase_shift)
                    Pros: Full control over frequency domain
                    Cons: 2x params, more complex gradients
        
        "dct":      Discrete Cosine Transform (Type-II)
                    Purely real-valued, full dim mask values.
                    Pros: No complex numbers, full transform space
                    Cons: Slightly slower than FFT
        
        "hadamard": Walsh-Hadamard Transform
                    Purely real-valued, full dim mask values.
                    Pros: Very fast, no complex numbers
                    Cons: Requires power-of-2 dim, different frequency semantics
    
    Modulation modes:
        Multiplicative (always on): x_freq * mask
            - Scales existing frequency content
            - mask = 1 + scale * act(raw), starts at identity
        
        Additive (optional): x_freq * mask + additive
            - Injects new frequency content independent of input
            - For FFT: real additive broadcast to complex (like magnitude injection)
            - For DCT/Hadamard: adds to coefficients (shifts representation)
    
    Args:
        dim: model dimension
        mask_bottleneck: bottleneck dim for mask generation (None = direct)
        transform: "fft", "dct", or "hadamard"
        mask_act: activation for mask values ("tanh", "silu", "none", "softsign")
        use_additive: whether to include additive frequency injection branch
    """

    def __init__(
        self,
        dim: int,
        mask_bottleneck: Optional[int] = None,
        transform: Literal["fft", "fft_complex", "dct", "hadamard"] = "fft",
        mask_act: Literal["tanh", "silu", "none", "softsign"] = "none",
        use_additive: bool = False,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.transform = transform
        self.mask_act = mask_act
        self.use_additive = use_additive

        self.w_up = nn.Linear(dim, dim * ff_mult, bias=True)
        # zero bias
        nn.init.zeros_(self.w_up.bias)
        self.w_down = nn.Linear(dim * ff_mult, dim, bias=True)
        # zero bias
        nn.init.zeros_(self.w_down.bias)
        self.act = nn.SiLU()
        
        # Determine mask output dimension based on transform
        if transform == "fft":
            # FFT of real dim-length signal gives dim//2+1 complex values
            # We use a real mask that scales magnitude (not phase)
            self.freq_dim = dim // 2 + 1
        elif transform == "fft_complex":
            # FFT with both magnitude and phase control
            # freq_dim is the FFT output size, but we generate 2x for mag + phase
            self.freq_dim = dim // 2 + 1
        elif transform == "dct":
            # DCT is real-to-real, full dimension
            self.freq_dim = dim
        elif transform == "hadamard":
            # Hadamard requires power of 2
            assert dim > 0 and (dim & (dim - 1)) == 0, \
                f"Hadamard transform requires power-of-2 dim, got {dim}"
            self.freq_dim = dim
        else:
            raise ValueError(f"Unknown transform: {transform}")

        # Multiplicative mask generator
        if mask_bottleneck is not None:
            self.mask_gen = nn.Sequential(
                nn.Linear(dim, mask_bottleneck, bias=True),
                nn.GELU(),
                nn.Linear(mask_bottleneck, self.freq_dim, bias=True),
            )
            # set bias 0
            nn.init.zeros_(self.mask_gen[0].bias)
            nn.init.zeros_(self.mask_gen[2].bias)
        else:
            self.mask_gen = nn.Linear(dim, self.freq_dim, bias=True)
            # set bias 0
            nn.init.zeros_(self.mask_gen.bias)

        # mask = 1 + scale * act(raw), starts at 1 (identity) when scale=0
        self.mask_scale = nn.Parameter(torch.zeros(self.freq_dim))
        
        # Phase generator for fft_complex mode
        if transform == "fft_complex":
            if mask_bottleneck is not None:
                self.phase_gen = nn.Sequential(
                    nn.Linear(dim, mask_bottleneck, bias=True),
                    nn.GELU(),
                    nn.Linear(mask_bottleneck, self.freq_dim, bias=True),
                )
                nn.init.zeros_(self.phase_gen[0].bias)
                nn.init.zeros_(self.phase_gen[2].bias)
            else:
                self.phase_gen = nn.Linear(dim, self.freq_dim, bias=True)
                nn.init.zeros_(self.phase_gen.bias)
            # phase = phase_scale * act(raw), starts at 0 (no phase shift) when scale=0
            self.phase_scale = nn.Parameter(torch.zeros(self.freq_dim))
        
        # Optional additive branch
        if use_additive:
            if mask_bottleneck is not None:
                self.additive_gen = nn.Sequential(
                    nn.Linear(dim, mask_bottleneck, bias=True),
                    nn.GELU(),
                    nn.Linear(mask_bottleneck, self.freq_dim, bias=True),
                )
                # set bias 0
                nn.init.zeros_(self.additive_gen[0].bias)
                nn.init.zeros_(self.additive_gen[2].bias)
            else:
                self.additive_gen = nn.Linear(dim, self.freq_dim, bias=True)
            # additive = additive_scale * raw, starts at 0 (no injection)
            self.additive_scale = nn.Parameter(torch.zeros(self.freq_dim))
            # set bias 0
            nn.init.zeros_(self.additive_gen.bias)
    
    def _apply_mask_act(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the chosen activation to raw mask values."""
        if self.mask_act == "tanh":
            return torch.tanh(x)
        elif self.mask_act == "silu":
            return torch.nn.functional.silu(x)
        elif self.mask_act == "none":
            return x
        elif self.mask_act == "softsign":
            return x / (1 + x.abs())
        else:
            raise ValueError(f"Unknown mask_act: {self.mask_act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, S, dim)
        Returns:
            (B, S, dim) 
        """
        B, S, D = x.shape

        #mlp ou

        # Generate multiplicative mask: 1 + scale * act(raw)
        # At init (scale=0): mask=1 everywhere -> identity transform
        mask_raw = self.mask_gen(x)  # (B, S, freq_dim)
        mask = 1.0 + self.mask_scale * self._apply_mask_act(mask_raw)
        
        # Generate additive term if enabled
        additive = None
        if self.use_additive:
            additive_raw = self.additive_gen(x)  # (B, S, freq_dim)
            additive = self.additive_scale * additive_raw

        orig_dtype = x.dtype
        
        if self.transform == "fft":
            # FFT path (complex internally, bf16 needs float32 cast)
            x_freq = torch.fft.rfft(x.float(), dim=-1)  # (B, S, freq_dim) complex
            x_freq = x_freq * mask.to(x_freq.dtype)     # real mask scales magnitude
            if additive is not None:
                # Real additive broadcasts to complex (adds to both real and imag)
                x_freq = x_freq + additive.to(x_freq.dtype)
            out = torch.fft.irfft(x_freq, n=D, dim=-1).to(orig_dtype)
        
        elif self.transform == "fft_complex":
            # FFT path with both magnitude AND phase control
            # z' = z * mag_scale * exp(i * phase_shift)
            x_freq = torch.fft.rfft(x.float(), dim=-1)  # (B, S, freq_dim) complex
            
            # Generate phase shift: phase_scale * act(raw), starts at 0
            phase_raw = self.phase_gen(x)  # (B, S, freq_dim)
            phase_shift = self.phase_scale * self._apply_mask_act(phase_raw)
            
            # Apply magnitude scaling and phase rotation
            # exp(i * phase) = cos(phase) + i * sin(phase)
            phase_factor = torch.complex(torch.cos(phase_shift), torch.sin(phase_shift))
            x_freq = x_freq * mask.to(x_freq.dtype) * phase_factor
            
            if additive is not None:
                x_freq = x_freq + additive.to(x_freq.dtype)
            out = torch.fft.irfft(x_freq, n=D, dim=-1).to(orig_dtype)
        
        elif self.transform == "dct":
            # DCT path (purely real)
            x_dct = _dct_type2(x.float())               # (B, S, dim) real
            x_dct = x_dct * mask
            if additive is not None:
                x_dct = x_dct + additive
            out = _idct_type2(x_dct).to(orig_dtype)
        
        elif self.transform == "hadamard":
            # Hadamard path (purely real, self-inverse)
            x_h = _hadamard_transform(x.float())        # (B, S, dim) real
            x_h = x_h * mask
            if additive is not None:
                x_h = x_h + additive
            out = _hadamard_transform(x_h).to(orig_dtype) # inverse = same transform

        out = self.w_down(self.act(self.w_up(out)))

        return out


# ---------------------------------------------------------------------------
# Experiment 7: Composite Dynamic FFN (combines structures)
# ---------------------------------------------------------------------------

class CompositeDynamicFFN(nn.Module):
    """
    FFN that combines multiple dynamic transform structures.
    
    Architecture:
        1. Input generates spectral mask + Kronecker delta + diagonal scale
        2. Apply spectral modulation (frequency mixing)
        3. Apply Kronecker delta (structured linear correction)
        4. Apply diagonal gate (per-feature scaling)
        5. Standard FFN projection
    
    Each component handles a different "type" of adaptation:
    - Spectral: global frequency-domain structure
    - Kronecker: factored linear mixing
    - Diagonal: per-feature importance weighting
    
    Args:
        dim: model dimension
        ff_mult: FFN expansion factor
        p, q: Kronecker factor sizes
        spectral_bottleneck: bottleneck for spectral mask generation
    """

    def __init__(
        self,
        dim: int,
        ff_mult: int = 4,
        p: int = 64,
        q: int = 64,
        spectral_bottleneck: int = 256,
    ):
        super().__init__()
        assert p * q == dim
        self.p, self.q = p, q
        self.ff_dim = dim * ff_mult
        self.freq_dim = dim // 2 + 1

        # Standard FFN
        self.w1 = nn.Linear(dim, self.ff_dim, bias=False)
        self.w2 = nn.Linear(self.ff_dim, dim, bias=False)
        self.act = nn.SiLU()

        # --- Dynamic component generators (shared bottleneck) ---
        self.hyper_bottleneck = nn.Sequential(
            nn.Linear(dim, spectral_bottleneck, bias=False),
            nn.GELU(),
        )
        
        # Spectral mask
        self.to_spectral = nn.Linear(spectral_bottleneck, self.freq_dim, bias=False)
        nn.init.zeros_(self.to_spectral.weight)
        
        # Kronecker factors
        self.to_kron = nn.Linear(spectral_bottleneck, p * p + q * q, bias=False)
        nn.init.normal_(self.to_kron.weight, std=0.01)
        
        # Diagonal gate
        self.to_diag = nn.Linear(spectral_bottleneck, dim, bias=False)
        nn.init.zeros_(self.to_diag.weight)

        # Per-component learnable scales (all start at zero for safe init)
        self.spectral_scale = nn.Parameter(torch.zeros(1))
        self.kron_scale = nn.Parameter(torch.zeros(1))
        self.diag_scale = nn.Parameter(torch.zeros(1))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        h = x

        # Shared bottleneck for all dynamic components
        hyper = self.hyper_bottleneck(h)  # (B, S, bottleneck)

        # 1. Spectral modulation (cast to float32 - FFT doesn't support bf16)
        orig_dtype = h.dtype
        spectral_mask = 1.0 + self.spectral_scale * torch.tanh(self.to_spectral(hyper))
        h_freq = torch.fft.rfft(h.float(), dim=-1)
        h_freq = h_freq * spectral_mask.to(h_freq.dtype)
        h = torch.fft.irfft(h_freq, n=D, dim=-1).to(orig_dtype)

        # 2. Kronecker delta
        kron_params = self.to_kron(hyper)
        A_flat, B_flat = kron_params.split([self.p * self.p, self.q * self.q], dim=-1)
        A = A_flat.view(B * S, self.p, self.p)
        B_mat = B_flat.view(B * S, self.q, self.q)
        h_flat = h.reshape(B * S, D)
        h_kron = _kron_apply(A, B_mat, h_flat, self.p, self.q)
        h = h_flat + self.kron_scale * h_kron
        h = h.view(B, S, D)

        # 3. Diagonal gate
        diag = 1.0 + self.diag_scale * torch.tanh(self.to_diag(hyper))
        h = h * diag

        # 4. Standard FFN
        h = self.w1(h)
        h = self.act(h)
        return self.w2(h)


# ---------------------------------------------------------------------------
# Quick test / parameter counting
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    B, S, D = 2, 128, 4096
    p, q = 64, 64
    x = torch.randn(B, S, D)

    print("=" * 70)
    print(f"Input shape: ({B}, {S}, {D})")
    print("=" * 70)

    modules = {
        # LowRankHyperGate (SwiGLU) - low-rank factor type
        "LowRankGate(r=16,gate_input)": LowRankHyperGate(D, ff_mult=4, rank=16, delta_target="gate_input"),
        "LowRankGate(r=16,both_input)": LowRankHyperGate(D, ff_mult=4, rank=16, delta_target="both_input"),
        "LowRankGate(r=16,gate_output)": LowRankHyperGate(D, ff_mult=4, rank=16, delta_target="gate_output"),
        "LowRankGate(r=16,both_output)": LowRankHyperGate(D, ff_mult=4, rank=16, delta_target="both_output"),
        "LowRankGate(r=16,w2_output)": LowRankHyperGate(D, ff_mult=4, rank=16, delta_target="w2_output"),
        "LowRankGate(r=16,full)": LowRankHyperGate(D, ff_mult=4, rank=16, delta_target="full"),
        # LowRankHyperFFN (plain FFN) - low-rank factor type
        "LowRankFFN(r=16,w1_input)": LowRankHyperFFN(D, ff_mult=4, rank=16, delta_target="w1_input"),
        "LowRankFFN(r=16,w1_output)": LowRankHyperFFN(D, ff_mult=4, rank=16, delta_target="w1_output"),
        "LowRankFFN(r=16,w2_input)": LowRankHyperFFN(D, ff_mult=4, rank=16, delta_target="w2_input"),
        "LowRankFFN(r=16,w2_output)": LowRankHyperFFN(D, ff_mult=4, rank=16, delta_target="w2_output"),
        "LowRankFFN(r=16,full)": LowRankHyperFFN(D, ff_mult=4, rank=16, delta_target="full"),
        # Other dynamic layers
        "SpectralModulation": SpectralModulation(D),
        "CompositeDynamicFFN": CompositeDynamicFFN(D, ff_mult=4, p=p, q=q),
    }

    # Reference: standard FFN parameter counts
    ff_dim = D * 4
    swiglu_params = D * ff_dim + D * ff_dim + ff_dim * D  # W1, V, W2
    plain_ffn_params = D * ff_dim + ff_dim * D  # W1, W2 (no gate)
    print(f"\nReference: SwiGLU FFN params = {swiglu_params:,}")
    print(f"Reference: Plain FFN params  = {plain_ffn_params:,}")
    print("-" * 70)

    for name, module in modules.items():
        n_params = sum(p.numel() for p in module.parameters())
        try:
            with torch.no_grad():
                out = module(x)
            status = f"output={tuple(out.shape)}"
        except Exception as e:
            status = f"ERROR: {e}"
        
        overhead = n_params / swiglu_params * 100
        print(f"{name:40s} | params={n_params:>12,} ({overhead:5.1f}% of SwiGLU) | {status}")
