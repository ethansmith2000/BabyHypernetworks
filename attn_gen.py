import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Literal
from utils import _make_bit_reversal_permutation, _make_stride_permutation, _dense_target_std, _scale_tensor_std

# ===========================================================================
# ATTENTION-GENERATED STRUCTURED TRANSFORMS
#
# Key insight: cross-attention decouples generator parameter count from the
# size of the generated transform. Parameters scale with attention dim (O(dim²)),
# not with the number of generated values. This lets us generate expressive
# transforms without massive projection matrices.
#
# All modules below use a common pattern:
# 1. Learned query seeds (small count) attend to input sequence
# 2. Attention output is reshaped into structured transform parameters
# 3. Structured transform is applied efficiently to input
# 4. Static base + dynamic delta for stable training
# ===========================================================================


class _AttnGenerator(nn.Module):
    """
    Shared cross-attention generation backbone.
    
    Learned query tokens attend to the input sequence to produce a 
    flat parameter vector. The number of queries and output projection
    determine how many values are generated.
    
    Parameter cost: O(dim²) regardless of how many values are generated.
    Compute cost: O(n_queries * S * dim) for generation.
    
    Args:
        dim: model dimension (key/value dimension)
        n_queries: number of query tokens
        out_dim: output dimension per query token
        heads: number of attention heads
        qk_norm: whether to use cosine attention with learned temperature
    """

    def __init__(
        self,
        dim: int,
        n_queries: int,
        out_dim: int,
        heads: int = 8,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.n_queries = n_queries
        self.qk_norm = qk_norm

        # Learned query seeds — these discover "what questions to ask"
        query_seed = torch.empty(1, n_queries, dim)
        nn.init.xavier_normal_(query_seed)
        self.query_seed = nn.Parameter(query_seed)

        # KV projection from input
        self.kv_norm = nn.RMSNorm(dim)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=False)

        # Output projection: attention space -> desired output dim
        self.to_out = nn.Linear(dim, out_dim, bias=False)
        # Initialize output small so generated params start near zero
        nn.init.normal_(self.to_out.weight, std=0.02)

        if qk_norm:
            # Cosine attention with learned temperature (sharper init)
            self.temp = nn.Parameter(torch.ones(1, heads, 1, 1) * 5.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, S, dim) — input sequence to condition on
        Returns:
            (B, n_queries, out_dim) — generated parameter tokens
        """
        B, S, D = x.shape
        nq = self.n_queries

        # Queries from learned seeds
        q = self.query_seed.expand(B, -1, -1)
        q = q.reshape(B, nq, self.heads, self.head_dim).transpose(1, 2)

        # Keys and values from input
        kv = self.kv_proj(self.kv_norm(x))
        k, v = kv.chunk(2, dim=-1)
        k = k.reshape(B, S, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, S, self.heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            q = F.normalize(q, p=2, dim=-1) * self.temp
            k = F.normalize(k, p=2, dim=-1)
            attn_out = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v)

        attn_out = attn_out.transpose(1, 2).reshape(B, nq, D)
        return self.to_out(attn_out)


# ---------------------------------------------------------------------------
# Experiment 7: Attention-Generated Kronecker FFN
# ---------------------------------------------------------------------------

class AttnGenKroneckerFFN(nn.Module):
    """
    Cross-attention generates Kronecker factors that perturb a static FFN.
    
    Combines the expressiveness of attention-based generation (parameters 
    independent of generated size) with efficient Kronecker application.
    
    Architecture:
        - Small set of learned queries attend to input sequence
        - Attention output reshaped into Kronecker factors A (p×p) and B (q×q)  
        - Applied as: x_eff = x + scale * kron(A,B) @ x, then standard FFN
    
    Generator params: O(dim²) — just the attention weights
    Generated values: p² + q² per sequence (e.g., 8192 for p=q=64)
    Application cost: O(S * dim * (p+q))
    
    This is a per-sequence transform (variant 2): one Kronecker generated per
    batch element, shared across all tokens. For causal models, use the
    per-token variant (AttnGenKroneckerFFN_PerToken) or ensure bidirectional
    context is acceptable.
    
    Args:
        dim: model dimension (must equal p * q)
        ff_mult: FFN expansion factor
        p, q: Kronecker factor sizes
        heads: attention heads for generator
        n_queries: number of query tokens (can be > p²+q² output values 
                   for richer intermediate representation)
    """

    def __init__(
        self,
        dim: int,
        ff_mult: int = 4,
        p: int = 64,
        q: int = 64,
        heads: int = 8,
        n_queries: Optional[int] = None,
    ):
        super().__init__()
        assert p * q == dim
        self.p, self.q = p, q
        self.ff_dim = dim * ff_mult
        kron_params = p * p + q * q

        # We use a small number of queries — each produces part of the factors
        # Default: one query per Kronecker factor row, so p + q queries
        n_queries = n_queries or (p + q)
        # Each query outputs enough to fill its portion
        # We'll just flatten everything and take what we need
        out_per_query = math.ceil(kron_params / n_queries)
        self.kron_params = kron_params
        self.total_out = n_queries * out_per_query

        self.generator = _AttnGenerator(dim, n_queries, out_per_query, heads)

        # Static FFN
        self.w1 = nn.Linear(dim, self.ff_dim, bias=False)
        self.w2 = nn.Linear(self.ff_dim, dim, bias=False)
        self.act = nn.SiLU()
        self.norm = nn.RMSNorm(dim)

        # Static Kronecker base (identity-like)
        self.base_A = nn.Parameter(torch.eye(p) + torch.randn(p, p) * 0.01)
        self.base_B = nn.Parameter(torch.eye(q) + torch.randn(q, q) * 0.01)

        self.delta_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        h = self.norm(x)

        # Generate Kronecker factors via cross-attention (one per sequence)
        gen_out = self.generator(h)               # (B, n_queries, out_per_query)
        gen_flat = gen_out.reshape(B, -1)[:, :self.kron_params]  # (B, p²+q²)

        A_flat, B_flat = gen_flat.split([self.p * self.p, self.q * self.q], dim=-1)
        A_delta = A_flat.view(B, self.p, self.p)
        B_delta = B_flat.view(B, self.q, self.q)

        # Static base + dynamic delta
        A = self.base_A.unsqueeze(0) + self.delta_scale * A_delta
        B_eff = self.base_B.unsqueeze(0) + self.delta_scale * B_delta

        # Apply same Kronecker to all tokens via einsum (per-sequence variant)
        h_reshaped = h.view(B, S, self.p, self.q)
        h_reshaped = torch.einsum('bij,bsjk->bsik', B_eff, h_reshaped)
        h_reshaped = torch.einsum('bsij,bkj->bsik', h_reshaped, A)
        h_kron = h_reshaped.reshape(B, S, D)

        # Residual Kronecker mixing then FFN
        h = h + h_kron
        h = self.w1(h)
        h = self.act(h)
        return self.w2(h)


# ---------------------------------------------------------------------------
# Experiment 8: Attention-Generated Butterfly FFN
# ---------------------------------------------------------------------------

class AttnGenButterflyFFN(nn.Module):
    """
    Cross-attention generates butterfly transform parameters.
    
    Multiple rounds of 2×2 block-diagonal transforms with fixed permutations,
    where the block parameters are generated via cross-attention to the input.
    
    This gives O(dim * rounds) application cost with data-dependent mixing.
    Generator parameters are O(dim²) independent of the number of rounds.
    
    Per-sequence variant: one butterfly transform per batch element.
    
    Args:
        dim: model dimension (must be even)
        ff_mult: FFN expansion factor
        n_rounds: number of butterfly rounds
        heads: attention heads for generator
        permutation: type of inter-round permutation
    """

    def __init__(
        self,
        dim: int,
        ff_mult: int = 4,
        n_rounds: int = 4,
        heads: int = 8,
        permutation: Literal["bit_reversal", "stride"] = "stride",
    ):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.n_rounds = n_rounds
        self.n_blocks = dim // 2
        self.ff_dim = dim * ff_mult

        # Total butterfly params: 4 per block × n_blocks × n_rounds
        self.params_per_round = self.n_blocks * 4
        total_butterfly_params = self.params_per_round * n_rounds

        # Attention generator: queries produce all butterfly params at once
        # Use enough queries to keep each query's output dimension reasonable
        n_queries = min(n_rounds * 4, 32)  # e.g., 16 queries for 4 rounds
        out_per_query = math.ceil(total_butterfly_params / n_queries)
        self.total_butterfly_params = total_butterfly_params
        self.total_out = n_queries * out_per_query

        self.generator = _AttnGenerator(dim, n_queries, out_per_query, heads)

        # Per-round bias toward identity: initialize 2x2 blocks as [[1,0],[0,1]]
        # These are added to the generated params so the transform starts as identity
        identity_block = torch.tensor([1., 0., 0., 1.]).repeat(self.n_blocks)
        self.register_buffer('identity_blocks', identity_block)  # (n_blocks * 4,)

        # Fixed permutations between rounds
        for i in range(n_rounds - 1):
            if permutation == "bit_reversal":
                perm = _make_bit_reversal_permutation(dim)
            elif permutation == "stride":
                stride = 2 ** (i + 1) if 2 ** (i + 1) < dim else 3
                perm = _make_stride_permutation(dim, stride)
            else:
                perm = torch.arange(dim)
            self.register_buffer(f'perm_{i}', perm)

        # Static FFN
        self.w1 = nn.Linear(dim, self.ff_dim, bias=False)
        self.w2 = nn.Linear(self.ff_dim, dim, bias=False)
        self.act = nn.SiLU()
        self.norm = nn.RMSNorm(dim)

        self.delta_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        h = self.norm(x)

        # Generate all butterfly parameters via cross-attention
        gen_out = self.generator(h)  # (B, n_queries, out_per_query)
        gen_flat = gen_out.reshape(B, -1)[:, :self.total_butterfly_params]

        # Split into per-round params
        round_params = gen_flat.view(B, self.n_rounds, self.params_per_round)

        # Apply butterfly rounds (same transform to all tokens)
        h_flat = h.reshape(B, S, D)  # keep B and S separate for broadcasting

        for i in range(self.n_rounds):
            # Per-round 2x2 blocks: add identity bias so starts as passthrough
            blocks_flat = self.identity_blocks + self.delta_scale * round_params[:, i, :]
            blocks = blocks_flat.view(B, self.n_blocks, 2, 2)  # (B, n_blocks, 2, 2)

            # Apply block-diagonal: pair adjacent dims
            h_paired = h_flat.view(B, S, self.n_blocks, 2)  # (B, S, n_blocks, 2)
            # Broadcast blocks (B, n_blocks, 2, 2) over S
            h_paired = torch.einsum('bnij,bsnj->bsni', blocks, h_paired)
            h_flat = h_paired.reshape(B, S, D)

            # Permute (except after last round)
            if i < self.n_rounds - 1:
                perm = getattr(self, f'perm_{i}')
                h_flat = h_flat[:, :, perm]

        # Residual then FFN
        h = h + h_flat
        h = self.w1(h)
        h = self.act(h)
        return self.w2(h)


# ---------------------------------------------------------------------------
# Experiment 9: Attention-Generated Spectral + FFN Delta
# ---------------------------------------------------------------------------

class AttnGenSpectralFFN(nn.Module):
    """
    Cross-attention generates both a spectral modulation mask and a low-rank
    FFN perturbation. Combines frequency-domain and spatial-domain adaptation
    in a single generation step.
    
    Architecture:
        - Queries attend to input, producing spectral mask + low-rank factors
        - Spectral mask modulates features in frequency domain  
        - Low-rank factors perturb the FFN up-projection
        - Single attention call generates both (shared computation)
    
    This demonstrates the efficiency of attention-based generation: one set
    of queries produces heterogeneous structured parameters for multiple
    transform types simultaneously.
    
    Args:
        dim: model dimension
        ff_mult: FFN expansion factor
        rank: rank of the FFN perturbation
        heads: attention heads for generator
        complex_mask: generate complex spectral mask (phase + magnitude)
    """

    def __init__(
        self,
        dim: int,
        ff_mult: int = 4,
        rank: int = 16,
        heads: int = 8,
        complex_mask: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.ff_dim = dim * ff_mult
        self.rank = rank
        self.freq_dim = dim // 2 + 1
        self.complex_mask = complex_mask

        # How many values we need to generate:
        # Spectral mask: freq_dim (or freq_dim*2 for complex)
        spectral_size = self.freq_dim * (2 if complex_mask else 1)
        # Low-rank FFN delta: two factors of dim × rank
        lora_size = dim * rank * 2
        total_gen = spectral_size + lora_size
        self.spectral_size = spectral_size
        self.lora_size = lora_size

        # Attention generator
        # Use modest query count — attention output gets projected to what we need
        n_queries = max(rank * 2, 32)
        out_per_query = math.ceil(total_gen / n_queries)
        self.total_gen = total_gen
        self.total_out = n_queries * out_per_query

        self.generator = _AttnGenerator(dim, n_queries, out_per_query, heads)

        # Static FFN
        self.w1 = nn.Linear(dim, self.ff_dim, bias=False)
        self.w2 = nn.Linear(self.ff_dim, dim, bias=False)
        self.act = nn.SiLU()
        self.norm = nn.RMSNorm(dim)

        # Separate scales for each component
        self.spectral_scale = nn.Parameter(torch.zeros(1))
        self.lora_scale = nn.Parameter(torch.zeros(1))

        self.up_std = _dense_target_std(dim, self.ff_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        h = self.norm(x)

        # Generate all structured params via single attention call
        gen_out = self.generator(h)  # (B, n_queries, out_per_query)
        gen_flat = gen_out.reshape(B, -1)[:, :self.total_gen]

        # Split into spectral and LoRA components
        spectral_raw, lora_raw = gen_flat.split([self.spectral_size, self.lora_size], dim=-1)

        # --- Spectral modulation (per-sequence, applied to all tokens) ---
        if self.complex_mask:
            mag, phase = spectral_raw.view(B, 2, self.freq_dim).unbind(1)
            mask = torch.complex(
                1.0 + self.spectral_scale * torch.tanh(mag),
                self.spectral_scale * torch.tanh(phase) * 0.5
            )
        else:
            mask = 1.0 + self.spectral_scale * torch.tanh(spectral_raw)

        # Apply spectral mask: broadcast (B, freq_dim) over S
        # Cast to float32 - FFT doesn't support bf16
        orig_dtype = h.dtype
        h_f32 = h.float() if h.dtype == torch.bfloat16 else h
        h_freq = torch.fft.rfft(h_f32, dim=-1)                    # (B, S, freq_dim)
        mask_broadcast = mask.unsqueeze(1).to(h_freq.dtype)
        h_freq = h_freq * mask_broadcast                          # broadcast over S
        h = torch.fft.irfft(h_freq, n=D, dim=-1).to(orig_dtype)   # (B, S, dim)

        # --- Low-rank FFN delta (per-sequence) ---
        lora_a, lora_b = lora_raw.view(B, 2, D, self.rank).unbind(1)  # each (B, dim, rank)
        lora_a = _scale_tensor_std(lora_a, self.up_std)
        lora_b = _scale_tensor_std(lora_b, self.up_std)
        # Fused low-rank delta: (B, dim, dim)
        # But don't materialize — apply sequentially
        # h @ (I + scale * a @ b^T) = h + scale * (h @ a) @ b^T

        # Static FFN up-projection + low-rank delta
        h_up = self.w1(h)  # (B, S, ff_dim) — standard path

        # Low-rank correction applied in dim-space before up-proj would require
        # materializing. Instead, we add a rank-r correction in ff_dim space:
        # This is a different but equally valid formulation
        h_low = torch.einsum('bsd,bdr->bsr', h, lora_a)       # (B, S, rank)
        h_low = torch.einsum('bsr,bdr->bsd', h_low, lora_b)   # (B, S, dim)
        h_up = h_up + self.lora_scale * self.w1(h_low)         # project correction through w1

        h_up = self.act(h_up)
        return self.w2(h_up)


# ---------------------------------------------------------------------------
# Experiment 10: Attention-Generated SwiGLU Gate
# ---------------------------------------------------------------------------

class AttnGenSwiGLU(nn.Module):
    """
    SwiGLU where the gate path is conditioned by cross-attention to the sequence.
    
    Standard SwiGLU: out = (xW1 ⊙ swish(xV)) W2
    This layer:      out = (xW1 ⊙ swish(x(V + Δ(seq)))) W2
    
    The gate projection V gets a per-sequence low-rank perturbation generated
    by cross-attention. This means the gating behavior adapts to the content
    of the entire sequence, while the static path remains unchanged.
    
    Key property: the attention generator's parameters are O(dim²) regardless
    of the rank of the gate perturbation. Higher rank = more expressive 
    gate adaptation at zero additional parameter cost (only more compute
    for more query tokens and larger output projections).
    
    Per-sequence variant (bidirectional). For causal models, the perturbation
    is shared across all positions, which is safe if it's acceptable that
    early tokens' gates are influenced by later tokens. Otherwise use a
    causal aggregation method.
    
    Args:
        dim: model dimension
        ff_mult: FFN expansion factor
        gate_rank: rank of the gate perturbation
        heads: attention heads for generator
    """

    def __init__(
        self,
        dim: int,
        ff_mult: int = 4,
        gate_rank: int = 32,
        heads: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.ff_dim = dim * ff_mult
        self.gate_rank = gate_rank

        # Standard SwiGLU components
        self.w1 = nn.Linear(dim, self.ff_dim, bias=False)  # linear path
        self.v = nn.Linear(dim, self.ff_dim, bias=False)    # gate path (static base)
        self.w2 = nn.Linear(self.ff_dim, dim, bias=False)   # output

        # Generate low-rank gate perturbation via attention
        # Need two factors: (dim, rank) and (ff_dim, rank)
        # Total: dim*rank + ff_dim*rank values
        gen_size = dim * gate_rank + self.ff_dim * gate_rank
        self.gen_size = gen_size
        self.dim_factor_size = dim * gate_rank
        self.ff_factor_size = self.ff_dim * gate_rank

        n_queries = max(gate_rank, 16)
        out_per_query = math.ceil(gen_size / n_queries)
        self.total_out = n_queries * out_per_query

        self.generator = _AttnGenerator(dim, n_queries, out_per_query, heads)

        self.act = nn.SiLU()
        self.norm = nn.RMSNorm(dim)
        self.delta_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        h = self.norm(x)

        # Static paths
        h_linear = self.w1(h)       # (B, S, ff_dim)
        h_gate = self.v(h)          # (B, S, ff_dim) — static gate

        # Generate gate perturbation via cross-attention
        gen_out = self.generator(h)  # (B, n_queries, out_per_query)
        gen_flat = gen_out.reshape(B, -1)[:, :self.gen_size]

        # Split into two low-rank factors
        down_flat, up_flat = gen_flat.split(
            [self.dim_factor_size, self.ff_factor_size], dim=-1
        )
        # down: (B, dim, rank), up: (B, ff_dim, rank)
        down = down_flat.view(B, D, self.gate_rank)
        up = up_flat.view(B, self.ff_dim, self.gate_rank)

        # Apply low-rank perturbation to gate:
        # gate_effective = V @ x + scale * up @ down^T @ x
        # = h_gate + scale * (x @ down) @ up^T
        h_delta = torch.einsum('bsd,bdr->bsr', h, down)        # (B, S, rank)
        h_delta = torch.einsum('bsr,bfr->bsf', h_delta, up)    # (B, S, ff_dim)
        h_gate = h_gate + self.delta_scale * h_delta

        # SwiGLU combination
        h = h_linear * self.act(h_gate)
        return self.w2(h)


# ---------------------------------------------------------------------------
# Experiment 11: Attention-Generated Composite Transform
# ---------------------------------------------------------------------------

class AttnGenCompositeFFN(nn.Module):
    """
    Single cross-attention call generates parameters for multiple structured
    transforms applied in sequence. Demonstrates maximal reuse of the 
    attention-based generation mechanism.
    
    One attention pass produces:
    - Kronecker factors (structured linear mixing)
    - Spectral mask (frequency-domain modulation)
    - Diagonal gate (per-feature scaling)
    
    All three are applied to the input before a standard FFN.
    The attention generator's parameter cost is O(dim²) regardless of how 
    many transform types we pack into the output.
    
    Per-sequence variant (bidirectional).
    
    Args:
        dim: model dimension (must equal p * q)
        ff_mult: FFN expansion factor
        p, q: Kronecker factor sizes
        heads: attention heads for generator
    """

    def __init__(
        self,
        dim: int,
        ff_mult: int = 4,
        p: int = 64,
        q: int = 64,
        heads: int = 8,
    ):
        super().__init__()
        assert p * q == dim
        self.p, self.q = p, q
        self.ff_dim = dim * ff_mult
        self.freq_dim = dim // 2 + 1

        # Calculate total generation budget
        self.kron_size = p * p + q * q
        self.spectral_size = self.freq_dim
        self.diag_size = dim
        total_gen = self.kron_size + self.spectral_size + self.diag_size
        self.total_gen = total_gen

        # Single attention generator for everything
        n_queries = max(p + q, 32)  # modest query count
        out_per_query = math.ceil(total_gen / n_queries)
        self.total_out = n_queries * out_per_query

        self.generator = _AttnGenerator(dim, n_queries, out_per_query, heads)

        # Static bases
        self.base_A = nn.Parameter(torch.eye(p) + torch.randn(p, p) * 0.01)
        self.base_B = nn.Parameter(torch.eye(q) + torch.randn(q, q) * 0.01)

        # Static FFN
        self.w1 = nn.Linear(dim, self.ff_dim, bias=False)
        self.w2 = nn.Linear(self.ff_dim, dim, bias=False)
        self.act = nn.SiLU()
        self.norm = nn.RMSNorm(dim)

        # Per-component scales
        self.kron_scale = nn.Parameter(torch.zeros(1))
        self.spectral_scale = nn.Parameter(torch.zeros(1))
        self.diag_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        h = self.norm(x)

        # Single attention call generates all transform params
        gen_out = self.generator(h)
        gen_flat = gen_out.reshape(B, -1)[:, :self.total_gen]

        kron_raw, spectral_raw, diag_raw = gen_flat.split(
            [self.kron_size, self.spectral_size, self.diag_size], dim=-1
        )

        # 1. Kronecker mixing (per-sequence)
        A_flat, B_flat = kron_raw.split([self.p * self.p, self.q * self.q], dim=-1)
        A = self.base_A.unsqueeze(0) + self.kron_scale * A_flat.view(B, self.p, self.p)
        B_eff = self.base_B.unsqueeze(0) + self.kron_scale * B_flat.view(B, self.q, self.q)

        h_r = h.view(B, S, self.p, self.q)
        h_r = torch.einsum('bij,bsjk->bsik', B_eff, h_r)
        h_r = torch.einsum('bsij,bkj->bsik', h_r, A)
        h = h_r.reshape(B, S, D)

        # 2. Spectral modulation (per-sequence, broadcast over S)
        # Cast to float32 - FFT doesn't support bf16
        orig_dtype = h.dtype
        spectral_mask = 1.0 + self.spectral_scale * torch.tanh(spectral_raw)  # (B, freq_dim)
        h_f32 = h.float() if h.dtype == torch.bfloat16 else h
        h_freq = torch.fft.rfft(h_f32, dim=-1)
        h_freq = h_freq * spectral_mask.unsqueeze(1).to(h_freq.dtype)
        h = torch.fft.irfft(h_freq, n=D, dim=-1).to(orig_dtype)

        # 3. Diagonal gate (per-sequence, broadcast over S)
        diag = 1.0 + self.diag_scale * torch.tanh(diag_raw)  # (B, dim)
        h = h * diag.unsqueeze(1)

        # Standard FFN
        h = self.w1(h)
        h = self.act(h)
        return self.w2(h)


# ---------------------------------------------------------------------------
# Experiment 6: Per-Sequence Kronecker Transform (Variant 2)
# ---------------------------------------------------------------------------

class SequenceKroneckerLayer(nn.Module):
    """
    Per-sequence Kronecker transform (Variant 2 from our discussion).
    
    Pools the entire sequence into a conditioning vector, generates one
    Kronecker transform per sequence, applies it to all tokens.
    
    WARNING: Uses full sequence pooling — NOT suitable for causal/autoregressive
    models. Designed for bidirectional models (ViT, diffusion transformers, 
    BERT-style encoders).
    
    Args:
        dim: model dimension (must equal p * q)
        p: first Kronecker factor size
        q: second Kronecker factor size
        pool_method: how to aggregate sequence into conditioning vector
    """

    def __init__(
        self,
        dim: int,
        p: int = 64,
        q: int = 64,
        pool_method: Literal["mean", "max", "attn"] = "mean",
    ):
        super().__init__()
        assert p * q == dim
        self.p, self.q = p, q
        self.pool_method = pool_method

        if pool_method == "attn":
            self.pool_query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
            self.pool_proj = nn.Linear(dim, dim, bias=False)

        # Generate Kronecker factors from pooled vector
        self.w_hyper = nn.Linear(dim, p * p + q * q, bias=False)
        nn.init.normal_(self.w_hyper.weight, std=0.01)

        # Static base (identity-like initialization)
        self.base_A = nn.Parameter(torch.eye(p) + torch.randn(p, p) * 0.01)
        self.base_B = nn.Parameter(torch.eye(q) + torch.randn(q, q) * 0.01)

        self.norm = nn.RMSNorm(dim)
        self.delta_scale = nn.Parameter(torch.zeros(1))

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce (B, S, D) -> (B, D)"""
        if self.pool_method == "mean":
            return x.mean(dim=1)
        elif self.pool_method == "max":
            return x.max(dim=1).values
        elif self.pool_method == "attn":
            # Single query attending to all tokens
            q = self.pool_query.expand(x.shape[0], -1, -1)
            k = v = self.pool_proj(x)
            out = F.scaled_dot_product_attention(q, k, v)
            return out.squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, S, dim) — BIDIRECTIONAL only
        Returns:
            (B, S, dim)
        """
        B, S, D = x.shape
        h = self.norm(x)

        # Pool sequence -> one vector per batch
        c = self._pool(h)  # (B, D)

        # Generate Kronecker delta factors
        hyper_out = self.w_hyper(c)  # (B, p^2 + q^2)
        A_flat, B_flat = hyper_out.split([self.p * self.p, self.q * self.q], dim=-1)
        A_delta = A_flat.view(B, self.p, self.p)
        B_delta = B_flat.view(B, self.q, self.q)

        # Effective factors = base + dynamic delta
        A = self.base_A.unsqueeze(0) + self.delta_scale * A_delta
        B_eff = self.base_B.unsqueeze(0) + self.delta_scale * B_delta

        # Apply same transform to all tokens
        # x: (B, S, p, q)
        h_reshaped = h.view(B, S, self.p, self.q)
        # B_eff @ h @ A^T  (batched over both B and S)
        # Expand factors for broadcasting over S
        h_reshaped = torch.einsum('bij,bsjk->bsik', B_eff, h_reshaped)
        h_reshaped = torch.einsum('bsij,bkj->bsik', h_reshaped, A)
        h_out = h_reshaped.reshape(B, S, D)

        return x + h_out