import math
import torch
from typing import Optional, Literal

from dynamic_transforms import (
    LowRankHyperGate,
    LowRankHyperFFN,
    SpectralModulation,
    CompositeDynamicFFN,
)


FFN_TYPE = Literal[
    "mlp",
    "swiglu",
    "lowrank_hypergate",
    "lowrank_hyperffn",
    "spectral",
    "butterfly",
    "composite",
]


def compute_kron_factors(dim: int) -> tuple[int, int]:
    """Compute balanced Kronecker factors p, q such that p * q = dim.
    
    Prefers square factorizations (p ≈ q) for balanced parameter counts.
    Returns (p, q) where p >= q.
    """
    sqrt_dim = int(math.sqrt(dim))
    for p in range(sqrt_dim, 0, -1):
        if dim % p == 0:
            q = dim // p
            return (max(p, q), min(p, q))
    return (dim, 1)


LOWRANK_DELTA_TARGET = Literal[
    "gate_input", "linear_input", "both_input",
    "gate_output", "linear_output", "both_output",
    "w2_output", "full"
]

FFN_DELTA_TARGET = Literal[
    "w1_input", "w1_output", "w2_input", "w2_output", "full"
]

LOWRANK_FACTOR_TYPE = Literal["lowrank", "kronecker"]

SPECTRAL_TRANSFORM = Literal["fft", "dct", "hadamard"]
SPECTRAL_MASK_ACT = Literal["tanh", "silu", "none", "softsign"]


def create_ffn(
    ffn_type: FFN_TYPE,
    dim: int,
    ff_mult: int = 4,
    kron_p: Optional[int] = None,
    kron_q: Optional[int] = None,
    lowrank_rank: int = 16,
    lowrank_delta_target: LOWRANK_DELTA_TARGET = "gate_input",
    ffn_delta_target: FFN_DELTA_TARGET = "w1_output",
    lowrank_factor_type: LOWRANK_FACTOR_TYPE = "lowrank",
    lowrank_use_bmm: bool = True,
    butterfly_rounds: int = 3,
    spectral_bottleneck: Optional[int] = None,
    spectral_transform: SPECTRAL_TRANSFORM = "dct",
    spectral_mask_act: SPECTRAL_MASK_ACT = "none",
    spectral_use_additive: bool = False,
    base_weight_mode: str = "regular",
) -> torch.nn.Module:
    """Factory function to create FFN layer based on type.
    
    For Kronecker-based FFNs, kron_p and kron_q must satisfy p * q = dim.
    If not provided or if p * q != dim, factors are auto-computed from dim.
    
    For LowRankHyperGate (SwiGLU-based):
        lowrank_delta_target: where to apply perturbation
            Input-space (dim→dim): gate_input, linear_input, both_input
            Output-space: gate_output, linear_output, both_output, w2_output, full
        lowrank_factor_type: "lowrank" for U@V^T, "kronecker" for A⊗B
        lowrank_use_bmm: True for bmm/matmul (better TF32), False for einsum
    
    For LowRankHyperFFN (simple FFN):
        ffn_delta_target: where to apply perturbation
            w1_input, w1_output, w2_input, w2_output, full
        base_weight_mode: how to parameterize base weights
            "regular": standard nn.Linear weights
            "lowrank_N": base weights as low-rank matrices of rank N
            "none": no base weights (pure dynamic)
    
    For SpectralModulation:
        spectral_bottleneck: bottleneck dim for mask generation (None = direct)
        spectral_transform: "fft" (complex), "dct" (real), "hadamard" (real, power-of-2)
        spectral_mask_act: "tanh", "silu", "none", "softsign"
        spectral_use_additive: add token-dependent frequency injection branch
    """
    if kron_p is None or kron_q is None or kron_p * kron_q != dim:
        auto_p, auto_q = compute_kron_factors(dim)
        if kron_p is not None and kron_q is not None and kron_p * kron_q != dim:
            print(f"Warning: kron_p*kron_q ({kron_p}*{kron_q}={kron_p*kron_q}) != dim ({dim}), "
                  f"auto-computing factors: p={auto_p}, q={auto_q}")
        kron_p, kron_q = auto_p, auto_q
    
    if ffn_type == "mlp":
        return FeedForward(dim, dim * ff_mult)
    elif ffn_type == "swiglu":
        return SwiGLU(dim)
    elif ffn_type == "lowrank_hypergate":
        return LowRankHyperGate(
            dim, ff_mult=ff_mult, rank=lowrank_rank,
            delta_target=lowrank_delta_target, factor_type=lowrank_factor_type,
            use_bmm=lowrank_use_bmm, pre_norm=False
        )
    elif ffn_type == "lowrank_hyperffn":
        return LowRankHyperFFN(
            dim, ff_mult=ff_mult, rank=lowrank_rank,
            delta_target=ffn_delta_target, factor_type=lowrank_factor_type,
            use_bmm=lowrank_use_bmm, base_weight_mode=base_weight_mode
        )
    elif ffn_type == "spectral":
        return SpectralModulation(dim, mask_bottleneck=spectral_bottleneck, transform=spectral_transform, mask_act=spectral_mask_act, use_additive=spectral_use_additive)
    elif ffn_type == "butterfly":
        return ButterflyTransformLayer(dim, n_rounds=butterfly_rounds)
    elif ffn_type == "composite":
        return CompositeDynamicFFN(dim, ff_mult=ff_mult, p=kron_p, q=kron_q)
    else:
        raise ValueError(f"Unknown ffn_type: {ffn_type}")


class Attention(torch.nn.Module):
    def __init__(
        self,
        dim,
        heads,
        is_causal=True,
        use_rope=True,
        rope_theta=10000.0,
        qk_norm=True,
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.is_causal = is_causal
        self.use_rope = use_rope
        self.qk_norm = qk_norm
        self.head_dim = dim // heads
        self.rope_theta = rope_theta

        if self.use_rope and (self.head_dim % 2 != 0):
            raise ValueError("RoPE requires head_dim to be even.")

        # Separate Q/K/V projections for independent blending control
        self.to_qkv = torch.nn.Linear(dim, 3 * dim, bias=False)
        self.to_out = torch.nn.Linear(dim, dim, bias=True)

    def _apply_qk_norm(self, q, k, eps=1e-6):
        if not self.qk_norm:
            return q, k
        q = q * torch.rsqrt(torch.mean(q * q, dim=-1, keepdim=True) + eps)
        k = k * torch.rsqrt(torch.mean(k * k, dim=-1, keepdim=True) + eps)
        return q, k

    def _apply_rope(self, q, k):
        if not self.use_rope:
            return q, k

        b, h, t, d = q.shape
        half = d // 2
        freqs = torch.arange(half, device=q.device, dtype=q.dtype)
        inv_freq = 1.0 / (self.rope_theta ** (freqs / half))
        positions = torch.arange(t, device=q.device, dtype=q.dtype)
        angles = torch.einsum("t,f->tf", positions, inv_freq)
        sin = angles.sin()[None, None, :, :]
        cos = angles.cos()[None, None, :, :]

        def rotate(x):
            x1, x2 = x[..., :half], x[..., half:]
            return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        return rotate(q), rotate(k)

    def forward(self, x):
        qkv = self.to_qkv(x)
        q, k, v = qkv.split(self.dim, dim=-1)
        q, k, v = map(lambda t: t.view(t.shape[0], t.shape[1], self.heads, -1).transpose(1, 2), (q, k, v))
        q, k = self._apply_qk_norm(q, k)
        q, k = self._apply_rope(q, k)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
        attn = attn.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1)
        return self.to_out(attn)


class FeedForward(torch.nn.Module):
    """Simple 2-layer MLP with GELU activation."""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class SwiGLU(torch.nn.Module):
    """Gated Linear Unit with SiLU (Swish) activation."""
    def __init__(self, dim, hidden_dim=None, align_multiple=64):
        super().__init__()
        # Default to SwiGLU parity (~2.67x) and align to a friendly GPU multiple
        hidden_dim = math.ceil(dim * 8 / 3)
        if align_multiple is not None and align_multiple > 1:
            hidden_dim = math.ceil(hidden_dim / align_multiple) * align_multiple
        # First projection produces value and gate in one matmul for efficiency
        self.proj_in = torch.nn.Linear(dim, hidden_dim)
        self.proj_gate = torch.nn.Linear(dim, hidden_dim)
        self.proj_out = torch.nn.Linear(hidden_dim, dim)
        self.act = torch.nn.SiLU()

    def forward(self, x):
        value = self.proj_in(x)
        gate = self.proj_gate(x)
        gated = value * self.act(gate)
        return self.proj_out(gated)


THIRD_LAYER_TYPE = Literal["spectral", "butterfly", "lowrank_hypergate", "lowrank_hyperffn", None]


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        dim,
        heads,
        ff_mult=4,
        is_causal=True,
        use_rope=True,
        rope_theta=10000.0,
        qk_norm=True,
        ffn_type: FFN_TYPE = "swiglu",
        kron_p: Optional[int] = None,
        kron_q: Optional[int] = None,
        lowrank_rank: int = 16,
        lowrank_delta_target: LOWRANK_DELTA_TARGET = "gate_input",
        ffn_delta_target: FFN_DELTA_TARGET = "w1_output",
        lowrank_factor_type: LOWRANK_FACTOR_TYPE = "lowrank",
        lowrank_use_bmm: bool = True,
        butterfly_rounds: int = 3,
        spectral_bottleneck: Optional[int] = None,
        spectral_transform: SPECTRAL_TRANSFORM = "dct",
        spectral_mask_act: SPECTRAL_MASK_ACT = "none",
        spectral_use_additive: bool = False,
        base_weight_mode: str = "regular",
        third_layer_type: THIRD_LAYER_TYPE = None,
    ):
        super().__init__()
        self.attn = Attention(
            dim,
            heads,
            is_causal=is_causal,
            use_rope=use_rope,
            rope_theta=rope_theta,
            qk_norm=qk_norm,
        )
        self.ff = create_ffn(
            ffn_type=ffn_type,
            dim=dim,
            ff_mult=ff_mult,
            kron_p=kron_p,
            kron_q=kron_q,
            lowrank_rank=lowrank_rank,
            lowrank_delta_target=lowrank_delta_target,
            ffn_delta_target=ffn_delta_target,
            lowrank_factor_type=lowrank_factor_type,
            lowrank_use_bmm=lowrank_use_bmm,
            butterfly_rounds=butterfly_rounds,
            spectral_bottleneck=spectral_bottleneck,
            spectral_transform=spectral_transform,
            spectral_mask_act=spectral_mask_act,
            spectral_use_additive=spectral_use_additive,
            base_weight_mode=base_weight_mode,
        )
        self.norm1 = torch.nn.RMSNorm(dim)
        self.norm2 = torch.nn.RMSNorm(dim)
        
        # Optional third residual layer (dynamic transform after FFN)
        self.third_layer_type = third_layer_type
        if third_layer_type is not None:
            self.norm3 = torch.nn.RMSNorm(dim)
            self.third = create_ffn(
                ffn_type=third_layer_type,
                dim=dim,
                ff_mult=ff_mult,
                kron_p=kron_p,
                kron_q=kron_q,
                lowrank_rank=lowrank_rank,
                lowrank_delta_target=lowrank_delta_target,
                ffn_delta_target=ffn_delta_target,
                lowrank_factor_type=lowrank_factor_type,
                lowrank_use_bmm=lowrank_use_bmm,
                butterfly_rounds=butterfly_rounds,
                spectral_bottleneck=spectral_bottleneck,
                spectral_transform=spectral_transform,
                spectral_mask_act=spectral_mask_act,
                spectral_use_additive=spectral_use_additive,
                base_weight_mode=base_weight_mode,
            )

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        x = self.ff(self.norm2(x)) + x
        if self.third_layer_type is not None:
            x = self.third(self.norm3(x)) + x
        return x


class Transformer(torch.nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        ff_mult,
        vocab_size,
        max_seq_len,
        gradient_checkpointing=False,
        use_rope=True,
        rope_theta=10000.0,
        qk_norm=False,
        ffn_type: FFN_TYPE = "swiglu",
        kron_p: Optional[int] = None,
        kron_q: Optional[int] = None,
        lowrank_rank: int = 16,
        lowrank_delta_target: LOWRANK_DELTA_TARGET = "gate_input",
        ffn_delta_target: FFN_DELTA_TARGET = "w1_output",
        lowrank_factor_type: LOWRANK_FACTOR_TYPE = "lowrank",
        lowrank_use_bmm: bool = True,
        butterfly_rounds: int = 3,
        spectral_bottleneck: Optional[int] = None,
        spectral_transform: SPECTRAL_TRANSFORM = "dct",
        spectral_mask_act: SPECTRAL_MASK_ACT = "none",
        spectral_use_additive: bool = False,
        base_weight_mode: str = "regular",
        third_layer_type: THIRD_LAYER_TYPE = None,
    ):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(vocab_size, dim)
        self.use_rope = use_rope
        if not use_rope:
            self.position_embedding = torch.nn.Embedding(max_seq_len, dim)
        self.in_proj = torch.nn.Sequential(
            torch.nn.RMSNorm(dim),
            torch.nn.Linear(dim, dim),
        )
        self.blocks = torch.nn.ModuleList([
            TransformerBlock(
                dim,
                heads,
                ff_mult=ff_mult,
                is_causal=True,
                use_rope=use_rope,
                rope_theta=rope_theta,
                qk_norm=qk_norm,
                ffn_type=ffn_type,
                kron_p=kron_p,
                kron_q=kron_q,
                lowrank_rank=lowrank_rank,
                lowrank_delta_target=lowrank_delta_target,
                ffn_delta_target=ffn_delta_target,
                lowrank_factor_type=lowrank_factor_type,
                lowrank_use_bmm=lowrank_use_bmm,
                butterfly_rounds=butterfly_rounds,
                spectral_bottleneck=spectral_bottleneck,
                spectral_transform=spectral_transform,
                spectral_mask_act=spectral_mask_act,
                spectral_use_additive=spectral_use_additive,
                base_weight_mode=base_weight_mode,
                third_layer_type=third_layer_type,
            ) for _ in range(depth)])
        self.out_proj = torch.nn.Sequential(
            torch.nn.RMSNorm(dim),
            torch.nn.Linear(dim, vocab_size),
        )
        self.gradient_checkpointing = gradient_checkpointing
        self.ffn_type = ffn_type

        # Initialize weights
        embed_std = dim ** -0.5
        lm_head_std = 0.02

        with torch.no_grad():
            for name, module in self.named_modules():
                if isinstance(module, torch.nn.Embedding):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=embed_std)
                elif isinstance(module, torch.nn.RMSNorm):
                    torch.nn.init.ones_(module.weight)
                elif isinstance(module, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)

            # LM head gets special treatment
            torch.nn.init.normal_(self.out_proj[1].weight, mean=0.0, std=lm_head_std)
            if self.out_proj[1].bias is not None:
                torch.nn.init.zeros_(self.out_proj[1].bias)


    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        tok_emb = self.token_embedding(input_ids)
        if self.use_rope:
            x = tok_emb
        else:
            pos = torch.arange(0, T, device=input_ids.device)
            pos_emb = self.position_embedding(pos)
            x = tok_emb + pos_emb
        x = self.in_proj(x)
        if self.gradient_checkpointing:
            for block in self.blocks:
                x = torch.utils.checkpoint.checkpoint(block, x, preserve_rng_state=False, use_reentrant=False, determinism_check="none")
        else:
            for block in self.blocks:
                x = block(x)
        logits = self.out_proj(x)
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), targets.reshape(-1))
            return loss, logits
        else:
            return logits

    def resize_token_embeddings(self, new_size: int):
        """Resize token embedding table, preserving existing weights."""
        if not isinstance(self.token_embedding, torch.nn.Embedding):
            raise NotImplementedError("resize_token_embeddings only supports dense Embedding tables.")

        old_weight = self.token_embedding.weight
        new_emb = torch.nn.Embedding(
            new_size,
            old_weight.shape[1],
            device=old_weight.device,
            dtype=old_weight.dtype,
        )
        with torch.no_grad():
            num_tokens = min(old_weight.shape[0], new_size)
            new_emb.weight[:num_tokens] = old_weight[:num_tokens]
        self.token_embedding = new_emb
        return self.token_embedding
