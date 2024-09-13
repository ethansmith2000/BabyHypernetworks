import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Standard Multi-head attention layer
    Weight matrices are provided through the forward method

    :param heads: number of heads
    :param x_dim: input dimension
    """

    def __init__(self, heads=8, x_dim=None):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(x_dim)

    def extra_repr(self):
        return f'heads={self.heads}'

    def forward(self, x, wq, wk, wv, wo):
        B, N, D = x.shape
        x = self.norm(x)
        Q, K, V = map(lambda w: torch.einsum('bnd,bdl->bnl', x, w), (wq, wk, wv))
        Q, K, V = map(lambda t: t.reshape(B, N, self.heads, -1).transpose(1, 2), (Q, K, V))
        attn_out = F.scaled_dot_product_attention(Q, K, V).transpose(1, 2).reshape(B, N, -1)
        return torch.einsum('bnd,bdl->bnl', attn_out, wo)


class HyperAttention(nn.Module):
    """
    HyperAttention. Do perceiver attention to create query, key, value, and out matrices
    then use that for attention recursively to create a new layer

    :param hidden_dim: hidden dimension for performing attention
    :param x_dim: input dimension
    :param depth: number of layers below this one
    :param heads: number of heads
    """

    def __init__(self,
                 hidden_dim,
                 x_dim,
                 depth,
                 heads=8
                 ):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(x_dim)

        q_proj_hyp, k_proj_hyp, v_proj_hyp = map(lambda t: torch.randn(1, x_dim, hidden_dim), range(3))
        to_out_hyp = torch.randn(1, hidden_dim, x_dim)
        q_proj_hyp, k_proj_hyp, v_proj_hyp, to_out_hyp = map(lambda t: torch.nn.init.kaiming_uniform_(t), (q_proj_hyp, k_proj_hyp, v_proj_hyp, to_out_hyp))

        base_weight = torch.cat([q_proj_hyp, k_proj_hyp, v_proj_hyp, to_out_hyp.transpose(1, 2)], dim=1)
        self.base_weight = torch.nn.Parameter(base_weight)

        self.sub_attention = HyperAttention(hidden_dim, x_dim, depth - 1, heads) if depth >= 2 else Attention(
            heads, x_dim) if depth >= 1 else None

    def extra_repr(self):
        return f'heads={self.heads}, base_weight={self.base_weight.shape}'

    def forward(self, x, wq, wk, wv, wo):
        B, N, D = x.shape
        L = self.base_weight.shape[1]
        x = self.norm(x)
        base_weight = self.base_weight.expand(B, -1, -1)
        Q = torch.einsum('bnd,bdl->bnl', base_weight, wq.transpose(1, 2))
        K = torch.einsum('bnd,bdl->bnl', x, wk.transpose(1, 2))
        V = torch.einsum('bnd,bdl->bnl', x, wv.transpose(1, 2))
        K, V = map(lambda t: t.reshape(B, N, self.heads, -1).transpose(1, 2), (K, V))
        Q = Q.reshape(B, L, self.heads, -1).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(Q, K, V).transpose(1, 2).reshape(B, L, -1)
        out = torch.einsum('bnd,bdl->bnl', attn_out, wo)

        layer = base_weight + out

        q_proj, k_proj, v_proj, o_proj = layer.chunk(4, dim=1)

        if self.sub_attention is not None:
            out = self.sub_attention(x, q_proj, k_proj, v_proj, o_proj)
        else:
            q2, k2, v2 = map(lambda w: torch.einsum('bnd,bdl->bnl', x, w), (q_proj, k_proj, v_proj))
            out = F.scaled_dot_product_attention(q2, k2, v2)
            out = torch.einsum('bnl,blc->bnc', out, o_proj.transpose(1, 2))

        return out


class InsaneRecursiveAttention(nn.Module):
    """
    To the joker this is just normal attention,
    but to the rest of us it's InsaneRecursiveAttention

    :param x_dim: input dimension
    :param hidden_dim: hidden dimension for performing attention
    :param hidden_attn_dim: hidden dimension for creating attention matrices
    :param kv_dim: key and value dimension
    :param heads: number of heads
    :param depth: number of layers below this one
    """

    def __init__(self,
                 x_dim,
                 hidden_dim,
                 hidden_attn_dim=None,
                 kv_dim=None,
                 heads=8,
                 depth=3
                 ):
        super().__init__()
        self.h = heads
        # if no external condition, x will be the input used to create the weight matrix
        kv_dim = kv_dim or x_dim
        # if no hidden_dim, use x_dim_out
        hidden_attn_dim = hidden_attn_dim or x_dim
        self.head_dim = hidden_attn_dim // heads


        q_proj_hyp, k_proj_hyp, v_proj_hyp = map(lambda t: torch.randn(1, x_dim, hidden_dim), range(3))
        to_out_hyp = torch.randn(1, hidden_dim, x_dim)
        q_proj_hyp, k_proj_hyp, v_proj_hyp, to_out_hyp = map(lambda t: torch.nn.init.kaiming_uniform_(t), (q_proj_hyp, k_proj_hyp, v_proj_hyp, to_out_hyp))

        base_weight = torch.cat([q_proj_hyp, k_proj_hyp, v_proj_hyp, to_out_hyp.transpose(1, 2)], dim=1)
        self.base_weight = torch.nn.Parameter(base_weight)

        self.q_proj = nn.Linear(hidden_dim, hidden_attn_dim, bias=False)
        self.kv_norm = nn.LayerNorm(kv_dim)
        self.kv_proj = nn.Linear(kv_dim, hidden_attn_dim * 2, bias=False)
        self.to_out = nn.Linear(hidden_attn_dim, hidden_dim)

        self.act_fn = nn.GELU()

        self.sub_attention = HyperAttention(hidden_dim, x_dim, depth - 1, heads) if depth >= 2 else Attention(
            heads, x_dim) if depth >= 1 else None

    def forward(self, x, context=None):
        # we can use external context or the input itself to guide the weight matrix creation
        context = context if context is not None else x
        B, N, Ci = context.shape
        L = self.base_weight.shape[1]
        hd = self.q_proj.out_features
        base_weight = self.base_weight.expand(B, -1, -1)

        # base weight serves as queries
        queries = self.q_proj(base_weight)
        norm_kv = self.kv_norm(context)
        k, v = map(lambda t: t.reshape(B, N, self.h, self.head_dim).transpose(1, 2),
                   self.kv_proj(norm_kv).chunk(2, dim=-1))
        q = queries.reshape(B, L, self.h, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, L, hd)
        attn_out = self.to_out(attn_out)

        # b, dim * 4, c_out
        layer = base_weight + attn_out

        # chunk to get up and down
        q_proj, k_proj, v_proj, o_proj = layer.chunk(4, dim=1)

        if self.sub_attention is not None:
            out = self.sub_attention(x, q_proj, k_proj, v_proj, o_proj)
        else:
            # apply our new layer to x
            q2, k2, v2 = map(lambda t: torch.einsum('bnd,bdl->bnl', x, t), (q_proj, k_proj, v_proj))
            out = F.scaled_dot_product_attention(q2, k2, v2)
            out = torch.einsum('bnl,blc->bnc', out, o_proj.transpose(1, 2))

        return out