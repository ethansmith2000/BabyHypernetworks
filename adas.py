import torch
from torch import nn
from .sparse_layers import AlmostMonarch, MothMatrix
import torch.nn.functional as F


class HyperAttentionLinear(nn.Module):

    """
    HyperAttention. Do perceiver attention to create weight matrix that is (dim*2 x dim), chunk it
    then use it as an MLP with activation to inputs
    """

    def __init__(self, x_dim_in, x_dim_out, hidden_attn_dim=None, kv_dim=None, heads=8):
        super().__init__()
        self.h = heads
        # if no external condition, x will be the input used to create the weight matrix
        kv_dim = kv_dim or x_dim_in
        # if no hidden_dim, use x_dim_out
        hidden_attn_dim = hidden_attn_dim or x_dim_out
        self.head_dim = hidden_attn_dim // heads

        # a learnable weight matrix, is used as queries and is combined with the attention output to give final weight matrix
        base_weight = torch.empty(1, x_dim_in, x_dim_out)
        torch.nn.init.kaiming_uniform_(base_weight)
        self.base_weight = torch.nn.Parameter(base_weight)

        # attention weights
        self.q_proj = nn.Linear(x_dim_out, hidden_attn_dim, bias=False)
        self.kv_norm = nn.LayerNorm(kv_dim)
        self.kv_proj = nn.Linear(kv_dim, hidden_attn_dim * 2, bias=False)
        self.to_out = nn.Linear(hidden_attn_dim, x_dim_out)


    def forward(self, x, context=None):
        # we can use external context or the input itself to guide the weight matrix creation
        context = context if context is not None else x
        B, N, Ci = context.shape
        hd = self.q_proj.out_features
        L, Co = self.base_weight.shape[1:]
        base_weight = self.base_weight.expand(B, -1, -1)

        # base weight serves as queries
        queries = self.q_proj(base_weight)
        norm_kv = self.kv_norm(context)
        k, v = map(lambda t: t.reshape(B, N, self.h, self.head_dim).transpose(1, 2), self.kv_proj(norm_kv).chunk(2, dim=-1))
        q = queries.reshape(B, L, self.h, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, L, hd)
        attn_out = self.to_out(attn_out)

        # b, c, c_out
        layer = base_weight + attn_out

        # apply our new layer to x
        x = torch.einsum('bnd,bdr->bnr', x, layer)

        return x



class HyperAttentionMLP(nn.Module):

    """
    HyperAttention. Do perceiver attention to create weight matrix that is (dim*2 x dim), chunk it
    then use it as an MLP with activation to inputs
    """

    def __init__(self, x_dim, hidden_dim, hidden_attn_dim=None, kv_dim=None, heads=8):
        super().__init__()
        self.h = heads
        # if no external condition, x will be the input used to create the weight matrix
        kv_dim = kv_dim or x_dim
        # if no hidden_dim, use x_dim_out
        hidden_attn_dim = hidden_attn_dim or x_dim
        self.head_dim = hidden_attn_dim // heads

        base_up = torch.randn(1, x_dim, hidden_dim)
        base_down = torch.randn(1, hidden_dim, x_dim)
        torch.nn.init.kaiming_uniform_(base_up)
        torch.nn.init.kaiming_uniform_(base_down)

        base_weight = torch.cat([base_down, base_up.transpose(1, 2)], dim=1)
        self.base_weight = torch.nn.Parameter(base_weight)

        self.q_proj = nn.Linear(x_dim, hidden_attn_dim, bias=False)
        self.kv_norm = nn.LayerNorm(kv_dim)
        self.kv_proj = nn.Linear(kv_dim, hidden_attn_dim * 2, bias=False)
        self.to_out = nn.Linear(hidden_attn_dim, x_dim)

        self.act_fn = nn.GELU()

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
        k, v = map(lambda t: t.reshape(B, N, self.h, self.head_dim).transpose(1, 2), self.kv_proj(norm_kv).chunk(2, dim=-1))
        q = queries.reshape(B, L, self.h, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, L, hd)
        attn_out = self.to_out(attn_out)

        # b, ff_dim * 2, c_out
        layer = base_weight + attn_out

        # chunk to get up and down
        up, down = layer.chunk(2, dim=1)

        # apply our new layer to x
        x = torch.einsum('bnd,bld->bnl', x, up)
        x = self.act_fn(x)
        x = torch.einsum('bnd,bld->bnl', x, down.transpose(1, 2))

        return x


class HyperAttentionAttention(nn.Module):

    """
    HyperAttention. Do perceiver attention to create query, key, value, and out matrices
    then use that for attention
    if you're a bit unhinged, this could technically be done recursively forever until you OOM
    """

    def __init__(self, x_dim, hidden_dim, hidden_attn_dim=None, kv_dim=None, heads=8):
        super().__init__()
        self.h = heads
        # if no external condition, x will be the input used to create the weight matrix
        kv_dim = kv_dim or x_dim
        # if no hidden_dim, use x_dim_out
        hidden_attn_dim = hidden_attn_dim or x_dim
        self.head_dim = hidden_attn_dim // heads

        query = torch.randn(1, x_dim, hidden_dim)
        key = torch.randn(1, x_dim, hidden_dim)
        value = torch.randn(1, x_dim, hidden_dim)
        to_out = torch.randn(1, hidden_dim, x_dim)
        torch.nn.init.kaiming_uniform_(query)
        torch.nn.init.kaiming_uniform_(key)
        torch.nn.init.kaiming_uniform_(value)
        torch.nn.init.kaiming_uniform_(to_out)

        base_weight = torch.cat([query, key, value, to_out.transpose(1, 2)], dim=1)
        self.base_weight = torch.nn.Parameter(base_weight)

        self.q_proj = nn.Linear(hidden_dim, hidden_attn_dim, bias=False)
        self.kv_norm = nn.LayerNorm(kv_dim)
        self.kv_proj = nn.Linear(kv_dim, hidden_attn_dim * 2, bias=False)
        self.to_out = nn.Linear(hidden_attn_dim, hidden_dim)

        self.act_fn = nn.GELU()

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
        k, v = map(lambda t: t.reshape(B, N, self.h, self.head_dim).transpose(1, 2), self.kv_proj(norm_kv).chunk(2, dim=-1))
        q = queries.reshape(B, L, self.h, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, L, hd)
        attn_out = self.to_out(attn_out)

        # b, dim * 4, c_out
        layer = base_weight + attn_out

        # chunk to get up and down
        q_proj, k_proj, v_proj, o_proj = layer.chunk(4, dim=1)

        # apply our new layer to x
        q2 = torch.einsum('bnd,bdl->bnl', x, q_proj)
        k2 = torch.einsum('bnd,bdl->bnl', x, k_proj)
        v2 = torch.einsum('bnd,bdl->bnl', x, v_proj)

        out = F.scaled_dot_product_attention(q2, k2, v2)
        out = torch.einsum('bnl,blc->bnc', out, o_proj.transpose(1, 2))

        return out