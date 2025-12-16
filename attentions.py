import math
import torch
from torch import nn
from sparse_layers import AlmostMonarch, MothMatrix
import torch.nn.functional as F


def _dense_target_std(fan_in: int, fan_out: int) -> float:
    return math.sqrt(2.0 / (fan_in + fan_out))


def _scale_tensor_std(tensor: torch.Tensor, target_std: float, eps: float = 1e-8) -> torch.Tensor:
    if tensor.dim() < 2:
        raise ValueError("HyperAttention tensors must be at least 2D.")
    flat_std = tensor.flatten(1).std(dim=1, keepdim=True, unbiased=False)
    scale = target_std / (flat_std + eps)
    reshape_dims = [tensor.shape[0]] + [1] * (tensor.dim() - 1)
    return tensor * scale.view(*reshape_dims)


class HyperAttentionLinear(nn.Module):

    """
    HyperAttention. Do perceiver attention to create weight matrix that is (dim*2 x dim), chunk it
    then use it as an MLP with activation to inputs

    :param x_dim_in: input dimension of our produced weight matrix
    :param x_dim_out: output dimension of our produced weight matrix
    :param hidden_attn_dim: hidden dimension to perform attention at
    :param kv_dim: key and value dimension if using seperate vector for conditioning
    :param heads: number of attention heads
    """

    def __init__(self, 
                x_dim_in, 
                x_dim_out, 
                hidden_attn_dim=None, 
                kv_dim=None, 
                heads=8
                ):
        super().__init__()
        self.h = heads
        # if no external condition, x will be the input used to create the weight matrix
        kv_dim = kv_dim or x_dim_in
        # if no hidden_dim, use x_dim_out
        hidden_attn_dim = hidden_attn_dim or x_dim_out
        self.head_dim = hidden_attn_dim // heads
        self.layer_std = _dense_target_std(x_dim_in, x_dim_out)

        # a learnable weight matrix, is used as queries and is combined with the attention output to give final weight matrix
        base_weight = torch.empty(1, x_dim_in, x_dim_out)
        torch.nn.init.kaiming_uniform_(base_weight)
        self.base_weight = torch.nn.Parameter(base_weight)
        query_seed = torch.empty(1, x_dim_in, hidden_attn_dim)
        torch.nn.init.kaiming_uniform_(query_seed)
        self.query_seed = torch.nn.Parameter(query_seed)

        # attention weights
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

        # learned queries already in attention hidden space
        queries = self.query_seed.expand(B, -1, -1)
        norm_kv = self.kv_norm(context)
        k, v = map(lambda t: t.reshape(B, N, self.h, self.head_dim).transpose(1, 2), self.kv_proj(norm_kv).chunk(2, dim=-1))
        q = queries.reshape(B, L, self.h, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, L, hd)
        attn_out = self.to_out(attn_out)
        attn_out = _scale_tensor_std(attn_out, self.layer_std)

        # b, c, c_out
        layer = base_weight + attn_out

        # apply our new layer to x
        x = torch.einsum('bnd,bdr->bnr', x, layer)

        return x



class HyperAttentionMLP(nn.Module):

    """
    HyperAttention. Do perceiver attention to create weight matrix that is (dim*2 x dim), chunk it
    then use it as an MLP with activation to inputs

    :param x_dim_in: input and output dimension of our produced weight matrix
    :param hidden_dim: hidden dimension of our produced weight matrix
    :param hidden_attn_dim: hidden dimension to perform attention at
    :param kv_dim: key and value dimension if using seperate vector for conditioning
    :param heads: number of attention heads
    """

    def __init__(self, 
                dim, 
                heads=8,
                ff_mult=2
                ):
        super().__init__()
        self.h = heads
        self.head_dim = dim // heads
        self.ff_dim = dim * ff_mult

        self.temp = nn.Parameter(torch.ones(1, self.h, 1, 1) * 10)

        # we need 2 * ff_dim tokens so chunking yields up and down matrices
        query_seed = torch.empty(1, self.ff_dim * 2, dim)
        torch.nn.init.xavier_normal_(query_seed)
        self.query_seed = nn.Parameter(query_seed)

        self.norm = lambda x: F.normalize(x, p=2, dim=-1)

        self.kv_proj = nn.Linear(dim, dim * 2, bias=False)
        self.to_out = nn.Linear(dim, dim)

        self.act_fn = nn.GELU()
        self.up_std = _dense_target_std(dim, self.ff_dim)
        self.down_std = _dense_target_std(self.ff_dim, dim)

    def forward(self, x):
        # we can use external context or the input itself to guide the weight matrix creation
        B, N, dim = x.shape
        q_toks = self.query_seed.shape[1]

        # learned queries already in attention hidden space
        queries = self.query_seed.expand(B, -1, -1)
        k, v = map(lambda t: t.reshape(B, N, self.h, self.head_dim).transpose(1, 2), self.kv_proj(x).chunk(2, dim=-1))
        q = queries.reshape(B, q_toks, self.h, self.head_dim).transpose(1, 2)

        q = self.norm(q)
        k = self.norm(k)
        q = q * self.temp

        attn_out = F.scaled_dot_product_attention(q, k, v, scale=1.0).transpose(1, 2).reshape(B, q_toks, dim)
        attn_out = self.to_out(attn_out)

        # b, ff_dim * 2, c_out
        layer = attn_out

        # chunk to get up and down
        up, down = layer.chunk(2, dim=1)
        up = _scale_tensor_std(up, self.up_std)
        down = _scale_tensor_std(down, self.down_std)

        # apply our new layer to x
        x = torch.einsum('bnd,bld->bnl', x, up)
        x = self.act_fn(x)
        x = torch.einsum('bnd,bld->bnl', x, down.transpose(1, 2))

        return x


class HyperAttentionAttention(nn.Module):

    """
    HyperAttention. Do perceiver attention to create query, key, value, and out matrices
    then use that for attention

    :param x_dim: input and output dimension of our produced weight matrix
    :param hidden_dim: hidden dimension of our produced weight matrix
    :param hidden_attn_dim: hidden dimension to perform attention at
    :param kv_dim: key and value dimension if using seperate vector for conditioning
    :param heads: number of attention heads
    """

    def __init__(self, 
                x_dim, 
                hidden_dim, 
                hidden_attn_dim=None, 
                kv_dim=None, 
                heads=8
                ):
        super().__init__()
        self.h = heads
        # if no external condition, x will be the input used to create the weight matrix
        kv_dim = kv_dim or x_dim
        # if no hidden_dim, use x_dim_out
        hidden_attn_dim = hidden_attn_dim or x_dim
        self.head_dim = hidden_attn_dim // heads

        query, key, value = map(lambda t: torch.randn(1, x_dim, hidden_dim), range(3))
        to_out = torch.randn(1, hidden_dim, x_dim)
        query, key, value, to_out = map(lambda t: torch.nn.init.kaiming_uniform_(t), [query, key, value, to_out])

        base_weight = torch.cat([query, key, value, to_out.transpose(1, 2)], dim=1)
        self.base_weight = torch.nn.Parameter(base_weight)

        query_seed = torch.empty(1, 4 * x_dim, hidden_attn_dim)
        torch.nn.init.kaiming_uniform_(query_seed)
        self.query_seed = nn.Parameter(query_seed)
        self.kv_norm = nn.LayerNorm(kv_dim)
        self.kv_proj = nn.Linear(kv_dim, hidden_attn_dim * 2, bias=False)
        self.to_out = nn.Linear(hidden_attn_dim, hidden_dim)

        self.act_fn = nn.GELU()
        self.qkv_std = _dense_target_std(x_dim, hidden_dim)
        self.o_std = _dense_target_std(hidden_dim, x_dim)

    def forward(self, x, context=None):
        # we can use external context or the input itself to guide the weight matrix creation
        context = context if context is not None else x
        B, N, Ci = context.shape
        L = self.base_weight.shape[1]
        hd = self.q_proj.out_features
        base_weight = self.base_weight.expand(B, -1, -1)

        # learned queries already in attention hidden space
        queries = self.query_seed.expand(B, -1, -1)
        norm_kv = self.kv_norm(context)
        k, v = map(lambda t: t.reshape(B, N, self.h, self.head_dim).transpose(1, 2), self.kv_proj(norm_kv).chunk(2, dim=-1))
        q = queries.reshape(B, L, self.h, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, L, hd)
        attn_out = self.to_out(attn_out)

        # b, dim * 4, c_out
        layer = base_weight + attn_out

        # chunk to get query/key/value/out projections
        q_proj, k_proj, v_proj, o_proj = layer.chunk(4, dim=1)
        q_proj = _scale_tensor_std(q_proj, self.qkv_std)
        k_proj = _scale_tensor_std(k_proj, self.qkv_std)
        v_proj = _scale_tensor_std(v_proj, self.qkv_std)
        o_proj = _scale_tensor_std(o_proj, self.o_std)

        # apply our new layer to x
        q2 = torch.einsum('bnd,bdl->bnl', x, q_proj)
        k2 = torch.einsum('bnd,bdl->bnl', x, k_proj)
        v2 = torch.einsum('bnd,bdl->bnl', x, v_proj)

        out = F.scaled_dot_product_attention(q2, k2, v2)
        out = torch.einsum('bnl,blc->bnc', out, o_proj.transpose(1, 2))

        return out