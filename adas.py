import torch
from torch import nn
import torch.nn.functional as F
from .sparse_layers import AlmostMonarch


class AdaLoRA(nn.Module):
    """
    Conditioning layer, can condition on external condition or input
    condition is projected into a LoRA, low rank A and B matrices
    which transform the input. This could potentially use a lot of parameters
    when using a large rank, so option to use a sparse projection

    shout out to rami_mmo for the initial design and idea!

    :param feat_dim: feature dimension
    :param ada_dim: condition dimension
    :param inter_dim: intermediate dimension, if act_fn is not None, this is the dimension
                    of the hidden layer of the condition projection
    :param rank: rank of the low rank matrices
    :param lora_proj_act_fn: activation function for the hidden layer of the condition projection
    :param sparse_heads: number of heads to use for the sparse projection, set to None for dense projection
    :param norm_cond: whether to normalize the condition before projecting
    """

    def __init__(self,
                 feat_dim,
                 ada_dim,
                 inter_dim=None,
                 rank=8,
                 lora_proj_act_fn=nn.GELU,
                 sparse_heads=None,
                 norm_cond=True
                 ):
        super().__init__()
        self.rank = rank
        layers = []
        inter_dim = ada_dim if inter_dim is None else inter_dim
        if lora_proj_act_fn is None:
            layers.append(nn.Linear(ada_dim, feat_dim * rank * 2))
        else:
            layers.append(nn.Linear(ada_dim, inter_dim))
            layers.append(lora_proj_act_fn())
            layers.append(
                AlmostMonarch(inter_dim, feat_dim * rank * 2, sparse_heads) if sparse_heads is not None else nn.Linear(
                    inter_dim, feat_dim * rank * 2))

        self.gen_weight = nn.Sequential(*layers)
        self.norm_cond = nn.LayerNorm(ada_dim) if norm_cond else nn.Identity()

    def forward(self, x, ada_emb):
        x_in = x
        B, D = x.shape[0], x.shape[-1]
        x_weights = self.gen_weight(self.norm_cond(ada_emb))
        x_a, x_b = x_weights.chunk(2, dim=-1)
        x_a, x_b = map(lambda t: t.reshape(-1, D, self.rank), (x_a, x_b))

        x = torch.einsum('bc,bco->bo', x, x_a)
        x = torch.einsum('bc,bco->bo', x, x_b.permute(0, 2, 1))

        # add back residual otherwise we kill the rank of our representation
        return x + x_in


class AdaLoRAMLP(nn.Module):
    """
    Like AdaLoRA but generates two sets of A/B LoRA matrices and applies them in sequence
    with an activation function in between

    :param feat_dim: feature dimension
    :param ada_dim: condition dimension
    :param inter_dim: intermediate dimension, if act_fn is not None, this is the dimension
                    of the hidden layer of the condition projection
    :param rank: rank of the low rank matrices
    :param lora_proj_act_fn: activation function for the hidden layer of the condition projection
    :param sparse_heads: number of heads to use for the sparse projection, set to None for dense projection
    :param act_fn: activation function to use between the two sets of A/B matrices
    """

    def __init__(self,
                 feat_dim,
                 ada_dim,
                 inter_dim=None,
                 rank=8,
                 lora_proj_act_fn=nn.GELU,
                 sparse_heads=None,
                 act_fn=nn.GELU,
                 norm_cond=True
                 ):
        super().__init__()
        self.rank = rank
        layers = []
        inter_dim = ada_dim if inter_dim is None else inter_dim
        if lora_proj_act_fn is None:
            layers.append(nn.Linear(ada_dim, feat_dim * rank * 4))
        else:
            layers.append(nn.Linear(ada_dim, inter_dim))
            layers.append(lora_proj_act_fn())
            layers.append(
                AlmostMonarch(inter_dim, feat_dim * rank * 4, sparse_heads) if sparse_heads is not None else nn.Linear(
                    inter_dim, feat_dim * rank * 4))

        self.gen_weight = nn.Sequential(*layers)
        self.act_fn = act_fn()
        self.norm_cond = nn.LayerNorm(ada_dim) if norm_cond else nn.Identity()

    def forward(self, x, ada_emb):
        x_in = x
        B, D = x.shape[0], x.shape[-1]
        x_weights = self.gen_weight(self.norm_cond(ada_emb))
        x_a_1, x_b_1, x_a_2, x_b_2 = x_weights.chunk(4, dim=-1)
        x_a_1, x_b_1, x_a_2, x_b_2 = map(lambda t: t.reshape(-1, D, self.rank), (x_a_1, x_b_1, x_a_2, x_b_2))

        x = torch.einsum('bc,bco->bo', x, x_a_1)
        x = torch.einsum('bc,bco->bo', x, x_b_1.permute(0, 2, 1))
        x = self.act_fn(x)
        x = torch.einsum('bc,bco->bo', x, x_a_2)
        x = torch.einsum('bc,bco->bo', x, x_b_2.permute(0, 2, 1))

        # add back residual otherwise we kill the rank of our representation
        return x + x_in


class AdaLoRAWithBase(nn.Module):
    """
    Like AdaLoRA but includes an additional base transformation, the fused LoRA is added
    to base layer to produce final transformation

    :param feat_dim: feature dimension
    :param ada_dim: condition dimension
    :param inter_dim: intermediate dimension, if act_fn is not None, this is the dimension
                    of the hidden layer of the condition projection
    :param rank: rank of the low rank matrices
    :param lora_proj_act_fn: activation function for the hidden layer of the condition projection
    :param sparse_heads: number of heads to use for the sparse projection, set to None for dense projection
    :param residual: whether to add residual connection
    """

    def __init__(self,
                 feat_dim,
                 ada_dim,
                 inter_dim=None,
                 rank=8,
                 lora_proj_act_fn = nn.GELU,
                 sparse_heads=None,
                 residual=True,
                    norm_cond=True
                 ):
        super().__init__()
        self.rank = rank
        self.base_layer = nn.Parameter(torch.randn(feat_dim, feat_dim))
        layers = []
        inter_dim = ada_dim if inter_dim is None else inter_dim
        if lora_proj_act_fn is None:
            layers.append(nn.Linear(ada_dim, feat_dim * rank * 2))
        else:
            layers.append(nn.Linear(ada_dim, inter_dim))
            layers.append(lora_proj_act_fn())
            layers.append(
                AlmostMonarch(inter_dim, feat_dim * rank * 2, sparse_heads) if sparse_heads is not None else nn.Linear(
                    inter_dim, feat_dim * rank * 2))

        self.gen_weight = nn.Sequential(*layers)

        self.residual = lambda x, y: x + y if residual else y
        self.norm_cond = nn.LayerNorm(ada_dim) if norm_cond else nn.Identity()

    def forward(self, x, ada_emb):
        x_in = x
        B, D = x.shape[0], x.shape[-1]
        x_weights = self.gen_weight(self.norm_cond(ada_emb))
        x_a, x_b = x_weights.chunk(2, dim=-1)
        x_a, x_b = map(lambda t: t.reshape(-1, D, self.rank), (x_a, x_b))

        # fuse
        layer = torch.einsum('bdr,brk->bdk', x_a, x_b.transpose(1, 2))
        layer = self.base_layer[None, ...] + layer

        x = torch.einsum('bc,bco->bo', x, layer)

        return self.residual(x_in, x)


class AdaLoRAMLPWithBase(nn.Module):
    """
    Like AdaLoRAMLP but includes an additional base transformation, the fused LoRA is added
    to base layer to produce final transformation

    :param feat_dim: feature dimension
    :param ada_dim: condition dimension
    :param inter_dim: intermediate dimension, if act_fn is not None, this is the dimension
                    of the hidden layer of the condition projection
    :param rank: rank of the low rank matrices
    :param lora_proj_act_fn: activation function for the hidden layer of the condition projection
    :param sparse_heads: number of heads to use for the sparse projection, set to None for dense projection
    :param lora_proj_act_fn: activation function to use between the two sets of A/B matrices
    :param residual: whether to add residual connection
    :param act_fn: activation function to use between the two sets of A/B matrices
    """

    def __init__(self,
                 feat_dim,
                 ada_dim,
                 inter_dim=None,
                 rank=8,
                 lora_proj_act_fn=nn.GELU,
                 act_fn=nn.GELU,
                 sparse_heads=None,
                 residual=True,
                    norm_cond=True
                 ):
        super().__init__()
        self.rank = rank
        layers = []
        self.base_up = nn.Parameter(torch.randn(feat_dim, feat_dim))
        self.base_down = nn.Parameter(torch.randn(feat_dim, feat_dim))
        inter_dim = ada_dim if inter_dim is None else inter_dim
        if lora_proj_act_fn is None:
            layers.append(nn.Linear(ada_dim, feat_dim * rank * 4))
        else:
            layers.append(nn.Linear(ada_dim, inter_dim))
            layers.append(lora_proj_act_fn())
            layers.append(
                AlmostMonarch(inter_dim, feat_dim * rank * 4, sparse_heads) if sparse_heads is not None else nn.Linear(
                    inter_dim, feat_dim * rank * 4))

        self.gen_weight = nn.Sequential(*layers)
        self.act_fn = act_fn()

        self.residual = lambda x, y: x + y if residual else y
        self.norm_cond = nn.LayerNorm(ada_dim) if norm_cond else nn.Identity()

    def forward(self, x, ada_emb):
        x_in = x
        B, D = x.shape[0], x.shape[-1]
        x_weights = self.gen_weight(self.norm_cond(ada_emb))
        x_a_1, x_b_1, x_a_2, x_b_2 = x_weights.chunk(4, dim=-1)
        x_a_1, x_b_1, x_a_2, x_b_2 = map(lambda t: t.reshape(-1, D, self.rank), (x_a_1, x_b_1, x_a_2, x_b_2))

        x_up = torch.einsum('bdr,brk->bdk', x_a_1, x_b_1.transpose(1, 2))
        x_down = torch.einsum('bdr,brk->bdk', x_a_2, x_b_2.transpose(1, 2))

        x_up = self.base_up[None, ...] + x_up
        x_down = self.base_down[None, ...] + x_down

        x = torch.einsum('bdl,bd->bl', x_down, x)
        x = self.act_fn(x)
        x = torch.einsum('bkl,bl->bk', x_up, x)

        return self.residual(x_in, x)