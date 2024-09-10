import torch
from torch import nn
import torch.nn.functional as F
from .sparse_layers import AlmostMonarch


class AdaLoRA(nn.Module):
    """
    Feels like an AdaNorm kinda layer, but condition is projected into a LoRA which hidden states are pushed through

    shout out to rami_mmo for the initial design and idea!
    """

    def __init__(self, feat_dim, ada_dim, inter_dim=None, rank=8, act_fn=nn.GELU, sparse_heads=None):
        super().__init__()
        self.rank = rank
        layers = []
        inter_dim = ada_dim if inter_dim is None else inter_dim
        if act_fn is None:
            layers.append(nn.Linear(ada_dim, feat_dim * rank * 2))
        else:
            layers.append(nn.Linear(ada_dim, inter_dim))
            layers.append(act_fn())
            layers.append(
                AlmostMonarch(inter_dim, feat_dim * rank * 2, sparse_heads) if sparse_heads is not None else nn.Linear(
                    inter_dim, feat_dim * rank * 2))

        self.gen_weight = nn.Sequential(*layers)

    def forward(self, x, ada_emb):
        x_in = x
        B, D = x.shape[0], x.shape[-1]
        x_weights = self.gen_weight(ada_emb)
        x_a, x_b = x_weights.chunk(2, dim=-1)
        x_a, x_b = map(lambda t: t.reshape(-1, D, self.rank), (x_a, x_b))

        x = torch.einsum('bc,bco->bo', x, x_a)
        x = torch.einsum('bc,bco->bo', x, x_b.permute(0, 2, 1))

        # add back residual otherwise we kill the rank of our representation
        return x + x_in


class AdaLoRAMLP(nn.Module):
    """
    Feels like an AdaNorm kinda layer, but condition is projected into a LoRA which hidden states are pushed through
    """

    def __init__(self, feat_dim, ada_dim, inter_dim=None, rank=8, act_fn=nn.GELU, sparse_heads=None):
        super().__init__()
        self.rank = rank
        layers = []
        inter_dim = ada_dim if inter_dim is None else inter_dim
        if act_fn is None:
            layers.append(nn.Linear(ada_dim, feat_dim * rank * 4))
        else:
            layers.append(nn.Linear(ada_dim, inter_dim))
            layers.append(act_fn())
            layers.append(
                AlmostMonarch(inter_dim, feat_dim * rank * 4, sparse_heads) if sparse_heads is not None else nn.Linear(
                    inter_dim, feat_dim * rank * 4))

        self.gen_weight = nn.Sequential(*layers)
        self.act_fn = act_fn()

    def forward(self, x, ada_emb):
        x_in = x
        B, D = x.shape[0], x.shape[-1]
        x_weights = self.gen_weight(ada_emb)
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
    Feels like an AdaNorm kinda layer, but condition is projected into a LoRA which hidden states are pushed through
    """

    def __init__(self, feat_dim, ada_dim, inter_dim=None, rank=8, act_fn=nn.GELU, sparse_heads=None, residual=True):
        super().__init__()
        self.rank = rank
        self.base_layer = nn.Parameter(torch.randn(feat_dim, feat_dim))
        layers = []
        inter_dim = ada_dim if inter_dim is None else inter_dim
        if act_fn is None:
            layers.append(nn.Linear(ada_dim, feat_dim * rank * 2))
        else:
            layers.append(nn.Linear(ada_dim, inter_dim))
            layers.append(act_fn())
            layers.append(
                AlmostMonarch(inter_dim, feat_dim * rank * 2, sparse_heads) if sparse_heads is not None else nn.Linear(
                    inter_dim, feat_dim * rank * 2))

        self.gen_weight = nn.Sequential(*layers)

        self.residual = lambda x, y: x + y if residual else y

    def forward(self, x, ada_emb):
        x_in = x
        B, D = x.shape[0], x.shape[-1]
        x_weights = self.gen_weight(ada_emb)
        x_a, x_b = x_weights.chunk(2, dim=-1)
        x_a, x_b = map(lambda t: t.reshape(-1, D, self.rank), (x_a, x_b))

        # fuse
        layer = torch.einsum('bdr,brk->bdk', x_a, x_b.transpose(1, 2))
        layer = self.base_layer[None, ...] + layer

        x = torch.einsum('bc,bco->bo', x, layer)

        return self.residual(x_in, x)


class AdaLoRAMLPWithBase(nn.Module):
    """
    Feels like an AdaNorm kinda layer, but condition is projected into a LoRA which hidden states are pushed through
    """

    def __init__(self, feat_dim, ada_dim, inter_dim=None, rank=8, act_fn=nn.GELU, sparse_heads=None, residual=True):
        super().__init__()
        self.rank = rank
        layers = []
        self.base_up = nn.Parameter(torch.randn(feat_dim, feat_dim))
        self.base_down = nn.Parameter(torch.randn(feat_dim, feat_dim))
        inter_dim = ada_dim if inter_dim is None else inter_dim
        if act_fn is None:
            layers.append(nn.Linear(ada_dim, feat_dim * rank * 4))
        else:
            layers.append(nn.Linear(ada_dim, inter_dim))
            layers.append(act_fn())
            layers.append(
                AlmostMonarch(inter_dim, feat_dim * rank * 4, sparse_heads) if sparse_heads is not None else nn.Linear(
                    inter_dim, feat_dim * rank * 4))

        self.gen_weight = nn.Sequential(*layers)
        self.act_fn = act_fn()

        self.residual = lambda x, y: x + y if residual else y

    def forward(self, x, ada_emb):
        x_in = x
        B, D = x.shape[0], x.shape[-1]
        x_weights = self.gen_weight(ada_emb)
        x_a_1, x_b_1, x_a_2, x_b_2 = x_weights.chunk(4, dim=-1)
        x_a_1, x_b_1, x_a_2, x_b_2 = map(lambda t: t.reshape(-1, D, self.rank), (x_a_1, x_b_1, x_a_2, x_b_2))

        x_up = torch.einsum('bdr,brk->bdk', x_a_1, x_b_1.transpose(1, 2))
        x_down = torch.einsum('bdr,brk->bdk', x_a_2, x_b_2.transpose(1, 2))

        x_up = self.base_up[None, ...] + x_up
        x_down = self.base_down[None, ...] + x_down

        x = torch.einsum('bdl,bd->bl', x_down, x)
        x = torch.einsum('bkl,bl->bk', x_up, x)

        return self.residual(x_in, x)