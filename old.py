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
            layers.append(AlmostMonarch(inter_dim, feat_dim * rank * 2, sparse_heads) if sparse_heads is not None else nn.Linear(inter_dim, feat_dim * rank * 2))

        self.gen_weight = nn.Sequential(*layers)

    def forward(self, x, ada_emb):
        x_in = x
        B, D = x.shape[0], x.shape[-1]
        x_weights = self.generate_weights(ada_emb)
        x_down, x_up = x_weights.chunk(2, dim=-1)
        x_down, x_up = map(lambda t: t.reshape(-1, D, self.rank), (x_down, x_up))

        x = torch.einsum('bc,bco->bo', x, x_down)
        x = torch.einsum('bc,bco->bo', x, x_up.permute(0, 2, 1))

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
        x_weights = self.generate_weights(ada_emb)
        x_down_1, x_up_1, x_down_2, x_up_2 = x_weights.chunk(4, dim=-1)
        x_down_1, x_up_1, x_down_2, x_up_2 = map(lambda t: t.reshape(-1, D, self.rank), (x_down_1, x_up_1, x_down_2, x_up_2))

        x = torch.einsum('bc,bco->bo', x, x_down_1)
        x = torch.einsum('bc,bco->bo', x, x_up_1.permute(0, 2, 1))
        x = self.act_fn(x)
        x = torch.einsum('bc,bco->bo', x, x_down_2)
        x = torch.einsum('bc,bco->bo', x, x_up_2.permute(0, 2, 1))

        # add back residual otherwise we kill the rank of our representation
        return x + x_in

