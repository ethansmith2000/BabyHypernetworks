"""
based on https://github.com/ethansmith2000/SparseNetworks
"""

import torch
from torch import nn
import math
import torch.nn.functional as F


class Permute(nn.Module):

    def __init__(self,
                 full_dim,
                 heads,
                 mode="random",  # random, roll, chunk_random
                 roll=0.4,
                 chunks=4,  # must divide the chunk dim evenly
                 ):
        super().__init__()
        dim = full_dim // heads
        roll = int(roll * full_dim)
        if mode == "random":
            permute = torch.randperm(full_dim)
        elif mode == "roll":
            permute = torch.roll(torch.arange(full_dim), roll)
        elif mode == "chunk_random":
            assert dim % chunks == 0, "chunks must divide the dim evenly"
            chunk_indices = torch.randperm(full_dim // (dim // chunks))
            permute = torch.cat([torch.arange((dim // chunks)) + i * (dim // chunks) for i in chunk_indices])
        else:
            raise NotImplementedError("mode not implemented")
        self.register_buffer("permute", permute)

    def forward(self, x):
        return x[:, self.permute]


class Unpermute(nn.Module):

    def __init__(self, indices):
        super().__init__()
        perm_matrix = F.one_hot(indices, num_classes=indices.shape[0]).float()
        unperm_matrix = perm_matrix.inverse()
        unperm = unperm_matrix.argmax(dim=-1).long()
        self.register_buffer("unperm", unperm)

    def forward(self, x):
        return x[:, self.unperm]


class BiasAdd(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x + self.bias


class BlockwiseDiagLinear(nn.Module):

    """
    Sparse linear layer that spares parameters by only maintaining blockwise diagonal matrices

    :param full_in_dim: input dimension
    :param full_out_dim: output dimension
    :param heads: number of blockwise diagonal chunks to divide features along
    :param bias: whether to add a bias term
    """

    def __init__(self, full_in_dim=1024, full_out_dim=1024, heads=8, bias=True):
        super(BlockwiseDiagLinear, self).__init__()
        self.full_in = full_in_dim
        self.full_out = full_out_dim
        self.in_dim = full_in_dim // heads
        self.out_dim = full_out_dim // heads
        self.h = heads

        # weights init
        weights = [torch.randn(self.in_dim, self.out_dim) for _ in range(heads)]
        for i in range(len(weights)):
            torch.nn.init.xavier_uniform_(weights[i])
        self.weight = nn.Parameter(torch.stack(weights, dim=0))

        # optionals
        self.bias_add = BiasAdd(self.full_out) if bias else nn.Identity()

    def extra_repr(self):
        return f"full_in={self.full_in}, full_out={self.full_out}, heads={self.h}, total_params={self.full_in * self.full_out / self.h}"

    def forward(self, x):
        b, h, in_dim = x.shape[0], self.h, self.in_dim
        x = x.reshape(b, h, in_dim)
        x = torch.einsum('bhd,hdl->bhl', x, self.weight)
        x = x.reshape(b, h * self.out_dim)
        x = self.permute(x)
        return x


class AlmostMonarch(nn.Module):
    """
    Kinda looks like monarch matrices but not really, uses a random permutation instead of the one described in the paper

    :param full_in_dim: input dimension
    :param full_out_dim: output dimension
    :param heads: number of blockwise diagonal chunks to divide features along
    :param heads_second: number of blockwise diagonal chunks to divide features along in the second layer
    :param bias: whether to add a bias term
    :param permute_mode: how to permute the features, options are random, roll, chunk
    """

    def __init__(self,
                 full_in_dim=1024,
                 full_out_dim=1024,
                 heads=8,
                 heads_second=None,
                 bias=True,
                 permute_mode="random"
                 ):
        super().__init__()
        # so we have layer -> permute -> layer
        heads_second = heads if heads_second is None else heads_second
        self.lin1 = BlockwiseDiagLinear(full_in_dim, full_out_dim, heads, bias=False)
        self.permute = Permute(full_out_dim, heads, mode=permute_mode) if permute_mode is not None else nn.Identity()
        self.lin2 = BlockwiseDiagLinear(full_out_dim, full_out_dim, heads_second, bias)

    def forward(self, x):
        return self.lin2(self.permute(self.lin1(x)))


class MothMatrix(nn.Module):
    """
    half the parameters of a monarch matrix
    shuffles features, does a computation, then unshuffles as to respect residual stream
    """

    def __init__(self,
                 full_in_dim=1024,
                 full_out_dim=1024,
                 heads=8,
                 bias=True,
                 permute_mode_in="random",
                 permute_mode_out="unpermute"
                 ):
        super().__init__()
        self.permute_in = Permute(full_in_dim, heads, mode=permute_mode_in) if permute_mode_in is not None else nn.Identity()
        self.lin = BlockwiseDiagLinear(full_in_dim, full_out_dim, heads, bias=bias)

        if permute_mode_out == "unpermute":
            self.permute_out = Unpermute(self.permute_in.permute)
        else:
            self.permute_out = Permute(full_out_dim, heads, mode=permute_mode_out) if permute_mode_out is not None else nn.Identity()

    def forward(self, x):
        return self.permute_out(self.lin(self.permute_in(x)))