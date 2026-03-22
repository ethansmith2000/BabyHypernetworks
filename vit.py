import torch
from torch import nn
import torch.nn.functional as F
# from attentions import HyperAttentionMLP

class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(t.shape[0], t.shape[1], self.heads, -1).transpose(1, 2), (q, k, v))
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1)
        return self.to_out(attn)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, ff_hidden_dim, use_hyper_ff=False):
        super().__init__()
        self.attn = Attention(dim, heads)
        self.ff = HyperAttentionMLP(dim, heads=heads, ff_mult=ff_hidden_dim // dim) if use_hyper_ff else FeedForward(dim, ff_hidden_dim)
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        x = self.ff(self.norm2(x)) + x
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, dim, patch_size):
        super().__init__()
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        ph, pw = self.patch_size
        if H % ph != 0 or W % pw != 0:
            raise ValueError(f"Image size ({H}, {W}) must be divisible by patch size {self.patch_size}")
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        ff_mult,
        image_size,
        patch_size,
        num_classes,
        in_channels=3,
        use_cls_token=True,
        gradient_checkpointing=False,
        ff_mode="vanilla",
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(in_channels, dim, patch_size)

        image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        ph, pw = self.patch_embed.patch_size
        num_patches = (image_size[0] // ph) * (image_size[1] // pw)

        self.use_cls_token = use_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)) if use_cls_token else None
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches + (1 if use_cls_token else 0), dim))

        self.in_proj = nn.Sequential(
            nn.RMSNorm(dim),
            nn.Linear(dim, dim),
        )
        # ff_mode controls use of hyper MLPs:
        # "vanilla" -> all standard FF
        # "hyper" -> all HyperAttentionMLP
        # "alternate" -> HyperAttentionMLP on odd idx, standard on even idx
        ff_mode = ff_mode.lower()
        if ff_mode not in {"vanilla", "hyper", "alternate"}:
            raise ValueError("ff_mode must be one of {'vanilla', 'hyper', 'alternate'}")

        blocks = []
        for i in range(depth):
            use_hyper = ff_mode == "hyper" or (ff_mode == "alternate" and (i % 2 == 1))
            blocks.append(TransformerBlock(dim, heads, dim * ff_mult, use_hyper_ff=use_hyper))
        self.blocks = nn.ModuleList(blocks)
        self.out_proj = nn.Sequential(
            nn.RMSNorm(dim),
            nn.Linear(dim, num_classes),
        )
        self.gradient_checkpointing = gradient_checkpointing

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)
            elif isinstance(module, nn.Parameter):
                continue

        if self.cls_token is not None:
            nn.init.zeros_(self.cls_token)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, images, labels=None):
        x = self.patch_embed(images)

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.position_embedding[:, : x.shape[1]]
        x = self.in_proj(x)

        if self.gradient_checkpointing:
            for block in self.blocks:
                x = torch.utils.checkpoint.checkpoint(block, x)
        else:
            for block in self.blocks:
                x = block(x)

        x = self.out_proj(x[:, 0] if self.use_cls_token else x.mean(dim=1))

        if labels is not None:
            loss = nn.functional.cross_entropy(x, labels)
            return loss, x
        return x
