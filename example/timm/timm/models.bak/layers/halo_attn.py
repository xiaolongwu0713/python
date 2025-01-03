""" Halo Self Attention

Paper: `Scaling Local Self-Attention for Parameter Efficient Visual Backbones`
    - https://arxiv.org/abs/2103.12731

@misc{2103.12731,
Author = {Ashish Vaswani and Prajit Ramachandran and Aravind Srinivas and Niki Parmar and Blake Hechtman and
    Jonathon Shlens},
Title = {Scaling Local Self-Attention for Parameter Efficient Visual Backbones},
Year = {2021},
}

Status:
This impl is a WIP, there is no official ref impl and some details in paper weren't clear to me.
The attention mechanism works but it's slow as implemented.

Hacked together by / Copyright 2021 Ross Wightman
"""
from typing import Tuple, List

import torch
from torch import nn
import torch.nn.functional as F

from .weight_init import trunc_normal_


def rel_logits_1d(q, rel_k, permute_mask: List[int]):
    """ Compute relative logits along one dimension

    As per: https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2
    Originally from: `Attention Augmented Convolutional Networks` - https://arxiv.org/abs/1904.09925

    Args:
        q: (batch, height, width, dim)
        rel_k: (2 * window - 1, dim)
        permute_mask: permute output dim according to this
    """
    B, H, W, dim = q.shape
    rel_size = rel_k.shape[0]
    win_size = (rel_size + 1) // 2

    x = (q @ rel_k.transpose(-1, -2))
    x = x.reshape(-1, W, rel_size)

    # pad to shift from relative to absolute indexing
    x_pad = F.pad(x, [0, 1]).flatten(1)
    x_pad = F.pad(x_pad, [0, rel_size - W])

    # reshape and slice out the padded elements
    x_pad = x_pad.reshape(-1, W + 1, rel_size)
    x = x_pad[:, :W, win_size - 1:]

    # reshape and tile
    x = x.reshape(B, H, 1, W, win_size).expand(-1, -1, win_size, -1, -1)
    return x.permute(permute_mask)


class PosEmbedRel(nn.Module):
    """ Relative Position Embedding
    As per: https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2
    Originally from: `Attention Augmented Convolutional Networks` - https://arxiv.org/abs/1904.09925

    """
    def __init__(self, block_size, win_size, dim_head, scale):
        """
        Args:
            block_size (int): block size
            win_size (int): neighbourhood window size
            dim_head (int): attention head dim
            scale (float): scale factor (for init)
        """
        super().__init__()
        self.block_size = block_size
        self.dim_head = dim_head
        self.scale = scale
        self.height_rel = nn.Parameter(torch.randn(win_size * 2 - 1, dim_head) * self.scale)
        self.width_rel = nn.Parameter(torch.randn(win_size * 2 - 1, dim_head) * self.scale)

    def forward(self, q):
        B, BB, HW, _ = q.shape

        # relative logits in width dimension.
        q = q.reshape(-1, self.block_size, self.block_size, self.dim_head)
        rel_logits_w = rel_logits_1d(q, self.width_rel, permute_mask=(0, 1, 3, 2, 4))

        # relative logits in height dimension.
        q = q.transpose(1, 2)
        rel_logits_h = rel_logits_1d(q, self.height_rel, permute_mask=(0, 3, 1, 4, 2))

        rel_logits = rel_logits_h + rel_logits_w
        rel_logits = rel_logits.reshape(B, BB, HW, -1)
        return rel_logits


class HaloAttn(nn.Module):
    """ Halo Attention

    Paper: `Scaling Local Self-Attention for Parameter Efficient Visual Backbones`
        - https://arxiv.org/abs/2103.12731
    """
    def __init__(
            self, dim, dim_out=None, stride=1, num_heads=8, dim_head=None, block_size=8, halo_size=3, qkv_bias=False):
        super().__init__()
        dim_out = dim_out or dim
        assert dim_out % num_heads == 0
        self.stride = stride
        self.num_heads = num_heads
        self.dim_head = dim_head or dim // num_heads
        self.dim_qk = num_heads * self.dim_head
        self.dim_v = dim_out
        self.block_size = block_size
        self.halo_size = halo_size
        self.win_size = block_size + halo_size * 2  # neighbourhood window size
        self.scale = self.dim_head ** -0.5

        # FIXME not clear if this stride behaviour is what the paper intended
        # Also, the paper mentions using a 3D conv for dealing with the blocking/gather, and leaving
        # data in unfolded block form. I haven't wrapped my head around how that'd look.
        self.q = nn.Conv2d(dim, self.dim_qk, 1, stride=self.stride, bias=qkv_bias)
        self.kv = nn.Conv2d(dim, self.dim_qk + self.dim_v, 1, bias=qkv_bias)

        self.pos_embed = PosEmbedRel(
            block_size=block_size // self.stride, win_size=self.win_size, dim_head=self.dim_head, scale=self.scale)

        self.reset_parameters()

    def reset_parameters(self):
        std = self.q.weight.shape[1] ** -0.5  # fan-in
        trunc_normal_(self.q.weight, std=std)
        trunc_normal_(self.kv.weight, std=std)
        trunc_normal_(self.pos_embed.height_rel, std=self.scale)
        trunc_normal_(self.pos_embed.width_rel, std=self.scale)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.block_size == 0
        assert W % self.block_size == 0
        num_h_blocks = H // self.block_size
        num_w_blocks = W // self.block_size
        num_blocks = num_h_blocks * num_w_blocks
        bs_stride = self.block_size // self.stride

        q = self.q(x)
        # unfold
        q = q.reshape(-1, self.dim_head, num_h_blocks, bs_stride, num_w_blocks, bs_stride).permute(0, 1, 3, 5, 2, 4)
        # B, num_heads * dim_head * block_size ** 2, num_blocks
        q = q.reshape(B * self.num_heads, self.dim_head, -1, num_blocks).transpose(1, 3)
        # B * num_heads, num_blocks, block_size ** 2, dim_head

        kv = self.kv(x)
        # generate overlapping windows for kv
        kv = F.pad(kv, [self.halo_size, self.halo_size, self.halo_size, self.halo_size])
        kv = kv.unfold(2, self.win_size, self.block_size).unfold(3, self.win_size, self.block_size).reshape(
            B * self.num_heads, self.dim_head + (self.dim_v // self.num_heads), num_blocks, -1).permute(0, 2, 3, 1)
        # NOTE these two alternatives are equivalent, but above is the best balance of performance and clarity
        # if self.stride_tricks:
        #     kv = F.pad(kv, [self.halo_size, self.halo_size, self.halo_size, self.halo_size]).contiguous()
        #     kv = kv.as_strided((
        #         B, self.dim_qk + self.dim_v, self.win_size, self.win_size, num_h_blocks, num_w_blocks),
        #         stride=(kv.stride(0), kv.stride(1), kv.shape[-1], 1, self.block_size * kv.shape[-1], self.block_size))
        # else:
        #     kv = F.unfold(kv, kernel_size=self.win_size, stride=self.block_size, padding=self.halo_size)
        # kv = kv.reshape(
        #    B * self.num_heads, self.dim_head + (self.dim_v // self.num_heads), -1, num_blocks).transpose(1, 3)
        k, v = torch.split(kv, [self.dim_head, self.dim_v // self.num_heads], dim=-1)
        # B * num_heads, num_blocks, block_size ** 2, dim_head or dim_v // num_heads

        attn_logits = (q @ k.transpose(-1, -2)) * self.scale  # FIXME should usual attn scale be applied?
        attn_logits = attn_logits + self.pos_embed(q)  # B * num_heads, block_size ** 2, win_size ** 2

        attn_out = attn_logits.softmax(dim=-1)
        attn_out = (attn_out @ v).transpose(1, 3)  # B * num_heads, dim_v // num_heads, block_size ** 2, num_blocks

        # fold
        attn_out = attn_out.reshape(-1, bs_stride, bs_stride, num_h_blocks, num_w_blocks)
        attn_out = attn_out.permute(0, 3, 1, 4, 2).contiguous().view(B, self.dim_v, H // self.stride, W // self.stride)
        # B, dim_out, H // stride, W // stride
        return attn_out
