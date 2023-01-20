# -*- coding: utf-8 -*-
# @Author  : Lan Zhang
# @Time    : 2022/12/1 20:55
# @File    : Layer.py
# @Software: PyCharm
from Block import *
import torch.utils.checkpoint as checkpoint


class Layer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, has_msa=True, local_ws=1, alpha=0.5):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        block = Block if has_msa else ConvFFNBlock
        self.blocks = nn.ModuleList([
            block(dim=dim, input_resolution=input_resolution,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop, attn_drop=attn_drop,
                  drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                  norm_layer=norm_layer, local_ws=local_ws, alpha=alpha)
            for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops
