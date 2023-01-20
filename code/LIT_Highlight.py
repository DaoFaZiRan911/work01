# -*- coding: utf-8 -*-
# @Author  : Lan Zhang
# @Time    : 2022/12/1 21:24
# @File    : LIT_Highlight.py
# @Software: PyCharm
from patch_embed import *
from Layer import *


def flops_to_string(flops, units='G', precision=2):
    if units.startswith('G'):
        return str(round(flops / 10. ** 9, precision)) + ' ' + units
    elif units.startswith('M'):
        return str(round(flops / 10. ** 6, precision)) + ' ' + units
    elif units.startswith('K'):
        return str(round(flops / 10. ** 3, precision)) + ' ' + units
    else:
        return str(flops) + units


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class LitHighlight(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=10575,
                 embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.5, norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, has_msa=[0, 0, 1, 1], alpha=0.9, local_ws=[0, 0, 2, 1],  **kwargs):
        super(LitHighlight, self).__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.has_msa = has_msa
        self.local_ws = local_ws

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.dp = nn.Dropout(0.3)

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = Layer(dim=int(embed_dim * 2 ** i_layer),
                          input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                            patches_resolution[1] // (2 ** i_layer)),
                          depth=depths[i_layer],
                          num_heads=num_heads[i_layer],
                          mlp_ratio=self.mlp_ratio,
                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                          drop=drop_rate, attn_drop=attn_drop_rate,
                          drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                          norm_layer=norm_layer,
                          downsample=DTM if (i_layer < self.num_layers - 1) else None,
                          use_checkpoint=use_checkpoint, has_msa=self.has_msa[i_layer] == 1,
                          local_ws=self.local_ws[i_layer], alpha=alpha)
            self.layers.append(layer)
            self.norm = norm_layer(self.num_features)
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
            self.apply(_init_weights)

    def forward_features(self, x):
        x = self.patch_embed(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.dp(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            temp = layer.flops()
            flops += temp
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
