"""
Modified SAM Image Encoder with Multi-scale Feature Extraction
支持提取 Layer 1, 3, 6, 9, 12 的中间特征

修改点:
1. forward_multiscale 方法提取指定层的中间特征
2. 特征提取层可配置
3. 兼容原始SAM的forward方法
"""

import math
from typing import Optional, Tuple, Type, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class Adapter(nn.Module):
    """轻量级适配器模块"""

    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int, mlp_dim: int, act: Type[nn.Module] = nn.GELU):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class Attention(nn.Module):
    """Multi-head Attention with relative position embeddings"""

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = self._add_rel_pos(attn, q, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)
        return x

    def _add_rel_pos(self, attn, q, q_size, k_size):
        """Add relative positional embeddings"""
        q_h, q_w = q_size
        k_h, k_w = k_size

        Rh = self._get_rel_pos(q_h, k_h, self.rel_pos_h)
        Rw = self._get_rel_pos(q_w, k_w, self.rel_pos_w)

        B, _, dim = q.shape
        r_q = q.reshape(B, q_h, q_w, dim)
        rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
        rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

        attn = (
                attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
        ).view(B, q_h * q_w, k_h * k_w)

        return attn

    def _get_rel_pos(self, q_size, k_size, rel_pos):
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        if rel_pos.shape[0] != max_rel_dist:
            rel_pos_resized = F.interpolate(
                rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
                size=max_rel_dist,
                mode="linear",
            )
            rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
        else:
            rel_pos_resized = rel_pos

        q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
        k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

        return rel_pos_resized[relative_coords.long()]


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows: torch.Tensor, window_size: int,
                       pad_hw: Tuple[int, int], hw: Tuple[int, int]) -> torch.Tensor:
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


class Block(nn.Module):
    """原始Transformer Block (无Adapter)"""

    def __init__(
            self,
            args,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos, rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class AdapterBlock(nn.Module):
    """带Adapter的Transformer Block"""

    def __init__(
            self,
            args,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            scale: float = 0.5,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.args = args
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos, rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        # Adapters
        self.MLP_Adapter = Adapter(dim, skip_connect=False)
        self.Space_Adapter = Adapter(dim, skip_connect=True)
        self.scale = scale

        # 深度Adapter (用于3D)
        if hasattr(args, 'thd') and args.thd:
            self.Depth_Adapter = Adapter(dim, skip_connect=False)
        else:
            self.Depth_Adapter = None

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        # 3D分支 (可选)
        if self.Depth_Adapter is not None and self.args.thd:
            hh, ww = x.shape[1], x.shape[2]
            depth = self.args.chunk
            xd = rearrange(x, '(b d) h w c -> (b h w) d c', d=depth)
            xd = self.norm1(xd)
            dh, dw = self._closest_numbers(depth)
            xd = rearrange(xd, 'bhw (dh dw) c -> bhw dh dw c', dh=dh)
            xd = self.Depth_Adapter(self.attn(xd))
            xd = rearrange(xd, '(b n) dh dw c -> (b dh dw) n c', n=hh * ww)

        x = self.norm1(x)
        x = self.attn(x)
        x = self.Space_Adapter(x)

        if self.Depth_Adapter is not None and self.args.thd:
            xd = rearrange(xd, 'b (hh ww) c -> b hh ww c', hh=hh)
            x = x + xd

        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        xn = self.norm2(x)
        x = x + self.mlp(xn) + self.scale * self.MLP_Adapter(xn)
        return x

    def _closest_numbers(self, target):
        a = int(target ** 0.5)
        b = a + 1
        while True:
            if a * b == target:
                return (a, b)
            elif a * b < target:
                b += 1
            else:
                a -= 1


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
            self,
            kernel_size: Tuple[int, int] = (16, 16),
            stride: Tuple[int, int] = (16, 16),
            padding: Tuple[int, int] = (0, 0),
            in_chans: int = 3,
            embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size,
                              stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)  # B C H W -> B H W C
        return x


class ImageEncoderViT(nn.Module):
    """
    SAM Image Encoder with Multi-scale Feature Extraction

    支持提取 Layer 1, 3, 6, 9, 12 的中间特征
    """

    def __init__(
            self,
            args,
            img_size: int = 1024,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            out_chans: int = 256,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_abs_pos: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            global_attn_indexes: Tuple[int, ...] = (),
            # 新增: 特征提取层配置
            feature_extract_layers: Tuple[int, ...] = (1, 3, 6, 9, 12),
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.args = args
        self.embed_dim = embed_dim
        self.depth = depth

        # 配置特征提取层
        self.feature_extract_layers = feature_extract_layers

        # Patch embedding
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=getattr(args, 'in_channels', in_chans),
            embed_dim=embed_dim,
        )

        # Position embedding
        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, 1024 // patch_size, 1024 // patch_size, embed_dim)
            )

        # Transformer blocks
        self.blocks = nn.ModuleList()
        block_class = AdapterBlock if getattr(args, 'mod', '') == 'sam_adpt' else Block

        for i in range(depth):
            block = block_class(
                args=args,
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        # Neck
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """原始forward方法，返回最终embedding"""
        x = self.patch_embed(x)

        if self.pos_embed is not None:
            new_abs_pos = F.interpolate(
                self.pos_embed.permute(0, 3, 1, 2),
                size=(x.shape[1], x.shape[2]),
                mode="bicubic",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            x = x + new_abs_pos

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))
        return x

    def forward_multiscale(self, x: torch.Tensor,
                           return_intermediate: bool = True,
                           custom_layers: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        """
        提取多尺度中间特征

        Args:
            x: 输入图像 [B, C, H, W]
            return_intermediate: 是否返回中间特征
            custom_layers: 自定义提取层 (覆盖默认配置)

        Returns:
            dict: {
                'embedding': 最终embedding [B, 256, H/16, W/16],
                'intermediate_features': {
                    'block1': [B, H/16, W/16, 768],
                    'block3': [B, H/16, W/16, 768],
                    ...
                }
            }
        """
        extract_layers = custom_layers if custom_layers else self.feature_extract_layers

        x = self.patch_embed(x)

        if self.pos_embed is not None:
            new_abs_pos = F.interpolate(
                self.pos_embed.permute(0, 3, 1, 2),
                size=(x.shape[1], x.shape[2]),
                mode="bicubic",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            x = x + new_abs_pos

        intermediate_features = {}

        for i, blk in enumerate(self.blocks):
            x = blk(x)

            block_idx = i + 1  # 1-indexed
            if return_intermediate and block_idx in extract_layers:
                # 保存中间特征 (clone避免被后续操作修改)
                intermediate_features[f'block{block_idx}'] = x.clone()

        # 最终embedding
        embedding = self.neck(x.permute(0, 3, 1, 2))

        return {
            'embedding': embedding,
            'intermediate_features': intermediate_features,
        }

    def get_feature_info(self) -> Dict:
        """获取特征信息 (用于调试)"""
        return {
            'embed_dim': self.embed_dim,
            'depth': self.depth,
            'extract_layers': self.feature_extract_layers,
            'feature_shape': f'[B, img_size//16, img_size//16, {self.embed_dim}]',
        }


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == '__main__':
    # 创建模拟args
    class Args:
        in_channels = 3
        mod = 'sam_adpt'  # 使用Adapter
        thd = False
        chunk = 1


    args = Args()

    # 创建Image Encoder
    encoder = ImageEncoderViT(
        args=args,
        img_size=256,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        out_chans=256,
        qkv_bias=True,
        use_abs_pos=True,
        use_rel_pos=False,
        window_size=14,
        global_attn_indexes=[2, 5, 8, 11],
        feature_extract_layers=(1, 3, 6, 9, 12),
    )

    print(f"Image Encoder created with {sum(p.numel() for p in encoder.parameters()):,} parameters")
    print(f"Feature info: {encoder.get_feature_info()}")

    # 测试forward
    x = torch.randn(2, 3, 256, 256)

    # 原始forward
    out = encoder(x)
    print(f"\nOriginal forward output shape: {out.shape}")

    # 多尺度forward
    result = encoder.forward_multiscale(x, return_intermediate=True)
    print(f"\nMulti-scale forward:")
    print(f"  Embedding shape: {result['embedding'].shape}")
    print(f"  Intermediate features:")
    for key, feat in result['intermediate_features'].items():
        print(f"    {key}: {feat.shape}")

    # Expected output:
    # Original forward output shape: torch.Size([2, 256, 16, 16])
    # Multi-scale forward:
    #   Embedding shape: torch.Size([2, 256, 16, 16])
    #   Intermediate features:
    #     block1: torch.Size([2, 16, 16, 768])
    #     block3: torch.Size([2, 16, 16, 768])
    #     block6: torch.Size([2, 16, 16, 768])
    #     block9: torch.Size([2, 16, 16, 768])
    #     block12: torch.Size([2, 16, 16, 768])

    print("\nTest passed!")