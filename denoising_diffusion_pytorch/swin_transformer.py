from functools import partial
from typing import Optional, Callable, List, Any

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from torchvision.ops.misc import MLP, Permute
from torchvision.ops.stochastic_depth import StochasticDepth
from torchvision.transforms._presets import ImageClassification, InterpolationMode
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import WeightsEnum, Weights
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param


__all__ = [
    "SwinTransformer",
    "Swin_T_Weights",
    "Swin_S_Weights",
    "Swin_B_Weights",
    "swin_t",
    "swin_s",
    "swin_b",
]


def _patch_merging_pad(x):
    H, W, _ = x.shape[-3:]
    x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
    return x


torch.fx.wrap("_patch_merging_pad")


class PatchMerging(nn.Module):
    """Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        _log_api_usage_once(self)
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): input tensor with expected layout of [..., H, W, C]
        Returns:
            Tensor with layout of [..., H/2, W/2, 2*C]
        """
        x = _patch_merging_pad(x)

        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)  # ... H/2 W/2 2*C
        return x


def shifted_window_attention(
    input: Tensor,
    qkv_weight: Tensor,
    proj_weight: Tensor,
    relative_position_bias: Tensor,
    window_size: List[int],
    num_heads: int,
    shift_size: List[int],
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias: Optional[Tensor] = None,
    proj_bias: Optional[Tensor] = None,
):
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        input (Tensor[N, H, W, C]): The input tensor or 4-dimensions.
        qkv_weight (Tensor[in_dim, out_dim]): The weight tensor of query, key, value.
        proj_weight (Tensor[out_dim, out_dim]): The weight tensor of projection.
        relative_position_bias (Tensor): The learned relative position bias added to attention.
        window_size (List[int]): Window size.
        num_heads (int): Number of attention heads.
        shift_size (List[int]): Shift size for shifted window attention.
        attention_dropout (float): Dropout ratio of attention weight. Default: 0.0.
        dropout (float): Dropout ratio of output. Default: 0.0.
        qkv_bias (Tensor[out_dim], optional): The bias tensor of query, key, value. Default: None.
        proj_bias (Tensor[out_dim], optional): The bias tensor of projection. Default: None.
    Returns:
        Tensor[N, H, W, C]: The output tensor after shifted window attention.
    """
    B, H, W, C = input.shape
    # pad feature maps to multiples of window size
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    x = F.pad(input, (0, 0, 0, pad_r, 0, pad_b))
    _, pad_H, pad_W, _ = x.shape

    # If window size is larger than feature size, there is no need to shift window
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0

    # cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

    # partition windows
    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
    x = x.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C

    # multi-head attention
    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = q * (C // num_heads) ** -0.5
    attn = q.matmul(k.transpose(-2, -1))
    # add relative position bias
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        # generate attention mask
        attn_mask = x.new_zeros((pad_H, pad_W))
        h_slices = ((0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None))
        w_slices = ((0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None))
        count = 0
        for h in h_slices:
            for w in w_slices:
                attn_mask[h[0] : h[1], w[0] : w[1]] = count
                count += 1
        attn_mask = attn_mask.view(pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1])
        attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, window_size[0] * window_size[1])
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))

    attn = F.softmax(attn, dim=-1)
    # attn = F.dropout(attn, p=attention_dropout)

    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
    x = F.linear(x, proj_weight, proj_bias)
    # x = F.dropout(x, p=dropout)

    # reverse windows
    x = x.view(B, pad_H // window_size[0], pad_W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

    # reverse cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

    # unpad features
    x = x[:, :H, :W, :].contiguous()
    return x


torch.fx.wrap("shifted_window_attention")


class ShiftedWindowAttention(nn.Module):
    """
    See :func:`shifted_window_attention`.
    """

    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError("window_size and shift_size must be of length 2")
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        #print(self.qkv.weight.shape, self.qkv.bias.shape)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        # coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords = torch.stack(torch.meshgrid(coords_h, coords_w))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).view(-1)  # Wh*Ww*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): Tensor with layout of [B, H, W, C]
        Returns:
            Tensor with same layout as input, i.e. [B, H, W, C]
        """

        N = self.window_size[0] * self.window_size[1]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index]  # type: ignore[index]
        relative_position_bias = relative_position_bias.view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        #print(self.qkv.weight.shape, self.proj.weight.shape)
        return shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
        )


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (List[int]): Window size.
        shift_size (List[int]): Shift size for shifted window attention.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
        norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm.
        attn_layer (nn.Module): Attention layer. Default: ShiftedWindowAttention
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: List[int],
        shift_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_layer: Callable[..., nn.Module] = ShiftedWindowAttention,
    ):
        super().__init__()
        _log_api_usage_once(self)

        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim,
            window_size,
            shift_size,
            num_heads,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: Tensor):
        x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        return x


class SwinTransformer(nn.Module):
    """
    Implements Swin Transformer from the `"Swin Transformer: Hierarchical Vision Transformer using
    Shifted Windows" <https://arxiv.org/pdf/2103.14030>`_ paper.
    Args:
        patch_size (List[int]): Patch size.
        embed_dim (int): Patch embedding dimension.
        depths (List(int)): Depth of each Swin Transformer layer.
        num_heads (List(int)): Number of attention heads in different layers.
        window_size (List[int]): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.0.
        num_classes (int): Number of classes for classification head. Default: 1000.
        block (nn.Module, optional): SwinTransformer Block. Default: None.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
    """

    def __init__(
        self,
        patch_size: List[int],
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        _log_api_usage_once(self)
        self.num_classes = num_classes

        if block is None:
            block = SwinTransformerBlock

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        layers: List[nn.Module] = []
        # split image into non-overlapping patches
        # layers.append(
        #     nn.Sequential(
        #         nn.Conv2d(
        #             3, embed_dim, kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1])
        #         ),
        #         Permute([0, 2, 3, 1]),
        #         norm_layer(embed_dim),
        #     )
        # )
        self.first_coonv = nn.Sequential(
                nn.Conv2d(
                    3, embed_dim, kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1])
                ),
                Permute([0, 2, 3, 1]),
                norm_layer(embed_dim),
            )

        total_stage_blocks = sum(depths)
        stage_block_id = 0
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim * 2 ** i_stage
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            # add patch merging layer
            if i_stage < (len(depths) - 1):
                layers.append(PatchMerging(dim, norm_layer))
        # self.features = nn.Sequential(*layers)
        self.features = nn.ModuleList(layers)

        num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        feats = []
        x = self.first_coonv(x)
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in [0, 2, 4, 6]:
                feats.append(x.permute(0, 3, 1, 2).contiguous())
        # x = self.features(x)
        # x = self.norm(x)
        # x = x.permute(0, 3, 1, 2)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.head(x)
        return feats


def _swin_transformer(
    patch_size: List[int],
    embed_dim: int,
    depths: List[int],
    num_heads: List[int],
    window_size: List[int],
    stochastic_depth_prob: float,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> SwinTransformer:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = SwinTransformer(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        stochastic_depth_prob=stochastic_depth_prob,
        **kwargs,
    )

    if weights is not None:
        ckpt1 = weights.get_state_dict(progress=progress)
        ckpt2 = model.state_dict()
        kl1 = list(ckpt1.keys())
        for i, k in enumerate(list(ckpt2.keys())):
            ckpt2[k] = ckpt1[kl1[i]]
        msg = model.load_state_dict(ckpt2, strict=False)
        print(f'Load swin_transformer: {msg}')

    return model


_COMMON_META = {
    "categories": _IMAGENET_CATEGORIES,
}


class Swin_T_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/swin_t-704ceda3.pth",
        transforms=partial(
            ImageClassification, crop_size=224, resize_size=232, interpolation=InterpolationMode.BICUBIC
        ),
        meta={
            **_COMMON_META,
            "num_params": 28288354,
            "min_size": (224, 224),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#swintransformer",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.474,
                    "acc@5": 95.776,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a similar training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


class Swin_S_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/swin_s-5e29d889.pth",
        transforms=partial(
            ImageClassification, crop_size=224, resize_size=246, interpolation=InterpolationMode.BICUBIC
        ),
        meta={
            **_COMMON_META,
            "num_params": 49606258,
            "min_size": (224, 224),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#swintransformer",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 83.196,
                    "acc@5": 96.360,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a similar training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


class Swin_B_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/swin_b-68c6b09e.pth",
        transforms=partial(
            ImageClassification, crop_size=224, resize_size=238, interpolation=InterpolationMode.BICUBIC
        ),
        meta={
            **_COMMON_META,
            "num_params": 87768224,
            "min_size": (224, 224),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#swintransformer",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 83.582,
                    "acc@5": 96.640,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a similar training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


def swin_t(*, weights: Optional[Swin_T_Weights] = None, progress: bool = True, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_tiny architecture from
    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows <https://arxiv.org/pdf/2103.14030>`_.

    Args:
        weights (:class:`~torchvision.models.Swin_T_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Swin_T_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.Swin_T_Weights
        :members:
    """
    weights = Swin_T_Weights.verify(weights)

    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        stochastic_depth_prob=0.2,
        weights=weights,
        progress=progress,
        **kwargs,
    )


def swin_s(*, weights: Optional[Swin_S_Weights] = None, progress: bool = True, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_small architecture from
    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows <https://arxiv.org/pdf/2103.14030>`_.

    Args:
        weights (:class:`~torchvision.models.Swin_S_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Swin_S_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.Swin_S_Weights
        :members:
    """
    weights = Swin_S_Weights.verify(weights)

    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        stochastic_depth_prob=0.3,
        weights=weights,
        progress=progress,
        **kwargs,
    )


from torchvision.models._utils import handle_legacy_interface
@handle_legacy_interface(weights=("pretrained", Swin_B_Weights.IMAGENET1K_V1))
def swin_b(*, weights: Optional[Swin_B_Weights] = None, progress: bool = True, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_base architecture from
    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows <https://arxiv.org/pdf/2103.14030>`_.

    Args:
        weights (:class:`~torchvision.models.Swin_B_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Swin_B_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.Swin_B_Weights
        :members:
    """
    weights = Swin_B_Weights.verify(weights)

    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[7, 7],
        stochastic_depth_prob=0.5,
        weights=weights,
        progress=progress,
        **kwargs,
    )

if __name__ == '__main__':
    model = swin_b(weights=Swin_B_Weights)
    x = torch.rand(1, 3, 320, 320)
    y = model(x)
    pause = 0