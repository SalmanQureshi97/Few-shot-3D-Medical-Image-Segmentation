from __future__ import annotations

from typing import Dict, List, Sequence, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint


def norm_layer(kind: str, channels: int) -> nn.Module:
    kind = kind.lower()
    if kind == "batch":
        return nn.BatchNorm3d(channels)
    if kind == "group":
        groups = 8 if channels >= 8 else 1
        return nn.GroupNorm(groups, channels)
    if kind == "instance":
        return nn.InstanceNorm3d(channels, affine=True)
    raise ValueError(f"Unknown norm: {kind}")


def match_spatial(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pad/crop source so its spatial shape matches target.

    Used to reconcile encoder skip features with the upsampled decoder feature
    when patch shapes are not divisible by 2^levels.
    """
    src = source.shape[-3:]
    dst = target.shape[-3:]
    if src == dst:
        return source
    out = source
    pads: List[int] = []
    for s, d in reversed(list(zip(src, dst))):
        delta = d - s
        before = max(0, delta // 2)
        after = max(0, delta - before)
        pads.extend([before, after])
    if any(pads):
        out = F.pad(out, pads)
    slices = []
    for s, d in zip(out.shape[-3:], dst):
        start = max(0, (s - d) // 2)
        slices.append(slice(start, start + d))
    return out[(...,) + tuple(slices)]


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str = "instance", dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(norm, out_channels),
            nn.LeakyReLU(0.01, inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout3d(dropout))
        layers.extend(
            [
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                norm_layer(norm, out_channels),
                nn.LeakyReLU(0.01, inplace=True),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str = "instance", dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = norm_layer(norm, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = norm_layer(norm, out_channels)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))
        return self.act(out + residual)


class SEBlock3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, hidden, kernel_size=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class AttentionGate3D(nn.Module):
    def __init__(self, skip_channels: int, gate_channels: int, inter_channels: int):
        super().__init__()
        self.theta = nn.Conv3d(skip_channels, inter_channels, kernel_size=1, bias=False)
        self.phi = nn.Conv3d(gate_channels, inter_channels, kernel_size=1, bias=False)
        self.psi = nn.Sequential(
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(inter_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        gate = match_spatial(gate, skip)
        attention = self.psi(self.theta(skip) + self.phi(gate))
        return skip * attention


def _maybe_checkpoint(fn, x, enabled: bool):
    if enabled and x.requires_grad:
        return gradient_checkpoint(fn, x, use_reentrant=False)
    return fn(x)


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 16,
        levels: int = 4,
        norm: str = "instance",
        dropout: float = 0.0,
        deep_supervision: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.gradient_checkpointing = gradient_checkpointing
        channels = [base_channels * (2**i) for i in range(levels)]
        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(ConvBlock(prev, ch, norm, dropout))
            prev = ch
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = ConvBlock(channels[-1], channels[-1] * 2, norm, dropout)
        bottleneck_channels = channels[-1] * 2
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        decoder_channels = list(reversed(channels))
        prev = bottleneck_channels
        for ch in decoder_channels:
            self.upconvs.append(nn.ConvTranspose3d(prev, ch, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(ch * 2, ch, norm, dropout))
            prev = ch
        self.head = nn.Conv3d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []
        out = x
        for encoder in self.encoders:
            out = _maybe_checkpoint(encoder, out, self.gradient_checkpointing)
            skips.append(out)
            out = self.pool(out)
        out = _maybe_checkpoint(self.bottleneck, out, self.gradient_checkpointing)
        for up, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            out = up(out)
            skip = match_spatial(skip, out)
            cat = torch.cat([out, skip], dim=1)
            out = _maybe_checkpoint(decoder, cat, self.gradient_checkpointing)
        return self.head(out)


class ResidualAttentionUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 16,
        levels: int = 4,
        norm: str = "instance",
        dropout: float = 0.1,
        deep_supervision: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.gradient_checkpointing = gradient_checkpointing
        channels = [base_channels * (2**i) for i in range(levels)]
        self.encoders = nn.ModuleList()
        self.se_blocks = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(ResidualBlock(prev, ch, norm, dropout))
            self.se_blocks.append(SEBlock3D(ch))
            prev = ch
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = ResidualBlock(channels[-1], channels[-1] * 2, norm, dropout)

        self.upconvs = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ds_heads = nn.ModuleList()
        prev = channels[-1] * 2
        for ch in reversed(channels):
            self.upconvs.append(nn.ConvTranspose3d(prev, ch, kernel_size=2, stride=2))
            self.attentions.append(AttentionGate3D(ch, ch, max(1, ch // 2)))
            self.decoders.append(ResidualBlock(ch * 2, ch, norm, dropout))
            self.ds_heads.append(nn.Conv3d(ch, out_channels, kernel_size=1))
            prev = ch
        self.head = nn.Conv3d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        skips: List[torch.Tensor] = []
        out = x
        for encoder, se in zip(self.encoders, self.se_blocks):
            out = _maybe_checkpoint(encoder, out, self.gradient_checkpointing)
            out = se(out)
            skips.append(out)
            out = self.pool(out)
        out = _maybe_checkpoint(self.bottleneck, out, self.gradient_checkpointing)

        outputs: List[torch.Tensor] = []
        for up, attn, decoder, ds_head, skip in zip(
            self.upconvs, self.attentions, self.decoders, self.ds_heads, reversed(skips)
        ):
            out = up(out)
            skip = attn(match_spatial(skip, out), out)
            skip = match_spatial(skip, out)
            cat = torch.cat([out, skip], dim=1)
            out = _maybe_checkpoint(decoder, cat, self.gradient_checkpointing)
            if self.deep_supervision:
                outputs.append(ds_head(out))
        logits = self.head(out)
        if not self.deep_supervision:
            return logits
        return [logits] + outputs[:-1]


class _SwinUNETRWrapper(nn.Module):
    """Lazy wrapper around MONAI's SwinUNETR.

    MONAI is an optional dependency: importing it only when this model is
    requested keeps the default install slim. Installation:
    ``pip install -r requirements-monai.txt``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        feature_size: int = 48,
        patch_size: Sequence[int] = (32, 96, 96),
        use_checkpoint: bool = False,
    ):
        super().__init__()
        try:
            from monai.networks.nets import SwinUNETR  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "MONAI is required for the swin_unetr architecture. "
                "Install with `pip install -r requirements-monai.txt`."
            ) from exc
        try:
            self.net = SwinUNETR(
                img_size=tuple(int(s) for s in patch_size),
                in_channels=int(in_channels),
                out_channels=int(out_channels),
                feature_size=int(feature_size),
                use_checkpoint=bool(use_checkpoint),
            )
        except TypeError:
            # Newer MONAI dropped the `img_size` arg in favour of inferring from input.
            self.net = SwinUNETR(
                in_channels=int(in_channels),
                out_channels=int(out_channels),
                feature_size=int(feature_size),
                use_checkpoint=bool(use_checkpoint),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(cfg: Dict) -> nn.Module:
    name = cfg.get("name", "unet3d").lower()
    common = {
        "in_channels": int(cfg["in_channels"]),
        "out_channels": int(cfg["out_channels"]),
        "base_channels": int(cfg.get("base_channels", 16)),
        "levels": int(cfg.get("levels", 4)),
        "norm": cfg.get("norm", "instance"),
        "dropout": float(cfg.get("dropout", 0.0)),
        "deep_supervision": bool(cfg.get("deep_supervision", False)),
        "gradient_checkpointing": bool(cfg.get("gradient_checkpointing", False)),
    }
    if name == "unet3d":
        return UNet3D(**common)
    if name in {"resattn_unet3d", "residual_attention_unet3d"}:
        return ResidualAttentionUNet3D(**common)
    if name in {"swin_unetr", "swinunetr"}:
        return _SwinUNETRWrapper(
            in_channels=common["in_channels"],
            out_channels=common["out_channels"],
            feature_size=int(cfg.get("feature_size", 48)),
            patch_size=cfg.get("patch_size", (32, 96, 96)),
            use_checkpoint=common["gradient_checkpointing"],
        )
    raise ValueError(f"Unknown model name: {name}")
