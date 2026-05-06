from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Union

import torch
from torch import nn
from torch.nn import functional as F


def one_hot(
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = target != ignore_index
    safe_target = target.clone()
    safe_target[~valid] = 0
    encoded = F.one_hot(safe_target.long(), num_classes=num_classes)
    encoded = encoded.permute(0, 4, 1, 2, 3).float()
    return encoded, valid.unsqueeze(1)


class DiceLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        include_background: bool = False,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.include_background = include_background
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        target_oh, valid = one_hot(target, self.num_classes, self.ignore_index)
        probs = probs * valid
        target_oh = target_oh * valid
        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * target_oh, dims)
        denominator = torch.sum(probs + target_oh, dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        if not self.include_background and dice.numel() > 1:
            dice = dice[1:]
        return 1.0 - dice.mean()


class TverskyLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        alpha: float = 0.3,
        beta: float = 0.7,
        include_background: bool = False,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.beta = beta
        self.include_background = include_background
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        target_oh, valid = one_hot(target, self.num_classes, self.ignore_index)
        probs = probs * valid
        target_oh = target_oh * valid
        dims = (0, 2, 3, 4)
        tp = torch.sum(probs * target_oh, dims)
        fp = torch.sum(probs * (1.0 - target_oh) * valid, dims)
        fn = torch.sum((1.0 - probs) * target_oh, dims)
        score = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        if not self.include_background and score.numel() > 1:
            score = score[1:]
        return 1.0 - score.mean()


class DiceCELoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
    ):
        super().__init__()
        self.dice = DiceLoss(num_classes, ignore_index)
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, target) + self.ce_weight * self.ce(logits, target)


def _as_output_list(outputs: Union[torch.Tensor, Sequence[torch.Tensor]]) -> List[torch.Tensor]:
    if isinstance(outputs, torch.Tensor):
        return [outputs]
    return list(outputs)


class DeepSupervisionLoss(nn.Module):
    def __init__(self, base_loss: nn.Module, weights: Optional[Iterable[float]] = None):
        super().__init__()
        self.base_loss = base_loss
        self.weights = list(weights) if weights is not None else None

    def forward(self, outputs: Union[torch.Tensor, Sequence[torch.Tensor]], target: torch.Tensor) -> torch.Tensor:
        output_list = _as_output_list(outputs)
        if self.weights is None:
            weights = [0.5**i for i in range(len(output_list))]
        else:
            weights = self.weights[: len(output_list)]
        weight_sum = sum(weights)
        loss = 0.0
        for output, weight in zip(output_list, weights):
            if output.shape[-3:] != target.shape[-3:]:
                output = F.interpolate(output, size=target.shape[-3:], mode="trilinear", align_corners=False)
            loss = loss + (weight / weight_sum) * self.base_loss(output, target)
        return loss


def build_loss(cfg: Dict, num_classes: int, ignore_index: int) -> nn.Module:
    name = cfg.get("loss", "dice").lower()
    if name == "dice":
        base: nn.Module = DiceLoss(num_classes, ignore_index)
    elif name == "dice_ce":
        base = DiceCELoss(
            num_classes,
            ignore_index,
            float(cfg.get("dice_weight", 0.6)),
            float(cfg.get("ce_weight", 0.4)),
        )
    elif name == "tversky":
        base = TverskyLoss(num_classes, ignore_index)
    else:
        raise ValueError(f"Unknown loss: {name}")
    return DeepSupervisionLoss(base)
