from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
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


def _as_class_weights(
    weights: Optional[Sequence[float]], num_classes: int, device: torch.device
) -> Optional[torch.Tensor]:
    if weights is None:
        return None
    arr = np.asarray(list(weights), dtype=np.float32)
    if arr.shape[0] != num_classes:
        raise ValueError(f"class_weights length {arr.shape[0]} != num_classes {num_classes}")
    return torch.from_numpy(arr).to(device)


class DiceLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        include_background: bool = False,
        smooth: float = 1e-5,
        class_weights: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.include_background = include_background
        self.smooth = smooth
        self._class_weights_cfg = list(class_weights) if class_weights is not None else None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        target_oh, valid = one_hot(target, self.num_classes, self.ignore_index)
        probs = probs * valid
        target_oh = target_oh * valid
        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * target_oh, dims)
        denominator = torch.sum(probs + target_oh, dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        weights = _as_class_weights(self._class_weights_cfg, self.num_classes, dice.device)
        if not self.include_background and dice.numel() > 1:
            dice = dice[1:]
            if weights is not None:
                weights = weights[1:]
        if weights is not None:
            return 1.0 - (dice * weights).sum() / weights.sum().clamp_min(1e-8)
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
        class_weights: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.beta = beta
        self.include_background = include_background
        self.smooth = smooth
        self._class_weights_cfg = list(class_weights) if class_weights is not None else None

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
        weights = _as_class_weights(self._class_weights_cfg, self.num_classes, score.device)
        if not self.include_background and score.numel() > 1:
            score = score[1:]
            if weights is not None:
                weights = weights[1:]
        if weights is not None:
            return 1.0 - (score * weights).sum() / weights.sum().clamp_min(1e-8)
        return 1.0 - score.mean()


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017) over logits."""

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        gamma: float = 2.0,
        class_weights: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.gamma = gamma
        self._class_weights_cfg = list(class_weights) if class_weights is not None else None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = _as_class_weights(self._class_weights_cfg, self.num_classes, logits.device)
        log_probs = F.log_softmax(logits, dim=1)
        nll = F.nll_loss(log_probs, target, weight=weight, ignore_index=self.ignore_index, reduction="none")
        with torch.no_grad():
            valid = target != self.ignore_index
            safe_target = target.clamp_min(0).clamp_max(self.num_classes - 1)
            pt = log_probs.gather(1, safe_target.unsqueeze(1)).squeeze(1).exp()
            modulator = (1.0 - pt).clamp_min(1e-6) ** self.gamma
        loss = nll * modulator
        loss = loss[valid]
        return loss.mean() if loss.numel() else logits.sum() * 0.0


class DiceCELoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        class_weights: Optional[Sequence[float]] = None,
        focal_gamma: float = 0.0,
    ):
        super().__init__()
        self.dice = DiceLoss(num_classes, ignore_index, class_weights=class_weights)
        self._class_weights_cfg = list(class_weights) if class_weights is not None else None
        self._focal_gamma = float(focal_gamma)
        if self._focal_gamma > 0:
            self.ce: nn.Module = FocalLoss(num_classes, ignore_index, focal_gamma, class_weights)
        else:
            weight = None
            if class_weights is not None:
                weight = torch.tensor(list(class_weights), dtype=torch.float32)
            self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.dice_weight = float(dice_weight)
        self.ce_weight = float(ce_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, target) + self.ce_weight * self.ce(logits, target)


class BoundaryLoss(nn.Module):
    """Signed-distance boundary loss (Kervadec et al., 2019).

    Penalises probability mass that lies far from the ground-truth boundary.
    Used as an auxiliary loss alongside Dice/CE; typical weight 0.1-0.5.
    """

    def __init__(self, num_classes: int, ignore_index: int = 255, include_background: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.include_background = include_background

    @staticmethod
    def _signed_distance(mask: torch.Tensor) -> torch.Tensor:
        """Approximate signed distance from class mask using avg-pool gradients.

        We avoid SciPy in the training loop: a 3D Sobel-like surface response
        gives a coarse but differentiable boundary proxy that is enough for
        the auxiliary loss to point predictions toward boundaries.
        """
        with torch.no_grad():
            kernel = torch.ones((1, 1, 3, 3, 3), device=mask.device, dtype=mask.dtype)
            blurred = F.conv3d(mask.unsqueeze(1), kernel, padding=1) / 27.0
            inside = blurred * mask.unsqueeze(1)
            outside = blurred * (1.0 - mask.unsqueeze(1))
            dist = (outside - inside).squeeze(1)
        return dist

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        target_oh, valid = one_hot(target, self.num_classes, self.ignore_index)
        loss_terms = []
        start = 0 if self.include_background else 1
        for c in range(start, self.num_classes):
            sd = self._signed_distance(target_oh[:, c])
            loss_terms.append((probs[:, c] * sd * valid.squeeze(1)).mean())
        if not loss_terms:
            return logits.sum() * 0.0
        return torch.stack(loss_terms).mean()


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
        loss: torch.Tensor = output_list[0].sum() * 0.0
        for output, weight in zip(output_list, weights):
            if output.shape[-3:] != target.shape[-3:]:
                output = F.interpolate(output, size=target.shape[-3:], mode="trilinear", align_corners=False)
            loss = loss + (weight / weight_sum) * self.base_loss(output, target)
        return loss


class CompoundLoss(nn.Module):
    """Sum of named loss modules with per-loss scalar weights."""

    def __init__(self, modules: Dict[str, nn.Module], weights: Dict[str, float]):
        super().__init__()
        self.losses = nn.ModuleDict(modules)
        self.weight_map = {k: float(v) for k, v in weights.items()}

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total: Optional[torch.Tensor] = None
        for name, module in self.losses.items():
            w = self.weight_map.get(name, 0.0)
            if w == 0.0:
                continue
            term = w * module(logits, target)
            total = term if total is None else total + term
        if total is None:
            return logits.sum() * 0.0
        return total


def build_loss(cfg: Dict, num_classes: int, ignore_index: int) -> nn.Module:
    """Build a (deeply-supervised) loss module from a training config block."""
    name = cfg.get("loss", "dice").lower()
    class_weights = cfg.get("class_weights")
    focal_gamma = float(cfg.get("focal_gamma", 0.0))
    boundary_weight = float(cfg.get("boundary_weight", 0.0))

    def make_primary() -> nn.Module:
        if name == "dice":
            return DiceLoss(num_classes, ignore_index, class_weights=class_weights)
        if name == "dice_ce":
            return DiceCELoss(
                num_classes,
                ignore_index,
                dice_weight=float(cfg.get("dice_weight", 0.6)),
                ce_weight=float(cfg.get("ce_weight", 0.4)),
                class_weights=class_weights,
                focal_gamma=focal_gamma,
            )
        if name == "tversky":
            return TverskyLoss(
                num_classes,
                ignore_index,
                alpha=float(cfg.get("tversky_alpha", 0.3)),
                beta=float(cfg.get("tversky_beta", 0.7)),
                class_weights=class_weights,
            )
        if name == "focal":
            return FocalLoss(num_classes, ignore_index, gamma=float(cfg.get("focal_gamma", 2.0)), class_weights=class_weights)
        raise ValueError(f"Unknown loss: {name}")

    primary = make_primary()
    if boundary_weight > 0:
        primary = CompoundLoss(
            {"primary": primary, "boundary": BoundaryLoss(num_classes, ignore_index)},
            {"primary": 1.0, "boundary": boundary_weight},
        )
    ds_weights = cfg.get("deep_supervision_weights")
    return DeepSupervisionLoss(primary, weights=ds_weights)
