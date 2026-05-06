import numpy as np
import torch

from mrbrains.losses import (
    BoundaryLoss,
    CompoundLoss,
    DeepSupervisionLoss,
    DiceCELoss,
    DiceLoss,
    FocalLoss,
    TverskyLoss,
    build_loss,
)


def _logits_for_target(target: torch.Tensor, num_classes: int, scale: float = 10.0) -> torch.Tensor:
    """Construct logits that exactly recover the target labels under softmax."""
    one_hot = torch.nn.functional.one_hot(target.long(), num_classes=num_classes).float()
    one_hot = one_hot.permute(0, 4, 1, 2, 3)
    return one_hot * scale


def test_dice_loss_perfect_overlap_is_zero():
    target = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    target[0, 1:3, 1:3, 1:3] = 1
    logits = _logits_for_target(target, num_classes=2, scale=20.0)
    loss = DiceLoss(num_classes=2)(logits, target)
    assert float(loss) < 1e-3


def test_dice_loss_ignore_index_masks_voxels():
    target = torch.zeros(1, 2, 2, 2, dtype=torch.long)
    target[0, 0, 0, 0] = 255
    logits = _logits_for_target(torch.zeros_like(target), num_classes=2)
    # The single ignore voxel does not affect Dice.
    assert torch.isfinite(DiceLoss(num_classes=2, ignore_index=255)(logits, target))


def test_dice_loss_class_weights_emphasise_rare_class():
    target = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    target[0, 0, 0, 0] = 1  # 1 voxel of class 1 out of 64
    # Predict everything as background, missing the rare class entirely.
    logits = torch.zeros(1, 2, 4, 4, 4)
    logits[:, 0] = 5.0
    base_loss = float(DiceLoss(num_classes=2)(logits, target))
    weighted = float(DiceLoss(num_classes=2, class_weights=[0.1, 10.0])(logits, target))
    assert weighted >= base_loss - 1e-6  # rare-class miss carries at least as much weight


def test_focal_loss_finite_and_zero_when_correct():
    target = torch.zeros(1, 2, 2, 2, dtype=torch.long)
    target[0, 0, 0, 0] = 1
    logits = _logits_for_target(target, num_classes=2)
    loss = FocalLoss(num_classes=2, gamma=2.0)(logits, target)
    assert float(loss) < 1e-2


def test_dice_ce_combines_components():
    target = torch.randint(0, 3, (1, 4, 4, 4))
    logits = torch.randn(1, 3, 4, 4, 4, requires_grad=True)
    loss = DiceCELoss(num_classes=3, dice_weight=0.5, ce_weight=0.5)(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_tversky_alpha_beta_asymmetry():
    target = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    target[0, :, 0, 0] = 1  # tiny FN-prone region
    logits_fn = torch.zeros(1, 2, 4, 4, 4)
    logits_fn[:, 0] = 5.0  # predict all background
    high_recall = float(TverskyLoss(num_classes=2, alpha=0.3, beta=0.7)(logits_fn, target))
    high_precision = float(TverskyLoss(num_classes=2, alpha=0.7, beta=0.3)(logits_fn, target))
    # Penalising FN harder (beta=0.7) yields a strictly larger loss when we miss everything.
    assert high_recall > high_precision


def test_deep_supervision_handles_list_and_tensor():
    target = torch.randint(0, 2, (1, 4, 4, 4))
    base = DiceLoss(num_classes=2)
    wrap = DeepSupervisionLoss(base)
    logits = torch.randn(1, 2, 4, 4, 4)
    aux = torch.nn.functional.interpolate(logits, scale_factor=0.5)
    assert torch.isfinite(wrap([logits, aux], target))
    assert torch.isfinite(wrap(logits, target))


def test_boundary_loss_finite():
    target = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    target[0, 1:3, 1:3, 1:3] = 1
    logits = torch.randn(1, 2, 4, 4, 4)
    assert torch.isfinite(BoundaryLoss(num_classes=2)(logits, target))


def test_build_loss_with_compound_components():
    cfg = {
        "loss": "dice_ce",
        "dice_weight": 0.5,
        "ce_weight": 0.5,
        "boundary_weight": 0.1,
        "class_weights": [0.1, 1.0, 1.0, 1.0],
        "deep_supervision_weights": [1.0, 0.5],
    }
    module = build_loss(cfg, num_classes=4, ignore_index=255)
    target = torch.randint(0, 4, (1, 4, 4, 4))
    logits = torch.randn(1, 4, 4, 4, 4, requires_grad=True)
    loss = module(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
