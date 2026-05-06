from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.nn import functional as F


def _compute_starts(size: int, roi: int, overlap: float) -> List[int]:
    if size <= roi:
        return [0]
    step = max(1, int(round(roi * (1.0 - overlap))))
    starts = list(range(0, size - roi + 1, step))
    if starts[-1] != size - roi:
        starts.append(size - roi)
    return starts


def _gaussian_weight(roi_size: Sequence[int], device: torch.device) -> torch.Tensor:
    coords = [torch.linspace(-1, 1, steps=int(s), device=device) for s in roi_size]
    zz, yy, xx = torch.meshgrid(coords[0], coords[1], coords[2], indexing="ij")
    dist = zz**2 + yy**2 + xx**2
    weight = torch.exp(-dist / 0.5).clamp_min(1e-3)
    return weight.unsqueeze(0).unsqueeze(0)


def _first_output(output: Union[torch.Tensor, Sequence[torch.Tensor]]) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    return output[0]


def _predict_batch(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    logits = _first_output(model(batch))
    return torch.softmax(logits, dim=1)


def sliding_window_predict(
    model: torch.nn.Module,
    image: torch.Tensor,
    roi_size: Sequence[int],
    num_classes: int,
    device: torch.device,
    sw_batch_size: int = 2,
    overlap: float = 0.5,
) -> torch.Tensor:
    """Return class probabilities for a full volume.

    Args:
        image: tensor shaped C,D,H,W.
    """
    model.eval()
    if image.ndim != 4:
        raise ValueError(f"Expected C,D,H,W image, got {tuple(image.shape)}")

    c, d, h, w = image.shape
    roi = tuple(int(v) for v in roi_size)
    pad_d = max(0, roi[0] - d)
    pad_h = max(0, roi[1] - h)
    pad_w = max(0, roi[2] - w)
    padded = F.pad(image.unsqueeze(0), (0, pad_w, 0, pad_h, 0, pad_d)).squeeze(0)
    _, pd, ph, pw = padded.shape

    starts_d = _compute_starts(pd, roi[0], overlap)
    starts_h = _compute_starts(ph, roi[1], overlap)
    starts_w = _compute_starts(pw, roi[2], overlap)
    probs = torch.zeros((num_classes, pd, ph, pw), dtype=torch.float32, device=device)
    norm = torch.zeros((1, pd, ph, pw), dtype=torch.float32, device=device)
    weight = _gaussian_weight(roi, device)

    patches: List[torch.Tensor] = []
    locations: List[tuple[int, int, int]] = []
    padded = padded.to(device)

    with torch.no_grad():
        for sd in starts_d:
            for sh in starts_h:
                for sw in starts_w:
                    patch = padded[:, sd : sd + roi[0], sh : sh + roi[1], sw : sw + roi[2]]
                    patches.append(patch)
                    locations.append((sd, sh, sw))
                    if len(patches) == sw_batch_size:
                        batch = torch.stack(patches, dim=0)
                        batch_probs = _predict_batch(model, batch)
                        for p, loc in zip(batch_probs, locations):
                            ld, lh, lw = loc
                            probs[:, ld : ld + roi[0], lh : lh + roi[1], lw : lw + roi[2]] += p * weight.squeeze(0)
                            norm[:, ld : ld + roi[0], lh : lh + roi[1], lw : lw + roi[2]] += weight.squeeze(0)
                        patches.clear()
                        locations.clear()
        if patches:
            batch = torch.stack(patches, dim=0)
            batch_probs = _predict_batch(model, batch)
            for p, loc in zip(batch_probs, locations):
                ld, lh, lw = loc
                probs[:, ld : ld + roi[0], lh : lh + roi[1], lw : lw + roi[2]] += p * weight.squeeze(0)
                norm[:, ld : ld + roi[0], lh : lh + roi[1], lw : lw + roi[2]] += weight.squeeze(0)

    probs = probs / norm.clamp_min(1e-6)
    return probs[:, :d, :h, :w].cpu()


def predict_with_tta(
    model: torch.nn.Module,
    image: torch.Tensor,
    roi_size: Sequence[int],
    num_classes: int,
    device: torch.device,
    sw_batch_size: int = 2,
    overlap: float = 0.5,
    tta_flips: Optional[Iterable[Sequence[int]]] = None,
) -> torch.Tensor:
    probs = sliding_window_predict(model, image, roi_size, num_classes, device, sw_batch_size, overlap)
    flips = list(tta_flips or [])
    for axes in flips:
        axes = tuple(int(a) for a in axes)
        flipped_image = torch.flip(image, dims=axes)
        flipped_probs = sliding_window_predict(model, flipped_image, roi_size, num_classes, device, sw_batch_size, overlap)
        probs = probs + torch.flip(flipped_probs, dims=axes)
    return probs / (1 + len(flips))
