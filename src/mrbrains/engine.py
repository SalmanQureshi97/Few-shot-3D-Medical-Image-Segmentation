from __future__ import annotations

import copy
import csv
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import count_parameters, get_device, save_config, seed_everything
from .data import FullVolumeDataset, LoadedSubject, PatchDataset, cached_loaded_subjects
from .infer import predict_with_tta
from .io import Subject
from .labels import (
    class_voxel_frequencies,
    foreground_classes,
    inverse_frequency_weights,
    label_names,
)
from .losses import build_loss
from .metrics import (
    assd_per_class,
    avd_per_class,
    confusion_matrix,
    dice_per_class,
    hd95_per_class,
    summarise_metric_dict,
    volume_similarity,
)
from .models import build_model
from .viz import save_overlay_panel


def split_loocv(subjects: List[Subject], fold: int) -> tuple[List[Subject], List[Subject]]:
    if fold < 0 or fold >= len(subjects):
        raise ValueError(f"Fold {fold} out of range for {len(subjects)} subjects")
    val = [subjects[fold]]
    train = [subject for i, subject in enumerate(subjects) if i != fold]
    return train, val


def load_subjects_for_config(
    subjects: Iterable[Subject], cfg: Dict, cache_dir: Optional[Path] = None
) -> List[LoadedSubject]:
    data_cfg = cfg["data"]
    return cached_loaded_subjects(
        subjects,
        modalities=data_cfg["modalities"],
        target=data_cfg["target"],
        ignore_index=int(data_cfg.get("ignore_index", 255)),
        ignore_hindbrain_in_coarse=bool(data_cfg.get("ignore_hindbrain_in_coarse", True)),
        clip_percentiles=data_cfg.get("intensity_clip_percentiles", [1.0, 99.0]),
        cache_dir=cache_dir if data_cfg.get("cache_subjects", False) else None,
        ignore_hindbrain_in_detailed=bool(data_cfg.get("ignore_hindbrain_in_detailed", False)),
    )


_HISTORY_FIELDS = [
    "epoch",
    "train_loss",
    "lr",
    "parameters",
    "epoch_seconds",
    "peak_memory_mb",
    "val_dice_mean",
    "val_avd_mean",
    "val_hd95_mean",
]


def _write_history_row(path: Path, row: Dict) -> None:
    """Append a row to history.csv with a fixed schema so pandas can read it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    full = {field: row.get(field, "") for field in _HISTORY_FIELDS}
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_HISTORY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(full)


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    best_metric: float,
    cfg: Dict,
    ema_state: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_metric": best_metric,
        "config": cfg,
    }
    if ema_state is not None:
        payload["ema_state"] = ema_state
    torch.save(payload, path)


class _EMA:
    """Exponential moving average of model weights for inference at validation."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {
            n: p.detach().clone() for n, p in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.state_dict().items():
            if not torch.is_floating_point(param):
                self.shadow[name] = param.detach().clone()
                continue
            # If the live model briefly contains NaN/Inf (e.g. AMP overflow on
            # a residual-attention forward), refuse to mix that into the EMA
            # shadow — otherwise the validation copy stays NaN forever.
            if not torch.isfinite(param).all():
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow

    def apply_to(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        backup = {n: p.detach().clone() for n, p in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=False)
        return backup

    @staticmethod
    def restore(model: torch.nn.Module, backup: Dict[str, torch.Tensor]) -> None:
        model.load_state_dict(backup, strict=False)


def _build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: Dict
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    name = str(cfg.get("scheduler", "cosine")).lower()
    epochs = int(cfg["epochs"])
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "cosine_warmup":
        warmup = int(cfg.get("warmup_epochs", max(1, epochs // 20)))

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return float(step + 1) / max(1, warmup)
            progress = (step - warmup) / max(1, epochs - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if name == "poly":
        power = float(cfg.get("poly_power", 0.9))
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: (1.0 - step / max(1, epochs)) ** power
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(cfg.get("plateau_factor", 0.5)),
            patience=int(cfg.get("plateau_patience", 10)),
            threshold=1e-4,
        )
    raise ValueError(f"Unknown scheduler: {name}")


def _amp_context(use_amp: bool, device: torch.device):
    if not use_amp:
        return nullcontext()
    if hasattr(torch, "amp"):
        return torch.amp.autocast(device_type=device.type, enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def _amp_scaler(use_amp: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def _peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device)) / (1024**2)


def _has_nonfinite_grads(model: torch.nn.Module) -> bool:
    for p in model.parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return True
    return False


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    amp: bool,
    grad_clip: float,
    scaler,
    ema: Optional[_EMA],
) -> float:
    model.train()
    total_loss = 0.0
    n_steps = 0
    n_skipped_loss = 0
    n_skipped_grad = 0
    use_amp = amp and device.type == "cuda"
    progress = tqdm(loader, desc=f"epoch {epoch}", leave=False)
    for batch in progress:
        image = batch["image"].to(device, non_blocking=True).float()
        label = batch["label"].to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        with _amp_context(use_amp, device):
            outputs = model(image)
            loss = criterion(outputs, label)
        # Skip iteration entirely if the forward produced NaN/Inf — this
        # prevents one bad patch (e.g. all-ignore, or AMP overflow on a
        # near-constant volume) from poisoning the model.
        if not torch.isfinite(loss):
            n_skipped_loss += 1
            progress.set_postfix(loss="nan-skip")
            continue
        scaler.scale(loss).backward()
        # Unscale before checking gradients so AMP scaling is removed first.
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        if _has_nonfinite_grads(model):
            # GradScaler also detects this and refuses the step, but on CPU
            # or when AMP is off we still need to bail out manually.
            n_skipped_grad += 1
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            progress.set_postfix(loss=f"{loss.item():.4f} grad-skip")
            continue
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)
        total_loss += float(loss.item())
        n_steps += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")
    if n_skipped_loss or n_skipped_grad:
        progress.write(
            f"epoch {epoch}: skipped {n_skipped_loss} non-finite-loss and "
            f"{n_skipped_grad} non-finite-grad iterations"
        )
    return total_loss / max(1, n_steps)


def validate(
    model: torch.nn.Module,
    subjects: List[LoadedSubject],
    cfg: Dict,
    device: torch.device,
    out_dir: Optional[Path] = None,
    epoch: Optional[int] = None,
    save_overlays: bool = True,
) -> List[Dict]:
    data_cfg = cfg["data"]
    infer_cfg = cfg["inference"]
    ignore_index = int(data_cfg.get("ignore_index", 255))
    classes = foreground_classes(data_cfg["target"])
    num_classes = int(data_cfg["num_classes"])
    rows: List[Dict] = []

    for subject in subjects:
        if subject.label is None:
            continue
        start = time.time()
        probs = predict_with_tta(
            model,
            torch.from_numpy(subject.image),
            infer_cfg["roi_size"],
            num_classes,
            device,
            int(infer_cfg.get("sw_batch_size", 2)),
            float(infer_cfg.get("overlap", 0.5)),
            infer_cfg.get("tta_flips", []),
            float(infer_cfg.get("gaussian_sigma_scale", 0.125)),
            bool(infer_cfg.get("amp", True)),
        )
        pred = probs.argmax(dim=0).numpy().astype(np.int64)
        elapsed = time.time() - start
        target = subject.label
        dice = dice_per_class(pred, target, classes, ignore_index)
        volume = volume_similarity(pred, target, classes, ignore_index)
        avd = avd_per_class(pred, target, classes, ignore_index)
        hd95 = hd95_per_class(pred, target, classes, subject.spacing_dhw, ignore_index)
        assd = assd_per_class(pred, target, classes, subject.spacing_dhw, ignore_index)
        cm = confusion_matrix(pred, target, num_classes, ignore_index)
        row: Dict[str, object] = {
            "subject_id": subject.subject_id,
            "epoch": epoch if epoch is not None else -1,
            "inference_seconds": elapsed,
        }
        row.update(summarise_metric_dict("dice", dice))
        row.update(summarise_metric_dict("volsim", volume))
        row.update(summarise_metric_dict("avd", avd))
        row.update(summarise_metric_dict("hd95", hd95))
        row.update(summarise_metric_dict("assd", assd))
        for i in range(num_classes):
            for j in range(num_classes):
                row[f"cm_{i}_{j}"] = int(cm[i, j])
        rows.append(row)
        if out_dir is not None and save_overlays:
            save_overlay_panel(
                subject.image,
                target,
                pred,
                out_dir / "figures" / f"{subject.subject_id}_epoch_{epoch}_overlay.png",
                subject.subject_id,
            )
    return rows


def _maybe_resume(
    resume_path: Optional[Path],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    ema: Optional[_EMA],
    device: torch.device,
) -> tuple[int, float]:
    if resume_path is None or not resume_path.exists():
        return 0, -float("inf")
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if ema is not None and "ema_state" in ckpt:
        ema.shadow = {k: v.to(device) for k, v in ckpt["ema_state"].items()}
    return int(ckpt.get("epoch", 0)), float(ckpt.get("best_metric", -float("inf")))


def _resolve_class_weights(cfg: Dict, train_subjects: List[LoadedSubject]) -> Optional[List[float]]:
    train_cfg = cfg.get("training", {})
    weights = train_cfg.get("class_weights")
    if weights == "auto":
        labels = [s.label for s in train_subjects if s.label is not None]
        freq = class_voxel_frequencies(labels, int(cfg["data"]["num_classes"]), int(cfg["data"].get("ignore_index", 255)))
        weights = inverse_frequency_weights(freq).tolist()
    return weights


def run_training(
    cfg: Dict,
    train_subjects: List[Subject],
    val_subjects: List[Subject],
    fold: int,
    device_name: str = "auto",
    resume: Optional[Path] = None,
) -> Path:
    seed_everything(int(cfg["experiment"].get("seed", 0)) + fold)
    device = get_device(device_name)
    run_root = Path(cfg["experiment"]["out_dir"]) / f"fold_{fold}"
    run_root.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(cfg["experiment"].get("cache_dir", run_root.parent / "_cache"))
    train_loaded = load_subjects_for_config(train_subjects, cfg, cache_dir=cache_dir)
    val_loaded = load_subjects_for_config(val_subjects, cfg, cache_dir=cache_dir)

    class_weights = _resolve_class_weights(cfg, train_loaded)
    if class_weights is not None:
        cfg["training"] = {**cfg["training"], "class_weights": class_weights}
    save_config(cfg, run_root / "config.yaml")

    train_ds = PatchDataset(
        train_loaded,
        cfg["training"]["patch_size"],
        int(cfg["training"]["samples_per_epoch"]),
        float(cfg["training"]["foreground_patch_ratio"]),
        int(cfg["data"].get("ignore_index", 255)),
        cfg.get("augmentation", {"enabled": False}),
        class_balanced_classes=cfg["training"].get("class_balanced_classes"),
        class_balanced_ratio=float(cfg["training"].get("class_balanced_ratio", 0.0)),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    model = build_model(cfg["model"]).to(device)
    criterion = build_loss(
        cfg["training"], int(cfg["data"]["num_classes"]), int(cfg["data"].get("ignore_index", 255))
    ).to(device)  # ensure CE weight buffers follow the model device
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )
    scheduler = _build_scheduler(optimizer, cfg["training"])
    ema_decay = float(cfg["training"].get("ema_decay", 0.0))
    ema = _EMA(model, decay=ema_decay) if ema_decay > 0 else None
    use_amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    scaler = _amp_scaler(use_amp)
    grad_clip = float(cfg["training"].get("grad_clip", 0.0))
    early_stop_patience = int(cfg["training"].get("early_stop_patience", 0))
    overlay_every = int(cfg["training"].get("overlay_every", 0))

    writer = SummaryWriter(str(run_root / "tensorboard"))
    writer.add_text("model/parameters", str(count_parameters(model)))

    start_epoch, best_metric = _maybe_resume(resume, model, optimizer, scheduler, ema, device)
    best_path = run_root / "checkpoints" / "best.pt"
    history_path = run_root / "history.csv"
    metrics_path = run_root / "metrics_val.csv"
    metric_exists = metrics_path.exists()
    no_improve = 0

    total_epochs = int(cfg["training"]["epochs"])
    for epoch in range(start_epoch + 1, total_epochs + 1):
        epoch_start = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            use_amp, grad_clip, scaler, ema,
        )
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
        epoch_time = time.time() - epoch_start
        peak_mem_mb = _peak_memory_mb(device)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("time/epoch_seconds", epoch_time, epoch)
        writer.add_scalar("memory/peak_mb", peak_mem_mb, epoch)
        row: Dict[str, object] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "parameters": count_parameters(model),
            "epoch_seconds": epoch_time,
            "peak_memory_mb": peak_mem_mb,
        }

        is_val_epoch = (
            epoch % int(cfg["training"].get("val_interval", 1)) == 0 or epoch == total_epochs
        )
        if is_val_epoch:
            ema_backup = None
            if ema is not None:
                ema_backup = ema.apply_to(model)
            save_overlay = overlay_every <= 0 or (epoch % overlay_every == 0) or epoch == total_epochs
            val_rows = validate(model, val_loaded, cfg, device, run_root, epoch, save_overlays=save_overlay)
            if ema_backup is not None:
                _EMA.restore(model, ema_backup)
            val_mean = float(np.nanmean([r["dice_mean"] for r in val_rows])) if val_rows else float("nan")
            row["val_dice_mean"] = val_mean
            row["val_avd_mean"] = float(np.nanmean([r["avd_mean"] for r in val_rows])) if val_rows else float("nan")
            row["val_hd95_mean"] = float(np.nanmean([r["hd95_mean"] for r in val_rows])) if val_rows else float("nan")
            writer.add_scalar("dice/val_mean", val_mean, epoch)
            writer.add_scalar("avd/val_mean", row["val_avd_mean"], epoch)
            writer.add_scalar("hd95/val_mean", row["val_hd95_mean"], epoch)
            if val_rows:
                with metrics_path.open("a", newline="", encoding="utf-8") as f:
                    writer_csv = csv.DictWriter(f, fieldnames=list(val_rows[0].keys()))
                    if not metric_exists:
                        writer_csv.writeheader()
                        metric_exists = True
                    writer_csv.writerows(val_rows)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and np.isfinite(val_mean):
                scheduler.step(val_mean)
            if np.isfinite(val_mean) and val_mean > best_metric:
                best_metric = val_mean
                no_improve = 0
                _save_checkpoint(
                    best_path, model, optimizer, scheduler, epoch, best_metric, cfg,
                    ema_state=ema.state_dict() if ema is not None else None,
                )
            else:
                no_improve += 1

        _write_history_row(history_path, row)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        save_every = int(cfg["training"].get("save_every", 0))
        if save_every and epoch % save_every == 0:
            _save_checkpoint(
                run_root / "checkpoints" / f"epoch_{epoch}.pt",
                model, optimizer, scheduler, epoch, best_metric, cfg,
                ema_state=ema.state_dict() if ema is not None else None,
            )

        if early_stop_patience > 0 and no_improve >= early_stop_patience:
            writer.add_text("early_stop", f"epoch {epoch}: no improvement for {no_improve} validations", epoch)
            break

    writer.close()
    return run_root
