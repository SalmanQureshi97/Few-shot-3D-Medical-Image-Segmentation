from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from contextlib import nullcontext
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import count_parameters, get_device, save_config, seed_everything
from .data import FullVolumeDataset, LoadedSubject, PatchDataset, load_subject
from .infer import predict_with_tta
from .io import Subject
from .labels import foreground_classes, label_names
from .losses import build_loss
from .metrics import assd_per_class, dice_per_class, hd95_per_class, summarise_metric_dict, volume_similarity
from .models import build_model
from .viz import save_overlay_panel


def split_loocv(subjects: List[Subject], fold: int) -> tuple[List[Subject], List[Subject]]:
    if fold < 0 or fold >= len(subjects):
        raise ValueError(f"Fold {fold} out of range for {len(subjects)} subjects")
    val = [subjects[fold]]
    train = [subject for i, subject in enumerate(subjects) if i != fold]
    return train, val


def load_subjects_for_config(subjects: Iterable[Subject], cfg: Dict) -> List[LoadedSubject]:
    data_cfg = cfg["data"]
    return [
        load_subject(
            subject,
            data_cfg["modalities"],
            data_cfg["target"],
            int(data_cfg.get("ignore_index", 255)),
            bool(data_cfg.get("ignore_hindbrain_in_coarse", True)),
            data_cfg.get("intensity_clip_percentiles", [1.0, 99.0]),
        )
        for subject in subjects
    ]


def _write_history_row(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_metric: float, cfg: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_metric": best_metric,
            "config": cfg,
        },
        path,
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    use_amp = amp and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    progress = tqdm(loader, desc=f"epoch {epoch}", leave=False)
    for batch in progress:
        image = batch["image"].to(device, non_blocking=True).float()
        label = batch["label"].to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        if use_amp and hasattr(torch, "amp"):
            autocast_context = torch.amp.autocast(device_type="cuda", enabled=True)
        elif use_amp:
            autocast_context = torch.cuda.amp.autocast(enabled=True)
        else:
            autocast_context = nullcontext()
        with autocast_context:
            outputs = model(image)
            loss = criterion(outputs, label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item())
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(1, len(loader))


def validate(
    model: torch.nn.Module,
    subjects: List[LoadedSubject],
    cfg: Dict,
    device: torch.device,
    out_dir: Optional[Path] = None,
    epoch: Optional[int] = None,
) -> List[Dict]:
    data_cfg = cfg["data"]
    infer_cfg = cfg["inference"]
    ignore_index = int(data_cfg.get("ignore_index", 255))
    classes = foreground_classes(data_cfg["target"])
    rows: List[Dict] = []

    for subject in subjects:
        if subject.label is None:
            continue
        start = time.time()
        probs = predict_with_tta(
            model,
            torch.from_numpy(subject.image),
            infer_cfg["roi_size"],
            int(data_cfg["num_classes"]),
            device,
            int(infer_cfg.get("sw_batch_size", 2)),
            float(infer_cfg.get("overlap", 0.5)),
            infer_cfg.get("tta_flips", []),
        )
        pred = probs.argmax(dim=0).numpy().astype(np.int64)
        elapsed = time.time() - start
        target = subject.label
        dice = dice_per_class(pred, target, classes, ignore_index)
        volume = volume_similarity(pred, target, classes, ignore_index)
        hd95 = hd95_per_class(pred, target, classes, subject.spacing_dhw, ignore_index)
        assd = assd_per_class(pred, target, classes, subject.spacing_dhw, ignore_index)
        row = {
            "subject_id": subject.subject_id,
            "epoch": epoch if epoch is not None else -1,
            "inference_seconds": elapsed,
        }
        row.update(summarise_metric_dict("dice", dice))
        row.update(summarise_metric_dict("volsim", volume))
        row.update(summarise_metric_dict("hd95", hd95))
        row.update(summarise_metric_dict("assd", assd))
        rows.append(row)
        if out_dir is not None:
            save_overlay_panel(
                subject.image,
                target,
                pred,
                out_dir / "figures" / f"{subject.subject_id}_epoch_{epoch}_overlay.png",
                subject.subject_id,
            )
    return rows


def run_training(
    cfg: Dict,
    train_subjects: List[Subject],
    val_subjects: List[Subject],
    fold: int,
    device_name: str = "auto",
) -> Path:
    seed_everything(int(cfg["experiment"].get("seed", 0)) + fold)
    device = get_device(device_name)
    run_root = Path(cfg["experiment"]["out_dir"]) / f"fold_{fold}"
    run_root.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_root / "config.yaml")

    train_loaded = load_subjects_for_config(train_subjects, cfg)
    val_loaded = load_subjects_for_config(val_subjects, cfg)

    train_ds = PatchDataset(
        train_loaded,
        cfg["training"]["patch_size"],
        int(cfg["training"]["samples_per_epoch"]),
        float(cfg["training"]["foreground_patch_ratio"]),
        int(cfg["data"].get("ignore_index", 255)),
        cfg.get("augmentation", {"enabled": False}),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    model = build_model(cfg["model"]).to(device)
    criterion = build_loss(cfg["training"], int(cfg["data"]["num_classes"]), int(cfg["data"].get("ignore_index", 255)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["training"]["epochs"]))
    writer = SummaryWriter(str(run_root / "tensorboard"))
    writer.add_text("model/parameters", str(count_parameters(model)))

    best_metric = -np.inf
    best_path = run_root / "checkpoints" / "best.pt"
    history_path = run_root / "history.csv"
    metrics_path = run_root / "metrics_val.csv"
    metric_exists = metrics_path.exists()

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            bool(cfg["training"].get("amp", True)),
        )
        scheduler.step()
        writer.add_scalar("loss/train", train_loss, epoch)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "parameters": count_parameters(model),
        }

        if epoch % int(cfg["training"].get("val_interval", 1)) == 0 or epoch == int(cfg["training"]["epochs"]):
            val_rows = validate(model, val_loaded, cfg, device, run_root, epoch)
            val_mean = float(np.nanmean([r["dice_mean"] for r in val_rows])) if val_rows else np.nan
            row["val_dice_mean"] = val_mean
            writer.add_scalar("dice/val_mean", val_mean, epoch)
            if val_rows:
                with metrics_path.open("a", newline="", encoding="utf-8") as f:
                    writer_csv = csv.DictWriter(f, fieldnames=list(val_rows[0].keys()))
                    if not metric_exists:
                        writer_csv.writeheader()
                        metric_exists = True
                    writer_csv.writerows(val_rows)
            if np.isfinite(val_mean) and val_mean > best_metric:
                best_metric = val_mean
                _save_checkpoint(best_path, model, optimizer, epoch, best_metric, cfg)

        _write_history_row(history_path, row)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        save_every = int(cfg["training"].get("save_every", 0))
        if save_every and epoch % save_every == 0:
            _save_checkpoint(run_root / "checkpoints" / f"epoch_{epoch}.pt", model, optimizer, epoch, best_metric, cfg)

    writer.close()
    return run_root
