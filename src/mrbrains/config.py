from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import torch
import yaml


def load_config(path: Union[str, Path], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load a YAML config and apply nested dictionary overrides."""
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def parse_override(token: str) -> Dict[str, Any]:
    """Parse 'training.epochs=5' or 'data.modalities=[t1,ir]' into a nested dict.

    Values are interpreted as YAML scalars so ints, floats, bools and lists
    are all valid right-hand sides.
    """
    if "=" not in token:
        raise ValueError(f"Override missing '=': {token!r}")
    key, raw = token.split("=", 1)
    keys = key.strip().split(".")
    value: Any = yaml.safe_load(raw)
    for k in reversed(keys):
        value = {k: value}
    return value  # type: ignore[return-value]


def apply_overrides(cfg: Dict[str, Any], tokens: Optional[Iterable[str]]) -> Dict[str, Any]:
    if not tokens:
        return cfg
    out = cfg
    for tok in tokens:
        out = deep_update(out, parse_override(tok))
    return out


def save_config(cfg: Dict[str, Any], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
