"""Checkpoint loading with explicit, auditable compatibility rules."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import torch


@dataclass(frozen=True)
class LoadReport:
    mode: str
    checkpoint: str
    loaded: tuple[str, ...]
    skipped: tuple[str, ...]

    def to_dict(self):
        return asdict(self)


def read_checkpoint(path):
    checkpoint = torch.load(Path(path), map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, Mapping) else checkpoint
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint must be a state_dict or contain a state_dict mapping")
    # Preserve PyTorch state-dict metadata. spconv uses it when reconstructing
    # sparse-convolution parameters from a checkpoint saved by this model.
    normalized = OrderedDict()
    for key, value in state.items():
        clean = key[7:] if key.startswith("module.") else key
        if clean in normalized:
            raise ValueError(f"duplicate state key after removing module prefix: {clean}")
        normalized[clean] = value
    metadata = getattr(state, "_metadata", None)
    if metadata is not None:
        normalized._metadata = OrderedDict()
        for key, value in metadata.items():
            if key == "module":
                clean = ""
            elif key.startswith("module."):
                clean = key[7:]
            else:
                clean = key
            normalized._metadata[clean] = value
    return checkpoint, normalized


def _check_tensor(key, source, target):
    if not torch.is_tensor(source):
        raise TypeError(f"{key}: checkpoint value is not a tensor")
    if source.shape != target.shape:
        raise ValueError(f"{key}: shape mismatch {tuple(source.shape)} != {tuple(target.shape)}")
    if source.dtype != target.dtype:
        raise ValueError(f"{key}: dtype mismatch {source.dtype} != {target.dtype}")


def load_strict_checkpoint(model, path):
    """Load a native RIGS checkpoint. Every model key, shape and dtype must match."""
    checkpoint, state = read_checkpoint(path)
    expected_contract = ["empty", "background", "foreground"]
    if not isinstance(checkpoint, Mapping) or checkpoint.get("class_contract") != expected_contract:
        raise ValueError(
            "strict resume requires a native RIGS checkpoint with class_contract="
            f"{expected_contract}"
        )
    target = model.state_dict()
    missing = sorted(set(target) - set(state))
    unexpected = sorted(set(state) - set(target))
    if missing or unexpected:
        raise ValueError(
            f"strict checkpoint key mismatch; missing={missing[:20]}, unexpected={unexpected[:20]}")
    for key, target_value in target.items():
        _check_tensor(key, state[key], target_value)
    model.load_state_dict(state, strict=True)
    return checkpoint, LoadReport("strict", str(Path(path)), tuple(sorted(state)), tuple())


def load_backbone_checkpoint(model, path, prefix="img_backbone.", *, prefixes=None,
                             require_complete=False):
    """Load exact tensors from one or more explicitly allowed backbone prefixes.

    ``prefix`` remains supported for callers that load only the main image
    backbone.  Release training passes ``prefixes`` so both independent
    ResNet-101 backbones are initialized from the same audited state dict.
    """
    if prefixes is None:
        prefixes = (prefix,)
    prefixes = tuple(prefixes)
    if not prefixes or any(not isinstance(item, str) or not item for item in prefixes):
        raise ValueError("backbone prefixes must be non-empty strings")
    _, state = read_checkpoint(path)
    target = model.state_dict()
    selected, skipped = {}, []
    for key, value in sorted(state.items()):
        if not key.startswith(prefixes):
            skipped.append(f"{key}: outside {prefixes}")
            continue
        if key not in target:
            skipped.append(f"{key}: absent from model")
            continue
        try:
            _check_tensor(key, value, target[key])
        except (TypeError, ValueError) as error:
            skipped.append(str(error))
            continue
        selected[key] = value
    if not selected:
        raise ValueError(f"checkpoint contains no compatible tensors under {prefixes}")
    model_backbone = {key for key in target if key.startswith(prefixes)}
    missing = sorted(model_backbone - set(selected))
    if require_complete and missing:
        raise ValueError(
            f"backbone checkpoint is incomplete for {prefixes}; missing={missing[:20]}"
        )
    result = model.load_state_dict(selected, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"unexpected backbone keys after filtering: {result.unexpected_keys}")
    skipped.extend(f"{key}: missing from checkpoint" for key in missing)
    return LoadReport("backbone-only", str(Path(path)), tuple(sorted(selected)), tuple(skipped))
