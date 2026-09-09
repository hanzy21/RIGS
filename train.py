#!/usr/bin/env python3
"""Single-GPU training entry point for RIGS."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import torch
from mmengine import Config
from mmengine.optim import build_optim_wrapper
from mmseg.models import build_segmentor
from timm.scheduler import CosineLRScheduler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model/head/localagg_prob"))

import model  # noqa: E402,F401
from dataset import get_dataloader  # noqa: E402
from loss import OPENOCC_LOSS  # noqa: E402
from misc.checkpoint_util import load_backbone_checkpoint  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move(item, device) for item in value]
    return value


def loss_inputs(cfg, output, metadata, global_step):
    values = dict(output)
    values.update({key: value for key, value in metadata.items() if key not in values})
    values["metas"] = metadata
    values["global_iter"] = global_step
    converted = {"metas": metadata, "global_iter": global_step}
    for destination, source in cfg.loss_input_convertion.items():
        if source not in values:
            raise KeyError(f"loss input {destination!r} requires missing value {source!r}")
        converted[destination] = values[source]
    return converted


def update_temporal_inputs(batch, state):
    current = batch["sample_idx"][0]
    expected_previous = batch["previous_sample_idx"][0]
    continuity = bool(batch["is_continuous_frame"][0])
    valid = continuity and state["token"] is not None and expected_previous == state["token"]
    batch["prev_gaussian_representation"] = state["anchor"] if valid else None
    batch["prev_gaussian_rep_features"] = state["features"] if valid else None
    if not valid:
        state.update(token=None, anchor=None, features=None)
    return current


def update_temporal_state(state, current, output):
    anchor = output.get("refined_anchor")
    features = output.get("refined_rep_features", output.get("rep_features"))
    if anchor is None or features is None:
        raise KeyError("temporal lifter must return refined_anchor and representation features")
    state.update(token=current, anchor=anchor.detach(), features=features.detach())


def save_checkpoint(path, model, optimizer, scheduler, epoch, global_step):
    """Atomically save a checkpoint produced by this training run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_iter": global_step,
        "last_iter": 0,
        "class_contract": ["empty", "background", "foreground"],
    }, temporary)
    os.replace(temporary, path)


def initialize(cfg, work_dir: Path, device, train_loader):
    network = build_segmentor(cfg.model)
    network.init_weights()
    network.to(device)
    optimizer = build_optim_wrapper(network, cfg.optimizer)
    total_steps = len(train_loader) * int(cfg.max_epochs)
    raw_optimizer = optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer
    scheduler = CosineLRScheduler(
        raw_optimizer, t_initial=total_steps,
        lr_min=cfg.optimizer["optimizer"]["lr"] * cfg.get("min_lr_ratio", 0.1),
        warmup_t=cfg.get("warmup_iters", 500), warmup_lr_init=1e-6,
        t_in_epochs=False,
    )
    if cfg.get("load_from"):
        report = load_backbone_checkpoint(
            network, cfg.load_from,
            prefixes=cfg.get("backbone_checkpoint_prefixes", ("img_backbone.",)),
            require_complete=bool(cfg.get("require_complete_backbone_checkpoint", False)),
        )
        (work_dir / "backbone_initialization.json").write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    elif cfg.get("require_backbone_checkpoint", False):
        raise ValueError("RIGS_BACKBONE_CHECKPOINT must point to a compatible backbone checkpoint")
    return network, optimizer, scheduler


def train_epoch(network, loader, objective, optimizer, scheduler, cfg, device,
                epoch, global_step, max_steps=None):
    network.train()
    state = dict(token=None, anchor=None, features=None)
    optimizer.zero_grad()
    for iteration, raw_batch in enumerate(loader):
        if max_steps is not None and iteration >= max_steps:
            break
        batch = move(raw_batch, device)
        current = update_temporal_inputs(batch, state)
        images = batch.pop("img")
        output = network(imgs=images, metas=batch, global_iter=global_step)
        update_temporal_state(state, current, output)
        loss, components = objective(loss_inputs(cfg, output, batch, global_step))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {global_step}: {float(loss)}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), cfg.grad_max_norm)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step_update(global_step)
        if iteration % int(cfg.print_freq) == 0:
            print(json.dumps({"epoch": epoch, "step": global_step,
                              "loss": float(loss.detach()), **components}, sort_keys=True))
        global_step += 1
    return global_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config/rigs_kradar.py")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--max-epochs", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("RIGS training requires an NVIDIA CUDA device")
    if args.max_epochs is not None and args.max_epochs <= 0:
        raise ValueError("--max-epochs must be positive")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, args.work_dir / "resolved_config.py")
    seed_everything(args.seed)
    cfg = Config.fromfile(args.config)
    if args.max_epochs is not None:
        cfg.max_epochs = args.max_epochs
    train_loader, _ = get_dataloader(
        cfg.train_dataset_config, cfg.val_dataset_config,
        cfg.train_loader, cfg.val_loader, dist=False,
        use_temporal_init=True,
    )
    device = torch.device("cuda")
    network, optimizer, scheduler = initialize(cfg, args.work_dir, device, train_loader)
    objective = OPENOCC_LOSS.build(cfg.loss).to(device)
    epoch = global_step = 0
    while epoch < int(cfg.max_epochs):
        global_step = train_epoch(
            network, train_loader, objective, optimizer, scheduler, cfg, device,
            epoch, global_step, args.max_train_steps)
        epoch += 1
        save_checkpoint(args.work_dir / "latest.pth", network, optimizer, scheduler,
                        epoch, global_step)


if __name__ == "__main__":
    main()
