#!/usr/bin/env python3
"""Entry point for training and evaluation from YAML config.

Usage::

    python main.py train --config config.yaml
    python main.py eval --config config.yaml
    python main.py all --config config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.base.codes import repetition_code, surface_code
from src.core.baseline_decoder import DecoderConfig, GraphDecoder
from src.core.mwpm_decoder import MWPMDecoder
from src.core.simulator import Simulator
from src.task.evaluation import scan_logical_error_rate
from src.task.losses import DecoderLoss
from src.task.training import Trainer, TrainerConfig

# Keys in the ``training:`` block of config.yaml
_DEM_KEYS = frozenset({"code_type", "code_distance", "error_rate", "batch_size", "device"})
_MODEL_KEYS = frozenset({
    "d_model", "n_unique", "d_ff", "depth_multiplier",
    "processor_type", "n_heads", "graph_modes", "update_edge_attr",
})
_LOSS_KEYS = frozenset({
    "objective", "rl_algorithm", "pos_weight",
    "grpo_group_size", "grpo_clip_range",
    "temperature", "entropy_coeff", "kl_penalty_coeff",
})


def _load_config(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping, got {type(data)}")
    return data


def _build_dem(tr: Dict[str, Any]):
    """Build a DEM from the ``training`` subsection."""
    code_type = tr["code_type"]
    distance = int(tr["code_distance"])
    error_rate = tr["error_rate"]
    batch_size = int(tr.get("batch_size", 1))
    device = tr.get("device", "cpu")
    if code_type in ("repetition", "repetition_code"):
        return repetition_code(distance, error_rate, batch_size=batch_size, device=device)
    if code_type in ("surface", "surface_code"):
        return surface_code(distance, error_rate, batch_size=batch_size, device=device)
    raise ValueError(f"Unsupported code_type: {code_type!r}")


def _build_teacher(name: Optional[str], dem):
    if name is None:
        return None
    key = str(name).lower().strip()
    if key in ("none", "", "null", "student", "self"):
        return None
    if key == "mwpm":
        return MWPMDecoder(dem)
    raise ValueError(
        f"Unknown teacher: {name!r}. Supported: 'mwpm' or omit for self-play."
    )


def run_training(cfg: Dict[str, Any]) -> None:
    tr = dict(cfg["training"])
    dem = _build_dem(tr)
    device = dem.device

    teacher_name = tr.pop("teacher", None)
    loss_kw = {k: tr.pop(k) for k in list(tr.keys()) if k in _LOSS_KEYS}
    model_kw = {k: tr.pop(k) for k in list(tr.keys()) if k in _MODEL_KEYS}
    for k in _DEM_KEYS:
        tr.pop(k, None)

    model_cfg = DecoderConfig(**model_kw)
    decoder = GraphDecoder(dem, config=model_cfg).to(device)
    simulator = Simulator(dem)
    teacher = _build_teacher(teacher_name, dem)

    loss_module = DecoderLoss(
        decoder,
        simulator,
        teacher=teacher,
        **loss_kw,
    )
    trainer_cfg = TrainerConfig(**tr)
    trainer = Trainer(loss_module, trainer_cfg)
    trainer.train()


def run_eval(cfg: Dict[str, Any]) -> None:
    if "evaluation" not in cfg:
        raise KeyError(
            "Config missing 'evaluation' section. "
            "See README and config.yaml for required keys (e.g. code_distances)."
        )
    scan_logical_error_rate(cfg)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train or evaluate graph decoder (GNN or transformer).",
    )
    parser.add_argument(
        "command",
        choices=("train", "eval", "all"),
        help="train: run Trainer; eval: run LER sweep; all: train then eval",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to YAML config (default: ./config.yaml)",
    )
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)

    if args.command == "train":
        run_training(cfg)
    elif args.command == "eval":
        run_eval(cfg)
    else:
        run_training(cfg)
        run_eval(cfg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
