"""Evaluation utilities for QEC decoders.

Two entry points:

* :func:`evaluate_logical_error_rate` — estimate logical error rate for
  a single (DEM, decoder) pair over multiple batches.
* :func:`scan_logical_error_rate` — sweep across code distances and error
  rates, comparing multiple decoder types. Results saved as CSV.

Supported decoder types:

* ``"none"`` — no decoder (raw error rate baseline).
* ``"mwpm"`` — Minimum Weight Perfect Matching (PyMatching).
* ``"baseline"`` — :class:`~src.core.baseline_decoder.GraphDecoder` (checkpoint optional).

**Protocol** — Each round passes ``sim.syndromes`` into ``decoder.update_corrections``;
the baseline embeds syndromes on check nodes then runs the graph stack (see
:meth:`~src.core.baseline_decoder.GraphDecoder.correction_logits`).
"""

from __future__ import annotations

import csv
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch

from ..base.codes import repetition_code, surface_code
from ..base.dem import DetectorErrorModel
from ..core.decoder import Decoder
from ..core.simulator import Simulator


# =========================================================================
# Device resolution
# =========================================================================

def _resolve_device(device: Optional[str | torch.device] = None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =========================================================================
# Error rate builders
# =========================================================================

def _build_error_rates(eval_cfg: Dict[str, Any]) -> Sequence[float]:
    if "error_rates" in eval_cfg:
        return [float(r) for r in eval_cfg["error_rates"]]
    logspace = eval_cfg.get("error_rates_logspace", {})
    start_exp = float(logspace.get("start_exp", -3.0))
    end_exp = float(logspace.get("end_exp", -1.0))
    steps = int(logspace.get("steps", 20))
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    return (10 ** torch.linspace(start_exp, end_exp, steps)).tolist()


def _load_ler_csv(path: str) -> Dict[float, Tuple[float, float]]:
    """Map physical error rate -> (mean LER, std LER) from a sweep CSV."""
    out: Dict[float, Tuple[float, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            er = float(row["error_rate"])
            mean = float(row["logical_error_rate_mean"])
            std = float(row["logical_error_rate_std"])
            out[er] = (mean, std)
    return out


def _try_load_ler_csv(path: str) -> Dict[float, Tuple[float, float]]:
    try:
        return _load_ler_csv(path)
    except (OSError, KeyError, ValueError, TypeError):
        return {}


def _reuse_cached_ler_csv(
    output_path: str,
    safe_name: str,
    eval_cfg: Dict[str, Any],
    baseline_checkpoint: Optional[str],
) -> bool:
    """Whether to load *output_path* instead of re-running the sweep cell.

    Baseline decoder CSVs are treated as stale when the configured
    checkpoint file is newer than the CSV (e.g. after ``train`` wrote a new
    ``checkpoint_best.pt``). Set ``skip_existing: false`` in the evaluation
    config to always recompute every decoder.
    """
    reuse = True
    skip_existing = bool(eval_cfg.get("skip_existing", True))
    if not skip_existing:
        reuse = False
    if reuse and not os.path.isfile(output_path):
        reuse = False

    ckpt = baseline_checkpoint
    out_mtime: Optional[float] = None
    ckpt_mtime: Optional[float] = None
    if reuse and safe_name == "baseline":
        if ckpt and os.path.isfile(ckpt):
            try:
                ckpt_mtime = os.path.getmtime(ckpt)
                out_mtime = os.path.getmtime(output_path)
                # Old logic: treat as stale when checkpoint is newer.
                if ckpt_mtime > out_mtime:
                    reuse = False
            except OSError:
                # If mtime is unavailable, keep reuse.
                pass

    return reuse


def _ler_at_rate(
    store: Dict[float, Tuple[float, float]], p: float
) -> Optional[Tuple[float, float]]:
    for k, v in store.items():
        if math.isclose(float(k), float(p), rel_tol=0.0, abs_tol=1e-15):
            return v
    return None


def _print_ler_comparison_table(
    code_type: str,
    dist: int,
    error_rates: Sequence[float],
    decoder_names: Sequence[str],
    ler_by_decoder: Dict[str, Dict[float, Tuple[float, float]]],
) -> None:
    """Print mean LER: rows = physical error rate *p*, columns = decoders."""
    labels = [str(d).strip() for d in decoder_names]
    safe_keys = [str(d).lower().strip() for d in decoder_names]

    header = ["p"] + labels
    rows_cells: List[List[str]] = []
    for p in error_rates:
        row = [f"{p:g}"]
        for sk in safe_keys:
            found = _ler_at_rate(ler_by_decoder.get(sk, {}), p)
            row.append(f"{found[0]:.6f}" if found is not None else "—")
        rows_cells.append(row)

    widths = [
        max(len(header[i]), max((len(rows_cells[j][i]) for j in range(len(rows_cells))), default=0))
        for i in range(len(header))
    ]

    def fmt_row(cells: List[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    print(f"\n=== Logical error rate (mean) | {code_type} d={dist} ===")
    print(fmt_row(header))
    print(fmt_row(["-" * widths[i] for i in range(len(widths))]))
    for row in rows_cells:
        print(fmt_row(row))


# =========================================================================
# DEM builder
# =========================================================================

def _build_dem(
    code_type: str,
    code_distance: int,
    error_rate: float,
    batch_size: int,
    device: torch.device,
) -> DetectorErrorModel:
    if code_type in ("repetition", "repetition_code"):
        return repetition_code(code_distance, error_rate,
                               batch_size=batch_size, device=device)
    if code_type in ("surface", "surface_code"):
        return surface_code(code_distance, error_rate,
                            batch_size=batch_size, device=device)
    raise ValueError(f"Unsupported code_type: {code_type!r}")


# =========================================================================
# Decoder factory
# =========================================================================

def build_decoder(
    name: str,
    dem: DetectorErrorModel,
    checkpoint_path: Optional[str] = None,
    baseline_config: Optional[Dict[str, Any]] = None,
) -> Optional[Decoder]:
    """Build a decoder by name.

    Args:
        name: ``"none"``, ``"mwpm"``, or ``"baseline"``.
        dem: DetectorErrorModel to decode.
        checkpoint_path: Weights for the graph baseline (optional if config alone suffices).
        baseline_config: Kwargs for :class:`~src.core.baseline_decoder.DecoderConfig`.

    Returns:
        Decoder instance, or ``None`` for ``"none"``.
    """
    key = name.lower().strip()

    if key in ("none", "no", "null"):
        return None

    if key == "mwpm":
        from ..core.mwpm_decoder import MWPMDecoder
        return MWPMDecoder(dem)

    if key == "baseline":
        from ..core.baseline_decoder import DecoderConfig, GraphDecoder

        ckpt = None
        if checkpoint_path:
            ckpt = torch.load(
                checkpoint_path, map_location=dem.device, weights_only=False
            )
        cfg_kw = baseline_config
        resolved_config = dict(cfg_kw or {})
        if ckpt is not None and "model_config" in ckpt and not cfg_kw:
            resolved_config = ckpt["model_config"]
        cfg = DecoderConfig(**resolved_config)
        decoder = GraphDecoder(dem, config=cfg)
        if ckpt is not None:
            decoder.load_state_dict(ckpt["model_state_dict"])
        decoder.to(dem.device)
        decoder.eval()
        return decoder

    raise ValueError(
        f"Unknown decoder: {name!r}. Supported: 'none', 'mwpm', 'baseline'."
    )


# =========================================================================
# Core evaluation
# =========================================================================

@torch.no_grad()
def evaluate_logical_error_rate(
    dem: DetectorErrorModel,
    decoder: Optional[Decoder] = None,
    num_batches: int = 1000,
    num_rounds: int = 1,
    progress: bool = False,
    progress_every: int = 100,
    temperature: Optional[float] = None,
) -> Tuple[float, float]:
    """Estimate logical error rate over multiple simulator batches.

    Each batch: reset → inject errors → (optionally) decode for
    ``num_rounds`` → measure logical error rate.

    Args:
        dem: DetectorErrorModel (must have batch_shape set).
        decoder: Optional decoder. If ``None``, measures raw error rate.
        num_batches: Number of independent batches to evaluate.
        num_rounds: Number of decode-correct rounds per batch.
        progress: Print progress to stdout.
        progress_every: Print every N batches (when progress=True).
        temperature: If set and *decoder* has a ``temperature`` attribute
            (e.g. :class:`~src.core.baseline_decoder.GraphDecoder`),
            use this value for Bernoulli decoding (lower → sharper, near-MAP).

    Returns:
        ``(mean, std)`` logical error rate across batches.
    """
    if num_batches < 1:
        raise ValueError(f"num_batches must be >= 1, got {num_batches}")

    prev_temp: Optional[float] = None
    if (
        decoder is not None
        and temperature is not None
        and hasattr(decoder, "temperature")
    ):
        prev_temp = float(decoder.temperature)  # type: ignore[arg-type]
        decoder.temperature = float(temperature)  # type: ignore[attr-defined]

    sim = Simulator(dem)
    batch_rates = []

    try:
        for batch_idx in range(num_batches):
            sim.reset_errors()
            sim.update_errors()

            if decoder is not None:
                for _ in range(num_rounds):
                    decoder.update_corrections(sim.syndromes)
                    sim.apply_corrections(decoder.corrections)

            ler = sim.logicals.any(dim=-1).float().mean().item()
            batch_rates.append(ler)

            if progress and (batch_idx + 1) % progress_every == 0:
                running_mean = sum(batch_rates) / len(batch_rates)
                print(f"  batch {batch_idx + 1}/{num_batches}: "
                      f"LER={running_mean:.6f}")
    finally:
        if prev_temp is not None and decoder is not None:
            decoder.temperature = prev_temp  # type: ignore[attr-defined]

    rates = torch.tensor(batch_rates, dtype=torch.float)
    return rates.mean().item(), rates.std().item()


# =========================================================================
# Sweep
# =========================================================================

def scan_logical_error_rate(config: Dict[str, Any]) -> None:
    """Sweep logical error rate across code distances and error rates.

    Reads from ``config["evaluation"]``. For each (distance, decoder_type,
    error_rate) triple, estimates the logical error rate and writes results
    to a CSV file.

    Config keys:

    * ``code_type``: ``"surface"`` or ``"repetition"``.
    * ``code_distances``: list of ints.
    * ``decoders``: list of decoder names (``"none"``, ``"mwpm"``, ``"baseline"``).
    * ``baseline_checkpoint``: optional path to trained baseline weights.
    * ``baseline_config``: optional dict of :class:`DecoderConfig` kwargs.
    * ``error_rates`` or ``error_rates_logspace``: rate specification.
    * ``batch_size``, ``num_batches``, ``num_rounds``: eval parameters
      (default ``num_rounds`` is 1).
    * ``eval_temperature``: optional float; for ``baseline`` decoder,
      Bernoulli temperature (lower → sharper / near-MAP sampling).
    * ``skip_existing``: if ``True`` (default), reuse existing per-decoder CSVs
      when present, except baseline CSVs are refreshed when ``baseline_checkpoint``
      is newer than the file. If ``False``, always recompute and overwrite.
    * ``output_dir``: directory for CSV output.
    * ``device``: device string or null.
    * ``progress_level``: 0=silent, 1=distances, 2=batches.

    After each code distance, prints an ASCII table (rows = *p*, columns =
    decoders) for side-by-side comparison. Skipped CSVs are loaded when valid.
    """
    eval_cfg = config.get("evaluation", {})
    code_type = eval_cfg.get("code_type", "surface")
    code_distances = [int(d) for d in eval_cfg["code_distances"]]
    batch_size = int(eval_cfg.get("batch_size", 64))
    num_batches = int(eval_cfg.get("num_batches", 1000))
    num_rounds = int(eval_cfg.get("num_rounds", 1))
    eval_temp = eval_cfg.get("eval_temperature")
    eval_temp_f: Optional[float] = float(eval_temp) if eval_temp is not None else None
    progress_level = int(eval_cfg.get("progress_level", 1))
    progress_every = int(eval_cfg.get("progress_every", 100))
    output_dir = eval_cfg.get("output_dir", "data/evaluation")
    device = _resolve_device(eval_cfg.get("device"))
    error_rates = _build_error_rates(eval_cfg)
    decoders = eval_cfg.get("decoders", ["none"])
    baseline_checkpoint = eval_cfg.get("baseline_checkpoint")
    baseline_config = eval_cfg.get("baseline_config") or {}

    os.makedirs(output_dir, exist_ok=True)

    if progress_level >= 1:
        print(f"Evaluation sweep: {code_type} code")
        print(f"  Distances: {code_distances}")
        print(f"  Decoders: {decoders}")
        print(f"  Error rates: {len(error_rates)} points")
        print(f"  Device: {device}")
        print(f"  Decode rounds / batch: {num_rounds}")
        if eval_temp_f is not None:
            print(f"  Baseline eval_temperature: {eval_temp_f}")

    print_table = bool(eval_cfg.get("print_ler_table", True))

    for dist in code_distances:
        ler_by_decoder: Dict[str, Dict[float, Tuple[float, float]]] = {}

        for decoder_name in decoders:
            safe_name = decoder_name.lower().strip()
            output_path = os.path.join(
                output_dir,
                f"ler_{code_type}_d{dist}_{safe_name}.csv",
            )
            existed = os.path.isfile(output_path)
            if _reuse_cached_ler_csv(
                output_path, safe_name, eval_cfg, baseline_checkpoint
            ):
                ler_by_decoder[safe_name] = _try_load_ler_csv(output_path)
                if progress_level >= 1:
                    print(f"Skipping (exists): {output_path}")
                continue

            results: List[Tuple[float, float, float]] = []
            if progress_level >= 1:
                if existed:
                    if not eval_cfg.get("skip_existing", True):
                        why = "skip_existing=false"
                    else:
                        why = "checkpoint newer than CSV"
                    print(f"\n{code_type} d={dist}, decoder={safe_name} ({why})")
                else:
                    print(f"\n{code_type} d={dist}, decoder={safe_name}")

            for rate_idx, rate in enumerate(error_rates):
                if progress_level >= 1:
                    print(f"  Rate {rate_idx + 1}/{len(error_rates)}: "
                          f"p={rate:.6f}")

                dem = _build_dem(code_type, dist, rate, batch_size, device)
                decoder = build_decoder(
                    safe_name,
                    dem,
                    checkpoint_path=baseline_checkpoint,
                    baseline_config=baseline_config,
                )

                stemp = eval_temp_f if safe_name == "baseline" else None
                mean_ler, std_ler = evaluate_logical_error_rate(
                    dem=dem,
                    decoder=decoder,
                    num_batches=num_batches,
                    num_rounds=num_rounds,
                    progress=(progress_level >= 2),
                    progress_every=progress_every,
                    temperature=stemp,
                )
                results.append((rate, mean_ler, std_ler))

                if progress_level >= 1:
                    print(f"    → LER = {mean_ler:.6f} ± {std_ler:.6f}")

            # Write CSV
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                f.write("error_rate,logical_error_rate_mean,"
                        "logical_error_rate_std\n")
                for er, mean, std in results:
                    f.write(f"{er},{mean},{std}\n")

            ler_by_decoder[safe_name] = {
                er: (mean, std) for er, mean, std in results
            }

            if progress_level >= 1:
                print(f"  Saved: {output_path}")

        if print_table:
            _print_ler_comparison_table(
                code_type, dist, error_rates, decoders, ler_by_decoder
            )


__all__ = [
    "evaluate_logical_error_rate",
    "scan_logical_error_rate",
    "build_decoder",
]
