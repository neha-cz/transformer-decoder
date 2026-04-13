"""Trainer for QEC decoders using the DecoderLoss game-based framework.

The Trainer wraps a :class:`DecoderLoss` instance and owns the optimizer,
training loop, checkpointing, and metric logging. All training modes
(CE, RL, mixed) are expressed via DecoderLoss objectives — no subclasses
needed.

Curriculum features
~~~~~~~~~~~~~~~~~~~

**Phases** — a list of dicts, each with:

* ``epochs`` (int): How many epochs this phase runs.
* ``objective`` (dict): Loss names → weights.
* ``num_rounds`` (int, optional): Correction rounds per game. Defaults to
  the global ``num_rounds``.
* ``error_rate`` (float or [lo, hi], optional): Physical error rate.
  A scalar fixes the rate; a two-element list samples uniformly per step.
* ``lr_scale`` (float, optional): Multiply the base learning rate for
  this phase (e.g., 0.1 for RL). Default 1.0.

**Per-phase LR scheduling** — each phase gets its own independent
warmup + cosine/constant cycle.  The scheduler resets at the start of
every phase so that RL phases after CE pretraining start with a fresh
warm LR instead of inheriting a near-zero decayed value.

**Smooth weight interpolation** is NOT done automatically — the objective
changes at phase boundaries.  Users who want smooth blending should define
many short phases with gradual weight changes, or use the provided
:func:`interpolate_phases` helper.

Usage::

    loss_module = DecoderLoss(decoder, simulator,
        objective={"full_suffix": 1.0},
    )
    config = TrainerConfig(
        steps_per_epoch=100,
        phases=[
            {"epochs": 50, "objective": {"full_suffix": 1.0}},
            {"epochs": 50, "objective": {"full_suffix": 0.1, "syndrome_density": 0.7},
             "num_rounds": 3, "error_rate": [0.04, 0.10], "lr_scale": 0.1},
        ],
    )
    trainer = Trainer(loss_module, config)
    trainer.train()
"""

from __future__ import annotations

import csv
import math
import os
import time
from typing import Any, Dict, List, Optional

import torch
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR

from ..core.player import Config
from .losses import DecoderLoss, Objective


# =========================================================================
# Phase interpolation helper
# =========================================================================

def interpolate_phases(
    start: Dict[str, Any],
    end: Dict[str, Any],
    transition_epochs: int,
) -> List[Dict[str, Any]]:
    """Generate intermediate phases for smooth objective blending.

    Creates ``transition_epochs`` phases that linearly interpolate between
    the ``start`` and ``end`` objectives.  Other phase keys (``num_rounds``,
    ``error_rate``, ``lr_scale``) interpolate linearly too when both sides
    provide them; otherwise they snap to ``end``'s value.

    Args:
        start: Starting phase dict (its ``epochs`` value is ignored).
        end: Ending phase dict (its ``epochs`` value is ignored).
        transition_epochs: Number of 1-epoch phases to generate.

    Returns:
        List of phase dicts, each with ``epochs=1``.

    Example::

        phases = interpolate_phases(
            {"objective": {"full_suffix": 1.0}},
            {"objective": {"full_suffix": 0.1, "syndrome_density": 0.7}},
            transition_epochs=10,
        )
    """
    start_obj = start.get("objective", {})
    end_obj = end.get("objective", {})
    all_keys = set(start_obj) | set(end_obj)

    phases = []
    for i in range(transition_epochs):
        alpha = (i + 1) / transition_epochs
        obj = {}
        for key in all_keys:
            w0 = start_obj.get(key, 0.0)
            w1 = end_obj.get(key, 0.0)
            w = w0 + alpha * (w1 - w0)
            if w > 1e-6:
                obj[key] = round(w, 6)

        phase: Dict[str, Any] = {"epochs": 1, "objective": obj}

        # Interpolate num_rounds (floor to int)
        if "num_rounds" in start and "num_rounds" in end:
            nr = start["num_rounds"] + alpha * (end["num_rounds"] - start["num_rounds"])
            phase["num_rounds"] = max(1, int(nr))
        elif "num_rounds" in end:
            phase["num_rounds"] = end["num_rounds"]

        # Interpolate lr_scale
        if "lr_scale" in start and "lr_scale" in end:
            phase["lr_scale"] = start["lr_scale"] + alpha * (end["lr_scale"] - start["lr_scale"])
        elif "lr_scale" in end:
            phase["lr_scale"] = end["lr_scale"]

        # Interpolate error_rate
        if "error_rate" in start and "error_rate" in end:
            s_er = start["error_rate"]
            e_er = end["error_rate"]
            # Normalize to [lo, hi] form
            s_lo, s_hi = (s_er, s_er) if isinstance(s_er, (int, float)) else (s_er[0], s_er[1])
            e_lo, e_hi = (e_er, e_er) if isinstance(e_er, (int, float)) else (e_er[0], e_er[1])
            lo = s_lo + alpha * (e_lo - s_lo)
            hi = s_hi + alpha * (e_hi - s_hi)
            phase["error_rate"] = round(lo, 6) if abs(lo - hi) < 1e-8 else [round(lo, 6), round(hi, 6)]
        elif "error_rate" in end:
            phase["error_rate"] = end["error_rate"]

        phases.append(phase)
    return phases


# =========================================================================
# Configuration
# =========================================================================

class TrainerConfig(Config):
    """Configuration for :class:`Trainer`.

    Args:
        learning_rate: Peak optimizer learning rate.
        optimizer_type: ``"adam"`` or ``"adamw"``.
        grad_clip: Max gradient norm for clipping (0 = disabled).
        steps_per_epoch: Training steps (game plays) per epoch.
        num_rounds: Default correction rounds per game.
        accumulation_steps: Gradient accumulation micro-steps.
        warmup_epochs: Epochs for linear LR warmup.
        lr_schedule: ``"cosine"`` or ``"constant"`` after warmup.
        min_lr: Minimum LR as fraction of peak (for cosine decay).
        log_every: Log metrics every N epochs (0 = disabled).
        checkpoint_every: Save checkpoint every N epochs (0 = disabled).
        eval_every: Run inline evaluation every N epochs (0 = disabled).
        eval_batches: Number of evaluation batches.
        eval_rounds: Correction rounds per eval game.
        eval_temperature: If set, Bernoulli sampling temperature for the
            student during inline eval (lower = sharper / near-MAP). ``None``
            uses :attr:`DecoderLoss.temperature`.
        checkpoint_dir: Directory for checkpoint files (under data/).
        metrics_path: Path for CSV metrics log (under data/).
        phases: Curriculum phases. Each dict has ``epochs`` (int) and
            ``objective`` (dict), plus optional ``num_rounds``, ``error_rate``,
            ``lr_scale``.
        num_epochs: Total epochs when ``phases`` is ``None``.
        seed: Random seed. ``None`` = no seeding.
    """

    def __init__(
        self,
        learning_rate: float = 2e-4,
        optimizer_type: str = "adamw",
        grad_clip: float = 1.0,
        steps_per_epoch: int = 100,
        num_rounds: int = 1,
        accumulation_steps: int = 1,
        use_amp: bool = False,
        warmup_epochs: int = 0,
        lr_schedule: str = "cosine",
        min_lr: float = 0.01,
        log_every: int = 1,
        checkpoint_every: int = 50,
        eval_every: int = 0,
        eval_batches: int = 100,
        eval_rounds: int = 1,
        eval_temperature: Optional[float] = None,
        checkpoint_dir: str = "data/checkpoints",
        metrics_path: str = "data/metrics.csv",
        phases: Optional[List[Dict[str, Any]]] = None,
        num_epochs: int = 100,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.grad_clip = grad_clip
        self.steps_per_epoch = steps_per_epoch
        self.num_rounds = num_rounds
        self.accumulation_steps = accumulation_steps
        self.use_amp = use_amp
        self.warmup_epochs = warmup_epochs
        self.lr_schedule = lr_schedule
        self.min_lr = min_lr
        self.log_every = log_every
        self.checkpoint_every = checkpoint_every
        self.eval_every = eval_every
        self.eval_batches = eval_batches
        self.eval_rounds = eval_rounds
        self.eval_temperature = eval_temperature
        self.checkpoint_dir = checkpoint_dir
        self.metrics_path = metrics_path
        self.phases = phases
        self.num_epochs = num_epochs
        self.seed = seed


# =========================================================================
# Trainer
# =========================================================================

class Trainer:
    """Training loop wrapping :class:`DecoderLoss`.

    Owns optimizer, LR scheduler, training loop, curriculum phases,
    checkpointing, and metric logging. DecoderLoss owns game play,
    loss computation, and references to decoder / simulator.

    Per-phase features:

    * **Objective switching** — ``loss_module.set_objective(phase["objective"])``
    * **num_rounds override** — per-phase correction rounds
    * **Error rate randomization** — ``error_rate: [lo, hi]`` or scalar
    * **LR scaling** — ``lr_scale: 0.1`` reduces LR for RL phases

    Args:
        loss_module: A :class:`DecoderLoss` instance.
        config: Training configuration.
    """

    def __init__(
        self,
        loss_module: DecoderLoss,
        config: Optional[TrainerConfig] = None,
    ):
        self.loss_module = loss_module
        self.decoder = loss_module.decoder
        self.config = config or TrainerConfig()
        self.device = next(self.decoder.parameters()).device

        self.optimizer = self._create_optimizer()
        self.scheduler: Optional[LambdaLR] = None
        self.epoch = 0
        self.phase_idx = 0
        self._best_eval_ler = float("inf")
        self._metrics_header_written = False

        # AMP (automatic mixed precision)
        self.use_amp = self.config.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Per-phase runtime state
        self._current_num_rounds = self.config.num_rounds
        self._current_error_rate = None  # None = use simulator default
        self._current_lr_scale = 1.0

    # ----- Optimizer & scheduler -------------------------------------------

    def _create_optimizer(self) -> Optimizer:
        params = list(self.decoder.parameters())
        if not params:
            raise ValueError("Decoder has no parameters to optimize")
        opt = self.config.optimizer_type.lower()
        if opt == "adam":
            return Adam(params, lr=self.config.learning_rate)
        if opt == "adamw":
            return AdamW(params, lr=self.config.learning_rate)
        raise ValueError(f"Unsupported optimizer: {opt!r}. Use 'adam' or 'adamw'.")

    def _create_phase_scheduler(
        self, phase_epochs: int, lr_scale: float = 1.0,
    ) -> Optional[LambdaLR]:
        """Build a fresh LR scheduler for one phase.

        Each phase gets its own independent warmup + cosine/constant cycle.
        This prevents RL phases from inheriting near-zero LR from a
        decayed CE phase.

        Args:
            phase_epochs: Number of epochs in this phase.
            lr_scale: Multiply the base LR for this phase (e.g. 0.1 for RL).
        """
        warmup = self.config.warmup_epochs
        if warmup <= 0 and self.config.lr_schedule == "constant":
            # Still apply lr_scale via a trivial scheduler
            if lr_scale == 1.0:
                return None
            return LambdaLR(self.optimizer, lambda epoch: lr_scale)

        min_lr_ratio = self.config.min_lr
        schedule = self.config.lr_schedule

        def lr_lambda(epoch: int) -> float:
            # Warmup (capped to phase length)
            phase_warmup = min(warmup, phase_epochs)
            if epoch < phase_warmup:
                base = (epoch + 1) / max(phase_warmup, 1)
            elif schedule == "constant":
                base = 1.0
            else:
                decay_epochs = max(phase_epochs - phase_warmup, 1)
                progress = (epoch - phase_warmup) / decay_epochs
                base = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
                    1.0 + math.cos(math.pi * progress)
                )
            return base * lr_scale

        return LambdaLR(self.optimizer, lr_lambda)

    # ----- Phases ----------------------------------------------------------

    def _build_phases(self) -> List[Dict[str, Any]]:
        """Resolve phase schedule from config."""
        if self.config.phases:
            return list(self.config.phases)
        return [{
            "epochs": self.config.num_epochs,
            "objective": dict(self.loss_module.objective),
        }]

    def _apply_phase(self, phase: Dict[str, Any]) -> None:
        """Apply phase settings: objective, num_rounds, error_rate, lr_scale,
        temperature, entropy_coeff, code_distances, teacher."""
        objective = phase.get("objective")
        if objective is not None:
            self.loss_module.set_objective(objective)

        self._current_num_rounds = phase.get("num_rounds", self.config.num_rounds)
        self._current_error_rate = phase.get("error_rate", None)
        self._current_lr_scale = phase.get("lr_scale", 1.0)
        self._current_code_distances = phase.get("code_distances", None)

        # KL penalty: snapshot reference policy when entering RL phase
        if "kl_penalty_coeff" in phase:
            coeff = float(phase["kl_penalty_coeff"])
            self.loss_module.kl_penalty_coeff = coeff
            if coeff > 0 and self.loss_module._reference_state is None:
                self.loss_module.snapshot_reference_policy()
                print(f"    Snapshotted reference policy for KL penalty (coeff={coeff})")

        # Per-phase teacher override: "none"/"self" removes teacher, "mwpm" sets MWPM
        if "teacher" in phase:
            teacher_name = phase["teacher"]
            if teacher_name in (None, "none", "self", "null", ""):
                self.loss_module.teacher = None
            elif teacher_name == "mwpm":
                from ..core.mwpm_decoder import MWPMDecoder
                self.loss_module.teacher = MWPMDecoder(self.loss_module.simulator.dem)

        # RL exploration knobs — per-phase override
        if "temperature" in phase:
            self.loss_module.temperature = float(phase["temperature"])
        if "entropy_coeff" in phase:
            self.loss_module.entropy_coeff = float(phase["entropy_coeff"])
            # Update existing RL modules
            for mod in self.loss_module._loss_modules.values():
                if hasattr(mod, "entropy_coeff"):
                    mod.entropy_coeff = float(phase["entropy_coeff"])

    def _sample_error_rate(self) -> Optional[float]:
        """Sample an error rate for the current step.

        Returns None if no error_rate is set (use simulator default).
        """
        er = self._current_error_rate
        if er is None:
            return None
        if isinstance(er, (list, tuple)):
            lo, hi = float(er[0]), float(er[1])
            return lo + (hi - lo) * torch.rand(1).item()
        return float(er)

    def _set_simulator_error_rate(self, rate: Optional[float]) -> None:
        """Update the simulator's error rate if specified."""
        if rate is not None:
            self.loss_module.simulator.update_error_model(error_rate=rate)

    def _sample_and_rebind_distance(self) -> None:
        """If code_distances is set, sample a random distance and rebind all components."""
        dists = self._current_code_distances
        if dists is None:
            return
        d = dists[torch.randint(len(dists), (1,)).item()]
        sim = self.loss_module.simulator
        old_dem = sim.dem
        if getattr(old_dem, 'code_distance', None) == d:
            return  # already bound to this distance
        # Build new DEM at sampled distance
        from ..base.codes import repetition_code, surface_code
        code_type = getattr(old_dem, 'code_type', 'surface')
        er = self._sample_error_rate()
        if er is None:
            er = 0.1
        batch_size = old_dem.batch_shape[0] if old_dem.batch_shape else 1
        device = self.device
        if code_type in ('repetition', 'repetition_code'):
            new_dem = repetition_code(d, er, batch_size=batch_size, device=device)
        else:
            new_dem = surface_code(d, er, batch_size=batch_size, device=device)
        # Rebind decoder, simulator, teacher
        self.decoder.bind(new_dem)
        sim.__init__(new_dem)
        if self.loss_module.teacher is not None:
            from ..core.mwpm_decoder import MWPMDecoder
            if isinstance(self.loss_module.teacher, MWPMDecoder):
                self.loss_module.teacher = MWPMDecoder(new_dem)

    # ----- Training loop ---------------------------------------------------

    def train_step(self) -> Dict[str, float]:
        """Run one training step: play game, compute loss, backprop.

        Supports AMP (automatic mixed precision) when ``use_amp=True``
        and device is CUDA. The forward pass runs in float16, backward
        uses GradScaler to avoid underflow.
        """
        self.optimizer.zero_grad()
        accum_n = self.config.accumulation_steps
        accum_metrics: Dict[str, float] = {}

        for micro in range(accum_n):
            # Randomize distance and error rate per micro-step
            self._sample_and_rebind_distance()
            rate = self._sample_error_rate()
            self._set_simulator_error_rate(rate)

            with torch.amp.autocast(
                self.device.type, enabled=self.use_amp,
            ):
                loss, metrics = self.loss_module(
                    num_rounds=self._current_num_rounds,
                )

            scaled_loss = loss / accum_n
            if self.scaler is not None:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            for k, v in metrics.items():
                accum_metrics[k] = accum_metrics.get(k, 0.0) + v / accum_n

        if self.scaler is not None:
            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.decoder.parameters(), self.config.grad_clip,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.decoder.parameters(), self.config.grad_clip,
                )
            self.optimizer.step()
        return accum_metrics

    def train_epoch(self) -> Dict[str, float]:
        """Run one epoch of training steps. Returns averaged metrics."""
        self.decoder.train()
        accum: Dict[str, float] = {}
        for _ in range(self.config.steps_per_epoch):
            metrics = self.train_step()
            for k, v in metrics.items():
                accum[k] = accum.get(k, 0.0) + v

        n = self.config.steps_per_epoch
        avg = {k: v / n for k, v in accum.items()}

        # Track current LR
        if self.scheduler is not None:
            avg["lr"] = self.scheduler.get_last_lr()[0]
        else:
            avg["lr"] = self.config.learning_rate

        return avg

    def train(self) -> List[Dict[str, float]]:
        """Main training loop with phase transitions.

        Each phase gets its own fresh LR scheduler (warmup + cosine cycle).
        This prevents RL phases from inheriting near-zero decayed LR from
        previous CE phases.

        Respects ``self.epoch`` and ``self.phase_idx`` for resume support.

        Returns:
            List of per-epoch metric dicts.
        """
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)

        phases = self._build_phases()
        all_metrics: List[Dict[str, float]] = []
        total_epochs = sum(p["epochs"] for p in phases)

        print(f"Training: {total_epochs} epochs across {len(phases)} phase(s)")
        print(f"  Device: {self.device}")
        print(f"  Steps/epoch: {self.config.steps_per_epoch}")
        print(f"  Num rounds: {self.config.num_rounds}")
        print(f"  RL algorithm: {self.loss_module.rl_algorithm}")
        if self.loss_module.teacher is not None:
            print(
                "  Note: per-epoch ler_t = teacher rollout LER (MWPM path); "
                "[Eval] LER = student rollout (matches sweep)."
            )
        if self.config.accumulation_steps > 1:
            print(f"  Accumulation steps: {self.config.accumulation_steps}")
        if self.config.warmup_epochs > 0:
            print(f"  Warmup: {self.config.warmup_epochs} epochs (per phase)")
        print(f"  LR schedule: {self.config.lr_schedule}")
        if self.use_amp:
            print(f"  AMP: enabled (float16)")
        if self.epoch > 0:
            print(f"  Resuming from epoch {self.epoch}")

        phase_end_epoch = 0
        for phase_idx, phase in enumerate(phases):
            phase_end_epoch += phase["epochs"]

            if phase_idx < self.phase_idx:
                continue

            self.phase_idx = phase_idx
            self._apply_phase(phase)

            obj = phase.get("objective", {})
            nr = phase.get("num_rounds", self.config.num_rounds)
            er = phase.get("error_rate", "default")
            lr_s = phase.get("lr_scale", 1.0)
            obj_str = " ".join(f"{k}={v}" for k, v in obj.items())
            print(f"\n--- Phase {phase_idx + 1}/{len(phases)} "
                  f"({phase['epochs']}ep T={nr} p={er} lr×{lr_s}) ---")
            print(f"    {obj_str}")

            # Fresh scheduler for this phase
            phase_epochs = phase["epochs"]
            self.scheduler = self._create_phase_scheduler(phase_epochs, lr_s)

            # Fast-forward scheduler if resuming mid-phase
            phase_start_epoch = phase_end_epoch - phase_epochs
            epochs_done_in_phase = max(0, self.epoch - phase_start_epoch)
            if epochs_done_in_phase > 0 and self.scheduler is not None:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", "Detected call of.*lr_scheduler",
                    )
                    for _ in range(epochs_done_in_phase):
                        self.scheduler.step()

            # Reset RL baselines at phase boundary so EMA doesn't carry
            # stale values from a different objective/error_rate regime
            if phase_idx > 0:
                self._reset_rl_baselines()

            for epoch_in_phase in range(phase_epochs):
                target_epoch = phase_start_epoch + epoch_in_phase + 1
                if target_epoch <= self.epoch:
                    continue

                self.epoch = target_epoch
                t0 = time.time()
                epoch_metrics = self.train_epoch()
                dt = time.time() - t0
                epoch_metrics["epoch"] = self.epoch
                epoch_metrics["phase"] = phase_idx
                epoch_metrics["num_rounds"] = self._current_num_rounds
                epoch_metrics["time_s"] = dt
                all_metrics.append(epoch_metrics)

                if self.scheduler is not None:
                    self.scheduler.step()

                if self.config.log_every > 0 and self.epoch % self.config.log_every == 0:
                    self._log_metrics(epoch_metrics)

                self._write_metrics_csv(epoch_metrics)

                if (self.config.checkpoint_every > 0
                        and self.epoch % self.config.checkpoint_every == 0):
                    self.save_checkpoint()

                if (self.config.eval_every > 0
                        and self.epoch % self.config.eval_every == 0):
                    eval_metrics = self.evaluate()
                    self._log_eval(eval_metrics)
                    ler = eval_metrics.get("eval_logical_error_rate", float("inf"))
                    if ler < self._best_eval_ler:
                        self._best_eval_ler = ler
                        self.save_checkpoint(tag="best")
                        print(f"  [Best] New best LER={ler:.6f}")

        self.save_checkpoint(tag="final")
        print(f"\nTraining complete. {self.epoch} epochs.")
        return all_metrics

    def _reset_rl_baselines(self) -> None:
        """Reset EMA baselines in RL loss modules at phase boundaries.

        Prevents stale baseline values from a previous phase (different
        objective/error_rate) from killing the advantage signal.
        """
        for module in self.loss_module._loss_modules.values():
            if hasattr(module, "baseline"):
                module.baseline.zero_()

    # ----- Inline evaluation -----------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Estimate LER / syndrome density by rolling out the **student** decoder.

        Uses ``DecoderLoss.play(..., student_rollout=True)`` so metrics are not
        taken from the MWPM teacher when teacher forcing is enabled.
        When :attr:`TrainerConfig.eval_temperature` is set, passes it as
        ``temperature=`` to sharpen Bernoulli decoding during eval.
        """
        self.decoder.eval()
        accum: Dict[str, float] = {}
        n = self.config.eval_batches
        play_kw: Dict[str, Any] = dict(
            num_rounds=self.config.eval_rounds,
            student_rollout=True,
        )
        if self.config.eval_temperature is not None:
            play_kw["temperature"] = float(self.config.eval_temperature)

        for _ in range(n):
            record = self.loss_module.play(**play_kw)
            logical_err = record.logicals.any(dim=-1).float().mean().item()
            syn_density = record.residual_syndromes.float().mean().item()
            accum["eval_logical_error_rate"] = (
                accum.get("eval_logical_error_rate", 0.0) + logical_err
            )
            accum["eval_syndrome_density"] = (
                accum.get("eval_syndrome_density", 0.0) + syn_density
            )
        self.decoder.train()
        return {k: v / n for k, v in accum.items()}

    # ----- Checkpointing ---------------------------------------------------

    def save_checkpoint(self, tag: Optional[str] = None) -> str:
        """Save model, optimizer, scheduler, and training state."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        label = tag or f"epoch_{self.epoch}"
        path = os.path.join(self.config.checkpoint_dir, f"checkpoint_{label}.pt")
        ckpt = {
            "model_state_dict": self.decoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "phase_idx": self.phase_idx,
            "best_eval_ler": self._best_eval_ler,
            "config": self.config.to_dict(),
        }
        # Save model config if the decoder exposes one (e.g., DecoderConfig)
        decoder_cfg = getattr(self.decoder, "config", None)
        if decoder_cfg is not None and hasattr(decoder_cfg, "to_dict"):
            ckpt["model_config"] = decoder_cfg.to_dict()
        if self.scheduler is not None:
            ckpt["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(ckpt, path)
        print(f"  Checkpoint saved: {path}")
        return path

    def load_checkpoint(self, path: str) -> None:
        """Load model, optimizer, scheduler, and training state."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.decoder.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.epoch = ckpt.get("epoch", 0)
        self.phase_idx = ckpt.get("phase_idx", 0)
        self._best_eval_ler = ckpt.get("best_eval_ler", float("inf"))
        if self.scheduler is not None and "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        print(f"  Resumed from {path} (epoch {self.epoch})")

    # ----- Logging ---------------------------------------------------------

    # Short aliases for compact logging
    _LOG_ALIASES = {
        "full_suffix": "fs",
        "corrections": "cor",
        "syndromes": "syn_ce",
        "syndrome_density": "sd",
        "logical_error_rate": "ler_rl",
    }

    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        """Print compact one-line metrics to stdout.

        Format: ``E 101 L=1.47 ler=.062 sd=.11 lr=6.7e-5 [fs=1.38 cor=1.53 syn_ce=1.44] 3.8s``
        """
        epoch = int(metrics.get("epoch", self.epoch))
        loss = metrics.get("total_loss", 0.0)
        ler = metrics.get("eval/logical_error_rate", 0.0)
        syn = metrics.get("eval/residual_syndrome_rate", 0.0)
        lr = metrics.get("lr", 0.0)
        dt = metrics.get("time_s", 0.0)
        nr = int(metrics.get("num_rounds", 1))
        ler_key = (
            "ler_t"
            if self.loss_module.teacher is not None
            else "ler"
        )

        # Core metrics (always shown)
        line = f"  E{epoch:4d} L={loss:.3f} {ler_key}={ler:.3f} sd={syn:.3f} lr={lr:.1e}"
        if nr > 1:
            line += f" T={nr}"

        # Per-objective losses in brackets (compact aliases)
        skip = {"total_loss", "epoch", "phase", "time_s", "lr", "num_rounds",
                "eval/residual_syndrome_rate", "eval/logical_error_rate"}
        obj_parts = []
        for k, v in sorted(metrics.items()):
            if k not in skip:
                alias = self._LOG_ALIASES.get(k, k)
                obj_parts.append(f"{alias}={v:.2f}")
        if obj_parts:
            line += f" [{' '.join(obj_parts)}]"

        line += f" {dt:.1f}s"
        print(line)

    def _log_eval(self, metrics: Dict[str, float]) -> None:
        """Print evaluation metrics."""
        ler = metrics.get("eval_logical_error_rate", 0.0)
        syn = metrics.get("eval_syndrome_density", 0.0)
        print(f"  [Eval] LER={ler:.4f} | syn_density={syn:.4f} (student rollout)")

    def _write_metrics_csv(self, metrics: Dict[str, float]) -> None:
        """Append metrics to CSV file."""
        if not self.config.metrics_path:
            return
        path = self.config.metrics_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        keys = sorted(metrics.keys())
        if not self._metrics_header_written or not os.path.exists(path):
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerow(metrics)
            self._metrics_header_written = True
        else:
            with open(path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writerow(metrics)


__all__ = ["TrainerConfig", "Trainer", "interpolate_phases"]
