"""Game-based training framework for QEC decoders.

The decoder and simulator play a QEC game:

1. Simulator resets and generates initial errors.
2. For each round: decoder proposes corrections, simulator applies them.
3. GameRecord holds the full trajectory and lazily serves derived data.
4. Loss classes read from GameRecord — they never touch decoder or simulator.

Architecture::

    DecoderLoss.play(num_rounds)           ← no_grad, public interfaces only
        → GameRecord(trajectory, decoder, simulator)

    GameRecord                             ← trajectory + lazy cached properties
        .logits  → single decoder.forward() → (suffix_logits, suffix_targets)
        .error_logits                      ← slice of suffix_logits at error nodes
        .log_probs                         ← full-vocab Bernoulli from error_logits
        .syndrome_density_reward           ← pure tensor math
        .logical_error_reward              ← pure tensor math

    SupervisedLoss(record) → scalar        ← reads record properties
    ReinforceLoss(record) → scalar         ← REINFORCE with EMA baseline
    GRPOLoss(record) → scalar              ← Group Relative Policy Optimization

Objective dict
~~~~~~~~~~~~~~

All training modes are specified via a single ``objective`` dict mapping
loss names to weights::

    {"full_suffix": 1.0, "corrections": 0.5, "syndrome_density": 0.3}

DecoderLoss dispatches each key to the appropriate loss class:

* CE modes:  ``"full_suffix"``, ``"corrections"``, ``"syndromes"``
* RL modes:  ``"syndrome_density"``, ``"logical_error_rate"``

Teacher-student design
~~~~~~~~~~~~~~~~~~~~~~

When a ``teacher`` (any :class:`Decoder`) is provided to
:class:`DecoderLoss`, the teacher plays the game and the student
observes. The student's ``forward()`` is teacher-forced with the
teacher's corrections, ensuring logits and targets are consistent.
The loss module is teacher-agnostic — it treats all teachers uniformly.

RL algorithm selection
~~~~~~~~~~~~~~~~~~~~~~

The ``rl_algorithm`` parameter on :class:`DecoderLoss` selects which
policy gradient estimator is used for RL-mode objectives:

* ``"reinforce"`` — vanilla REINFORCE with EMA baseline (default).
* ``"grpo"``      — Group Relative Policy Optimization. Requires
  ``grpo_group_size`` (K) samples per syndrome. Advantage is computed
  relative to the group mean — no value network needed. Supports
  optional ``clip_range`` for PPO-style clipping on top of group
  normalization.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.decoder import Decoder
from ..core.simulator import Simulator


# Known loss names by category
CE_MODES = frozenset({"full_suffix", "corrections", "syndromes"})
RL_MODES = frozenset({"syndrome_density", "logical_error_rate"})
ALL_MODES = CE_MODES | RL_MODES
RL_ALGORITHMS = frozenset({"reinforce", "grpo"})

Objective = Dict[str, float]


# =========================================================================
# GameRecord — trajectory + lazy data server
# =========================================================================

class GameRecord:
    """Full game trajectory with lazy cached properties for training data.

    Stores the raw MDP trajectory from T rounds of play. Holds a reference
    to the decoder for lazy evaluation of training-specific quantities
    (logits, world-model targets).

    **Raw trajectory** (populated by :meth:`DecoderLoss.play`):

    * ``syndromes``   — ``[*B, T+1, num_det]`` (T+1 snapshots: s0…sT)
    * ``corrections`` — ``[*B, T, num_err]``
    * ``logicals``    — ``[*B, T, num_log]``

    **Views** (zero-cost slices):

    * ``input_syndromes``    — ``syndromes[…, :-1, :]`` = ``[*B, T, num_det]``
    * ``residual_syndromes`` — ``syndromes[…, 1:, :]``  = ``[*B, T, num_det]``

    **Cached properties** (computed once on first access):

    * ``logits``   — single ``decoder.forward()`` call → ``(suffix_logits, suffix_targets)``
    * ``suffix_logits``, ``suffix_targets`` — full-vocab logits and target token ids
    * ``error_logits`` — slice of ``suffix_logits`` at error nodes, position 0
    * ``log_probs`` — full-vocab Bernoulli from ``error_logits``
    * ``syndrome_density_reward``, ``logical_error_reward`` — pure tensor math

    For GRPO, the record may have a group dimension K (first batch dim):

    * ``[K, *B, T, ...]`` where K is the group size.

    Invariant: :class:`SupervisedLoss` and RL loss classes only
    read properties from this class — they never call decoder or simulator.
    """

    # Known reward names
    REWARDS = frozenset({"syndrome_density", "logical_error_rate"})

    def __init__(
        self,
        syndromes: torch.Tensor,
        corrections: torch.Tensor,
        logicals: torch.Tensor,
        decoder: Decoder,
    ):
        # Raw trajectory
        self.syndromes = syndromes      # [*B, T+1, num_det]
        self.corrections = corrections  # [*B, T, num_err]
        self.logicals = logicals        # [*B, T, num_log]
        # Reference for lazy eval (only used by cached properties)
        self.decoder = decoder

    # ----- Views (zero-cost slices) -----------------------------------------

    @cached_property
    def input_syndromes(self) -> torch.Tensor:
        """``[*B, T, num_det]`` — syndromes the decoder sees each round."""
        return self.syndromes[..., :-1, :]

    @cached_property
    def residual_syndromes(self) -> torch.Tensor:
        """``[*B, T, num_det]`` — syndromes after correction each round.

        For rounds ``t < T-1``, this includes post-correction + new noise
        (from ``sim.update_errors()``). For the last round ``t = T-1``,
        this is purely post-correction (no noise injection). Both are
        valid for reward computation — the last round measures the clean
        effect of the decoder's final correction.
        """
        return self.syndromes[..., 1:, :]

    @property
    def num_rounds(self) -> int:
        return self.corrections.shape[-2]

    # ----- Logits (single decoder.forward, cached) --------------------------

    @cached_property
    def logits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single ``decoder.forward()`` call — source of all derived logits.

        Returns ``(suffix_logits, suffix_targets)``.
        T is folded into ``*B`` — all rounds processed in parallel.

        Teacher forcing uses ``corrections`` — which are the teacher's
        corrections when a teacher is present, or the student's own
        corrections during self-play. This ensures logits and targets
        are always consistent.
        """
        return self.decoder.forward(
            self.input_syndromes,
            self.corrections,           # teacher forcing
            self.residual_syndromes,    # teacher forcing: residual syndromes
        )

    @cached_property
    def suffix_logits(self) -> torch.Tensor:
        """``[*B, T, V, G, vocab]`` — full-vocab logits over entire suffix."""
        return self.logits[0]

    @cached_property
    def suffix_targets(self) -> torch.Tensor:
        """``[*B, T, V, G]`` — target token ids for full-vocab CE."""
        return self.logits[1]

    @cached_property
    def error_logits(self) -> torch.Tensor:
        """``[*B, T, num_bits, vocab]`` — full-vocab logits at error nodes, position 0."""
        return self.suffix_logits[..., :self.decoder.num_bits, 0, :]

    # ----- Log probs (full-vocab, matching parser semantics) -----------------

    @cached_property
    def log_probs(self) -> torch.Tensor:
        """``[*B, T]`` — log p(corrections | syndromes), per-round.

        Uses full-vocab softmax to match ``generate()``'s sampling + parser:
        ``p(corr=1) = p(token_one)``, ``p(corr=0) = 1 - p(token_one)``.
        This is a Bernoulli parameterized by the full-vocab probability of
        ``token_one``, consistent with the parser treating everything except
        ``token_one`` as correction=0.
        """
        probs = F.softmax(self.error_logits, dim=-1)  # [*B, T, num_bits, vocab]
        p_one = probs[..., self.decoder.token_one]     # [*B, T, num_bits]

        log_p = torch.where(
            self.corrections.bool(),
            p_one.clamp(min=1e-8).log(),
            (1 - p_one).clamp(min=1e-8).log(),
        )  # [*B, T, num_bits]
        return log_p.sum(dim=-1)  # [*B, T]

    # ----- Entropy (for exploration bonus) ------------------------------------

    @cached_property
    def action_entropy(self) -> torch.Tensor:
        """``[*B, T]`` — per-round Bernoulli entropy of the correction policy.

        For each error bit, entropy is ``-p*log(p) - (1-p)*log(1-p)`` where
        ``p = softmax(error_logits)[token_one]``. Summed over bits, averaged
        would give per-bit entropy. We sum over bits to match ``log_probs``
        convention.

        High entropy = exploring diverse corrections.
        Low entropy = deterministic (collapsed) policy.
        """
        probs = F.softmax(self.error_logits, dim=-1)       # [*B, T, num_bits, vocab]
        p_one = probs[..., self.decoder.token_one]          # [*B, T, num_bits]
        p_one = p_one.clamp(1e-8, 1 - 1e-8)
        bit_entropy = -(p_one * p_one.log() + (1 - p_one) * (1 - p_one).log())
        return bit_entropy.sum(dim=-1)  # [*B, T]

    # ----- Rewards (pure tensor math) ---------------------------------------

    @cached_property
    def syndrome_density_reward(self) -> torch.Tensor:
        """``[*B, T]`` — fraction of syndromes cleared (dense surrogate)."""
        return 1.0 - self.residual_syndromes.float().mean(dim=-1)

    @cached_property
    def logical_error_reward(self) -> torch.Tensor:
        """``[*B, T]`` — 1 if no logical error, 0 otherwise (sparse)."""
        return 1.0 - self.logicals.any(dim=-1).float()

    def reward(self, name: str) -> torch.Tensor:
        """Select reward tensor by name.

        Args:
            name: ``"syndrome_density"`` or ``"logical_error_rate"``.

        Returns:
            ``[*B, T]`` reward tensor.
        """
        if name == "syndrome_density":
            return self.syndrome_density_reward
        elif name == "logical_error_rate":
            return self.logical_error_reward
        raise ValueError(
            f"Unknown reward: {name!r}. Known: {sorted(self.REWARDS)}"
        )


# =========================================================================
# SupervisedLoss
# =========================================================================

class SupervisedLoss(nn.Module):
    """Full-vocab CE loss that reads from :class:`GameRecord`.

    All modes use full-vocab CE (suppresses garbage tokens). The mode
    selects which nodes to supervise:

    * ``"full_suffix"``: CE on entire suffix (all nodes, all positions).
    * ``"corrections"``: CE on error-node logits, position 0 only.
    * ``"syndromes"``: CE on check-node logits, position 0 only.

    The supervision targets come from ``record.corrections``, which are
    the **teacher's** corrections when a teacher is provided to
    :class:`DecoderLoss`, or the student's own corrections during
    self-play. The loss module is teacher-agnostic — it does not know
    or care what kind of teacher produced the corrections.

    Args:
        mode: Supervision scope.
        pos_weight: Weight multiplier for correction=1 positions.
    """

    def __init__(self, mode: str = "full_suffix", pos_weight: float = 1.0):
        super().__init__()
        self.mode = mode
        self.pos_weight = pos_weight

    def forward(self, record: GameRecord) -> torch.Tensor:
        """Compute full-vocab CE from the game record.

        When ``pos_weight > 1`` and mode is an alt-teacher or ``corrections``,
        positions where the target is ``token_one`` (correction=1) receive
        ``pos_weight`` times the loss of other positions.  This compensates
        for class imbalance at low physical error rates where correction=1
        is rare (~p of positions).

        Args:
            record: A :class:`GameRecord` with cached logits.

        Returns:
            Scalar CE loss.
        """
        logits = record.suffix_logits       # [*B, T, V, G, vocab]
        num_bits = record.decoder.num_bits

        # Select targets based on mode.
        # When MWPM corrections are available, record.suffix_targets
        # already uses them (via teacher_corrections → build_suffix).
        # So "mwpm" and "corrections" modes are identical slicing —
        # the difference is which corrections the GameRecord was built with.
        targets = record.suffix_targets  # [*B, T, V, G]

        if getattr(record.decoder, "is_binary_correction_decoder", False):
            if self.mode != "corrections":
                raise ValueError(
                    f"Decoder only supports supervised mode 'corrections', not {self.mode!r}. "
                    "Use RL objectives for other signals, or a different decoder."
                )

        if self.mode == "corrections":
            logits = logits[..., :num_bits, 0:1, :]
            targets = targets[..., :num_bits, 0:1]
        elif self.mode == "syndromes":
            logits = logits[..., num_bits:, 0:1, :]
            targets = targets[..., num_bits:, 0:1]
        elif self.mode != "full_suffix":
            raise ValueError(f"Unknown mode: {self.mode!r}")

        vocab = logits.shape[-1]
        flat_logits = logits.reshape(-1, vocab)
        flat_targets = targets.reshape(-1)

        # Apply class weighting for correction positions
        if self.pos_weight != 1.0 and self.mode == "corrections":
            token_one = record.decoder.token_one
            t1 = token_one.item() if hasattr(token_one, "item") else int(token_one)
            per_sample = F.cross_entropy(
                flat_logits, flat_targets, reduction="none",
            )
            weights = torch.where(
                flat_targets == t1,
                self.pos_weight,
                1.0,
            )
            return (per_sample * weights).sum() / weights.sum()

        return F.cross_entropy(flat_logits, flat_targets)


# =========================================================================
# ReinforceLoss — vanilla REINFORCE with EMA baseline
# =========================================================================

class ReinforceLoss(nn.Module):
    """REINFORCE policy gradient with EMA baseline and entropy bonus.

    Loss: ``-E[A * log_prob] - entropy_coeff * E[H(policy)]``

    The entropy bonus encourages exploration by penalizing
    deterministic (collapsed) policies. Without it, the model may
    converge to a single correction pattern per syndrome and lose
    the within-batch diversity needed for REINFORCE to learn.

    Args:
        reward_fn: Which reward signal to use.
        ema_decay: EMA decay for baseline (default 0.99).
        entropy_coeff: Weight on entropy bonus (default 0.0 = off).
            Typical values: 0.01–0.1.
    """

    def __init__(
        self,
        reward_fn: Literal["syndrome_density", "logical_error_rate"] = "syndrome_density",
        ema_decay: float = 0.99,
        entropy_coeff: float = 0.0,
    ):
        super().__init__()
        self.reward_fn = reward_fn
        self.register_buffer("baseline", torch.tensor(0.0))
        self._ema_decay = ema_decay
        self.entropy_coeff = entropy_coeff

    def forward(self, record: GameRecord) -> torch.Tensor:
        """Compute REINFORCE policy gradient loss."""
        rewards = record.reward(self.reward_fn)

        # Average over rounds, keep batch
        rewards_mean = rewards.mean(dim=-1)    # [*B]
        log_probs = record.log_probs.sum(dim=-1)  # [*B] — sum over rounds

        # Move baseline to reward device if needed (handles device mismatch
        # when loss module is created on CPU but data is on GPU/MPS)
        if self.baseline.device != rewards_mean.device:
            self.baseline = self.baseline.to(rewards_mean.device)

        # Update EMA baseline
        with torch.no_grad():
            self.baseline.lerp_(rewards_mean.mean(), 1.0 - self._ema_decay)

        advantage = (rewards_mean - self.baseline).detach()
        pg_loss = -(advantage * log_probs).mean()

        # Entropy bonus: maximize entropy → explore diverse corrections
        if self.entropy_coeff > 0:
            entropy = record.action_entropy.mean()  # scalar
            return pg_loss - self.entropy_coeff * entropy
        return pg_loss


# =========================================================================
# GRPOLoss — Group Relative Policy Optimization
# =========================================================================

class GRPOLoss(nn.Module):
    """Group Relative Policy Optimization (DeepSeek-Math, 2024).

    For each syndrome, K correction chains are sampled (the "group").
    Advantage is computed relative to the group mean reward — no value
    network or EMA baseline needed. The group normalization provides
    natural variance reduction.

    The group dimension is expected as the **first** batch dimension
    of the record's tensors: ``[K, *B, T, ...]``.

    Advantage for sample k::

        A_k = (r_k - mean(r)) / (std(r) + eps)

    Loss::

        L = -mean_k[ A_k * log_prob_k ]

    Args:
        reward_fn: Which reward signal to use.
        eps: Stability constant for advantage normalization.
        clip_range: Reserved for future use (PPO-style clipping on top
            of GRPO). Currently unused.
        entropy_coeff: Weight on entropy bonus (default 0.0 = off).
    """

    def __init__(
        self,
        reward_fn: Literal["syndrome_density", "logical_error_rate"] = "syndrome_density",
        eps: float = 1e-8,
        clip_range: float = 0.0,
        entropy_coeff: float = 0.0,
    ):
        super().__init__()
        self.reward_fn = reward_fn
        self.eps = eps
        self.clip_range = clip_range
        self.entropy_coeff = entropy_coeff

    def forward(self, record: GameRecord) -> torch.Tensor:
        """Compute GRPO loss from a grouped record.

        Expects record tensors with shape ``[K, *B, T, ...]`` where K
        is the group size (dim 0).
        """
        rewards = record.reward(self.reward_fn)  # [K, *B, T]

        # Average over rounds, keep group and batch
        rewards_mean = rewards.mean(dim=-1)  # [K, *B]
        log_probs = record.log_probs.sum(dim=-1)  # [K, *B]

        # Group-relative advantage: normalize over K (dim 0)
        with torch.no_grad():
            group_mean = rewards_mean.mean(dim=0, keepdim=True)  # [1, *B]
            group_std = rewards_mean.std(dim=0, keepdim=True)    # [1, *B]
            advantage = (rewards_mean - group_mean) / (group_std + self.eps)

        pg_loss = -(advantage * log_probs).mean()

        if self.entropy_coeff > 0:
            entropy = record.action_entropy.mean()
            return pg_loss - self.entropy_coeff * entropy
        return pg_loss


# =========================================================================
# Backward-compatible alias
# =========================================================================

ReinforcementLoss = ReinforceLoss  # legacy name


# =========================================================================
# DecoderLoss — thin orchestrator
# =========================================================================

class DecoderLoss(nn.Module):
    """Thin orchestrator: plays the QEC game, routes to loss classes.

    Game flow (inside :meth:`play`, all under ``no_grad``)::

        simulator.reset_errors()
        simulator.update_errors()
        for each round:
            record syndromes
            dec.update_corrections(syndromes)
            record corrections
            simulator.apply_corrections(corrections)
            record logicals
        record final syndromes (T+1-th snapshot)

    The ``dec`` (decoder) role is filled by:

    * **Supervised learning** — ``teacher`` (Simulator, MWPM, or another
      Decoder).  The student only observes via teacher-forced ``forward()``.
    * **RL / self-play** — the student decoder itself.

    After play, ``forward`` reads from :class:`GameRecord` only.

    The ``objective`` dict maps loss names to weights::

        {"full_suffix": 1.0, "syndrome_density": 0.3}

    Known CE modes: ``"full_suffix"``, ``"corrections"``, ``"syndromes"``.
    Known RL modes: ``"syndrome_density"``, ``"logical_error_rate"``.

    The ``rl_algorithm`` selects which policy gradient class is used
    for RL-mode objectives:

    * ``"reinforce"`` — :class:`ReinforceLoss` (default)
    * ``"grpo"``      — :class:`GRPOLoss`

    Args:
        decoder: Student :class:`Decoder` to train.
        simulator: A :class:`Simulator` instance.
        teacher: Optional teacher (any :class:`Player`). When provided,
            the teacher plays the game and the student observes. The
            student is teacher-forced with the teacher's corrections.
        objective: Dict mapping loss names to weights.
        rl_algorithm: RL algorithm for policy gradient objectives.
        grpo_group_size: K samples per syndrome for GRPO.
        grpo_clip_range: Optional clipping for GRPO (0 = disabled).
        temperature: Sampling temperature for ``generate()`` during
            ``play()``. Default 1.0. Values > 1 increase exploration
            by flattening the softmax. Values < 1 sharpen it.
            Temperature is restored to 1.0 after play.
        entropy_coeff: Weight on entropy bonus in RL losses.
            Encourages diverse corrections by penalizing collapsed
            policies. Default 0.0 (off). Typical: 0.01–0.1.
    """

    def __init__(
        self,
        decoder: Decoder,
        simulator: Simulator,
        teacher: Optional[Decoder] = None,
        objective: Optional[Objective] = None,
        rl_algorithm: str = "reinforce",
        grpo_group_size: int = 4,
        grpo_clip_range: float = 0.0,
        temperature: float = 1.0,
        entropy_coeff: float = 0.0,
        pos_weight: float = 1.0,
        kl_penalty_coeff: float = 0.0,
    ):
        super().__init__()
        self.decoder = decoder
        self.simulator = simulator
        self.teacher = teacher
        self.rl_algorithm = rl_algorithm
        self.grpo_group_size = grpo_group_size
        self.grpo_clip_range = grpo_clip_range
        self.temperature = temperature
        self.entropy_coeff = entropy_coeff
        self.pos_weight = pos_weight
        self.kl_penalty_coeff = kl_penalty_coeff
        self._reference_state: Optional[dict] = None

        if rl_algorithm not in RL_ALGORITHMS:
            raise ValueError(
                f"Unknown rl_algorithm: {rl_algorithm!r}. "
                f"Known: {sorted(RL_ALGORITHMS)}"
            )

        # Build loss modules from objective dict
        self._loss_modules = nn.ModuleDict()
        self._weights: Dict[str, float] = {}

        if objective:
            self.set_objective(objective)

    def _make_rl_module(self, reward_fn: str) -> nn.Module:
        """Create the appropriate RL loss module based on ``rl_algorithm``."""
        if self.rl_algorithm == "reinforce":
            return ReinforceLoss(
                reward_fn=reward_fn,
                entropy_coeff=self.entropy_coeff,
            )
        elif self.rl_algorithm == "grpo":
            return GRPOLoss(
                reward_fn=reward_fn,
                clip_range=self.grpo_clip_range,
                entropy_coeff=self.entropy_coeff,
            )
        else:
            raise ValueError(f"Unknown rl_algorithm: {self.rl_algorithm!r}")

    def set_objective(self, objective: Objective) -> None:
        """Set or replace the full objective.

        Args:
            objective: Dict mapping loss names to weights. Unknown keys
                raise ``ValueError``.
        """
        for name in objective:
            if name not in ALL_MODES:
                raise ValueError(
                    f"Unknown objective: {name!r}. "
                    f"Known: {sorted(ALL_MODES)}"
                )

        # Create modules for any new keys
        for name in objective:
            if name not in self._loss_modules:
                if name in CE_MODES:
                    self._loss_modules[name] = SupervisedLoss(
                        mode=name, pos_weight=self.pos_weight,
                    )
                else:
                    self._loss_modules[name] = self._make_rl_module(name)

        self._weights = dict(objective)

    def snapshot_reference_policy(self) -> None:
        """Save a frozen copy of current decoder weights as KL reference."""
        import copy
        self._reference_state = copy.deepcopy(self.decoder.state_dict())

    def _kl_penalty(self, record: GameRecord) -> torch.Tensor:
        """Compute KL(current || reference) penalty for self-play RL stability.

        Uses the Bernoulli KL: sum_bits[ p*log(p/q) + (1-p)*log((1-p)/(1-q)) ]
        where p = current policy, q = reference policy.
        """
        if self._reference_state is None:
            return torch.tensor(0.0, device=record.syndromes.device)

        # Get current logits
        current_logits = record.error_logits[..., record.decoder.token_one]  # [*B, T, bits]
        p = torch.sigmoid(current_logits).clamp(1e-6, 1 - 1e-6)

        # Get reference logits (frozen)
        import copy
        ref_decoder = copy.deepcopy(self.decoder)
        ref_decoder.load_state_dict(self._reference_state)
        ref_decoder.eval()
        with torch.no_grad():
            ref_logits = ref_decoder.correction_logits(record.input_syndromes)
        q = torch.sigmoid(ref_logits).clamp(1e-6, 1 - 1e-6)

        # Bernoulli KL: p*log(p/q) + (1-p)*log((1-p)/(1-q))
        kl = p * (p.log() - q.log()) + (1 - p) * ((1 - p).log() - (1 - q).log())
        return kl.sum(dim=-1).mean()  # mean over batch and rounds

    @property
    def objective(self) -> Objective:
        """Current objective dict (read-only copy)."""
        return dict(self._weights)

    @property
    def has_rl(self) -> bool:
        """Whether the current objective includes any RL modes."""
        return any(name in RL_MODES for name, w in self._weights.items() if w > 0)

    # ----- Game play (no_grad, public interfaces only) ----------------------

    def play(
        self,
        num_rounds: int = 1,
        *,
        student_rollout: bool = False,
        temperature: Optional[float] = None,
    ) -> GameRecord:
        """Play the QEC game and return a trajectory record.

        All decoder and simulator interaction happens here under
        ``torch.no_grad()``. The returned :class:`GameRecord` lazily
        computes gradient-carrying tensors on first property access.

        When ``K > 1`` (GRPO with RL objectives), K independent games
        are played from the same initial errors and stacked along a new
        leading dimension ``[K, *B, …]``.

        Args:
            num_rounds: Number of correction rounds (T).
            student_rollout: If ``True``, always use ``self.decoder`` for
                ``update_corrections`` (ignores ``teacher``). Use this for
                metrics that should reflect the **student** (e.g. inline
                validation). Default ``False``: ``teacher`` drives the
                simulator when present (supervised training trajectory).
            temperature: If set, temporarily sets the student decoder's
                :attr:`~src.core.baseline_decoder.GraphDecoder.temperature`
                for this rollout (sharper Bernoulli = closer to MAP). If
                ``None``, uses :attr:`DecoderLoss.temperature`.

        Returns:
            A :class:`GameRecord` holding the full trajectory.
        """
        sim = self.simulator
        K = self.grpo_group_size if (self.rl_algorithm == "grpo" and self.has_rl) else 1
        if student_rollout:
            dec = self.decoder
        else:
            dec = self.teacher or self.decoder

        rollout_temp = float(self.temperature if temperature is None else temperature)
        old_temp = self._set_temperature(rollout_temp)
        try:
            with torch.no_grad():
                sim.reset_errors()
                sim.update_errors()

                if K == 1:
                    traj = self._play_once(sim, dec, num_rounds)
                    return GameRecord(decoder=self.decoder, **traj)

                # K > 1: replay from the same initial errors
                init_errors = sim.errors.clone()
                trajs = []
                for _ in range(K):
                    sim.reset_errors(init_errors.clone())
                    trajs.append(self._play_once(sim, dec, num_rounds))

                return GameRecord(
                    syndromes=torch.stack(
                        [t["syndromes"] for t in trajs], dim=0),
                    corrections=torch.stack(
                        [t["corrections"] for t in trajs], dim=0),
                    logicals=torch.stack(
                        [t["logicals"] for t in trajs], dim=0),
                    decoder=self.decoder,
                )
        finally:
            self._set_temperature(old_temp)

    def _set_temperature(self, temp: float) -> float:
        """Set sampling temperature, return previous value."""
        dec = self.decoder
        if hasattr(dec, "temperature"):
            old = float(dec.temperature)
            dec.temperature = float(temp)
            return old
        return 1.0

    def _play_once(self, sim: Simulator, dec, num_rounds: int) -> dict:
        """Run one game trajectory (caller holds ``no_grad``).

        Args:
            sim: Simulator with errors already initialized.
            dec: Any Player acting as the decoder.
            num_rounds: Number of correction rounds (T).

        Returns:
            Dict with ``syndromes`` ``[*B, T+1, …]``,
            ``corrections`` and ``logicals`` ``[*B, T, …]``.
        """
        all_syndromes = []   # will have T+1 entries
        all_corrections = []
        all_logicals = []

        for t in range(num_rounds):
            all_syndromes.append(sim.syndromes.clone())

            dec.update_corrections(sim.syndromes)
            all_corrections.append(dec.corrections.clone())

            sim.apply_corrections(dec.corrections)
            all_logicals.append(sim.logicals.clone())

            if t < num_rounds - 1:
                sim.update_errors()

        all_syndromes.append(sim.syndromes.clone())

        return {
            "syndromes":   torch.stack(all_syndromes, dim=-2),
            "corrections": torch.stack(all_corrections, dim=-2),
            "logicals":    torch.stack(all_logicals, dim=-2),
        }

    # ----- Forward (reads GameRecord only) ----------------------------------

    def forward(self, num_rounds: int = 1) -> tuple:
        """Play the game and compute combined loss.

        Uses :meth:`play` with default ``student_rollout=False``. When a
        ``teacher`` is set (e.g. MWPM), the teacher drives
        ``update_corrections``; ``eval/logical_error_rate`` is then the
        teacher's post-correction logical error rate, **not** the student's.
        :meth:`~src.task.training.Trainer.evaluate` uses
        ``play(..., student_rollout=True)`` to measure the student.

        Args:
            num_rounds: Number of correction rounds.

        Returns:
            ``(loss, metrics)`` — scalar loss and dict for logging.
        """
        record = self.play(num_rounds)
        metrics: dict = {}
        device = record.syndromes.device
        loss = torch.tensor(0.0, device=device)

        for name, w in self._weights.items():
            if w <= 0:
                continue
            module = self._loss_modules[name]
            term = module(record)
            loss = loss + w * term
            metrics[name] = term.item()

        # ── KL penalty (for self-play RL stability) ──
        if self.kl_penalty_coeff > 0 and self._reference_state is not None:
            kl = self._kl_penalty(record)
            loss = loss + self.kl_penalty_coeff * kl
            metrics["kl_penalty"] = kl.item()

        # ── Summary metrics (prefixed to avoid collision with loss names) ──
        metrics["total_loss"] = loss.item()
        metrics["eval/residual_syndrome_rate"] = (
            record.residual_syndromes.float().mean().item()
        )
        metrics["eval/logical_error_rate"] = (
            record.logicals.any(dim=-1).float().mean().item()
        )

        return loss, metrics


__all__ = [
    "GameRecord",
    "SupervisedLoss",
    "ReinforceLoss",
    "ReinforcementLoss",
    "GRPOLoss",
    "DecoderLoss",
    "CE_MODES",
    "RL_MODES",
    "ALL_MODES",
    "RL_ALGORITHMS",
    "Objective",
]
