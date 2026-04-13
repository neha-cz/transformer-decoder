"""Graph decoder for quantum error correction (GNN or graph transformer).

Syndromes are embedded on **detector (check) nodes**; error (bit) nodes
start with node-identity features. After the graph processor (GNN or
graph transformer), a linear head reads **error-node** hidden states
and outputs one logit per bit (probability of applying a ``1`` correction).

System prompt conditioning injects predetermined calibration information
(code type, distance, error rate) as a shared base feature for all nodes.
This is locality-compliant: the information comes from device calibration,
not from runtime global observables.

The ``processor_type`` config field selects the graph processor:

- ``"gnn"`` — AdaptiveSharedGNN (GCN-style message passing, tanner graph only)
- ``"transformer"`` — AdaptiveSharedGraphTransformer (multi-head graph attention
  with edge features, tanner/dual mode alternation, SwiGLU FFN)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base.dem import DetectorErrorModel
from ..util.tokenizer import load_tokenizer, build_vocab
from .decoder import Decoder
from .player import Config
from .layers.graph_processors import build_graph_processor


class DecoderConfig(Config):
    """Configuration for :class:`GraphDecoder`.

    Args:
        d_model: Feature dimension for nodes.
        n_unique: Number of distinct shared layers/blocks.
        d_ff: Feed-forward intermediate dimension.
        depth_multiplier: Iterations = depth_multiplier * code_distance.
        processor_type: ``"gnn"`` or ``"transformer"``.
        n_heads: Attention heads (transformer only).
        graph_modes: Per-block ``"tanner"``/``"dual"`` (transformer only).
        update_edge_attr: Whether to update edge features (transformer only).
    """

    def __init__(
        self,
        d_model: int = 96,
        n_unique: int = 2,
        d_ff: int = 192,
        depth_multiplier: int = 2,
        processor_type: str = "transformer",
        n_heads: int = 4,
        graph_modes: Union[str, List[str]] = "tanner",
        update_edge_attr: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_unique = n_unique
        self.d_ff = d_ff
        self.depth_multiplier = depth_multiplier
        self.processor_type = processor_type
        self.n_heads = n_heads
        self.graph_modes = graph_modes
        self.update_edge_attr = update_edge_attr

    @property
    def vocab_size(self) -> int:
        """Two-class head (no correction / correction) for CE compatibility."""
        return 2

    @property
    def max_gen_tokens(self) -> int:
        return 1


class GraphDecoder(Decoder, nn.Module):
    """Adaptive shared-weight GNN decoder.

    Pipeline::

        syndromes → system prompt + node embeddings + syndrome injection
            → AdaptiveSharedGNN (2 * code_distance iterations)
            → linear bit head → per-bit Bernoulli logits

    Attributes:
        is_binary_correction_decoder: Always ``True``; losses use Bernoulli semantics.
        temperature: Sampling temperature for :meth:`generate`.
    """

    is_binary_correction_decoder: bool = True

    def __init__(
        self,
        dem: DetectorErrorModel,
        config: Optional[DecoderConfig] = None,
        **kwargs,
    ):
        if config is None:
            config = DecoderConfig(**kwargs)
        Decoder.__init__(self, dem, config=config)
        nn.Module.__init__(self)

        cfg = self.config
        self.temperature: float = 1.0

        self.register_buffer("token_zero", torch.tensor(0, dtype=torch.long))
        self.register_buffer("token_one", torch.tensor(1, dtype=torch.long))

        d = cfg.d_model

        # ----- Learned parameters (graph-agnostic) ----------------------------
        vocab = build_vocab()
        pad_id = vocab["<pad>"]
        self.token_embed = nn.Embedding(len(vocab), d, padding_idx=pad_id)
        self.syndrome_embed = nn.Embedding(2, d)  # 0 → no-trigger, 1 → trigger

        self.gnn = build_graph_processor(
            processor_type=cfg.processor_type,
            d_model=d,
            d_ff=cfg.d_ff,
            n_unique=cfg.n_unique,
            depth_multiplier=cfg.depth_multiplier,
            n_heads=cfg.n_heads,
            graph_modes=cfg.graph_modes,
            update_edge_attr=cfg.update_edge_attr,
        )

        self.bit_head = nn.Linear(d, 1, bias=True)
        self._init_weights()

        # ----- Bind graph-specific data (not in state_dict) -------------------
        self.bind(dem)

    def _get_device(self) -> torch.device:
        return self.token_embed.weight.device

    def _apply(self, fn):
        """Move non-state-dict tensors alongside parameters."""
        super()._apply(fn)
        if hasattr(self, "prompt_token_ids") and self.prompt_token_ids is not None:
            self.prompt_token_ids = fn(self.prompt_token_ids)
        if hasattr(self, "system_prompt_ids") and self.system_prompt_ids is not None:
            self.system_prompt_ids = fn(self.system_prompt_ids)
        self.gnn._move_graph_tensors(fn)
        return self

    # ----- Binding ------------------------------------------------------------

    def bind(self, dem: DetectorErrorModel) -> None:
        """Rebind to a new DEM (different code/distance). Learned params unchanged."""
        self.num_bits = dem.num_errors
        self.num_checks = dem.num_detectors
        self.prompt_token_ids = self._tokenize_prompts(dem).to(self._get_device())
        self.system_prompt_ids = self._tokenize_system_prompt(dem).to(self._get_device())
        code_distance = dem.graph.get('code_distance', None)
        self.gnn.bind(dem.detector_tanner_graph, code_distance=code_distance)

    def _tokenize_prompts(self, dem: DetectorErrorModel) -> torch.Tensor:
        """Tokenize DEM node prompts into padded token IDs ``[V, L]``."""
        tokenizer = load_tokenizer()
        error_prompts = dem.generate_node_prompts(
            "error", encode_weights=True, add_ans=False,
        )
        detector_prompts = dem.generate_node_prompts(
            "detector", add_ans=False,
        )
        all_prompts = error_prompts + detector_prompts
        encoded = tokenizer(
            all_prompts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=16,
            return_tensors="pt",
        )
        return encoded["input_ids"]  # [V, L]

    def _tokenize_system_prompt(self, dem: DetectorErrorModel) -> torch.Tensor:
        """Tokenize DEM system prompt into token IDs ``[1, L_sys]``.

        Encodes predetermined calibration info (code type, distance, error rate).
        Satisfies locality: these are device parameters, not runtime observables.
        """
        tokenizer = load_tokenizer()
        sys_prompt = dem.generate_system_prompt()
        encoded = tokenizer(
            [sys_prompt],
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors="pt",
        )
        return encoded["input_ids"]  # [1, L_sys]

    # ----- Initialization -----------------------------------------------------

    def _init_weights(self) -> None:
        cfg = self.config
        if hasattr(self.gnn, 'blocks'):
            n_layers = len(self.gnn.blocks)
        else:
            n_layers = cfg.n_unique
        residual_scale = 1.0 / math.sqrt(2.0 * max(n_layers, 1))

        for name, param in self.named_parameters():
            if param.dim() < 2:
                continue
            if "token_embed" in name or "syndrome_embed" in name:
                nn.init.normal_(param, mean=0.0, std=1.0 / math.sqrt(cfg.d_model))
            elif name == "bit_head.weight":
                nn.init.xavier_uniform_(param)
                param.data.mul_(residual_scale)
            elif param.dim() >= 2:
                nn.init.xavier_uniform_(param)

    # ----- Feature construction -----------------------------------------------

    def _node_features(self, syndromes: torch.Tensor) -> torch.Tensor:
        """Build ``[*, V, d]`` node features from prompts + syndromes."""
        *lead, nd = syndromes.shape
        if nd != self.num_checks:
            raise ValueError(
                f"syndromes last dim {nd} != num_checks {self.num_checks}"
            )
        d = self.config.d_model
        s = syndromes.reshape(-1, nd)
        n = s.shape[0]
        v = self.num_bits + self.num_checks

        # 1. System prompt → shared base feature for all nodes
        sys_emb = self.token_embed(self.system_prompt_ids)          # [1, L_sys, d]
        sys_pad = (
            self.system_prompt_ids != self.token_embed.padding_idx
        ).unsqueeze(-1).float()                                     # [1, L_sys, 1]
        sys_feat = (sys_emb * sys_pad).sum(dim=1) / sys_pad.sum(dim=1).clamp(min=1)  # [1, d]

        # 2. Node prompts → per-node identity features
        tok_emb = self.token_embed(self.prompt_token_ids)           # [V, L, d]
        pad_mask = (
            self.prompt_token_ids != self.token_embed.padding_idx
        ).unsqueeze(-1).float()                                     # [V, L, 1]
        node_emb = (tok_emb * pad_mask).sum(dim=1) / pad_mask.sum(dim=1).clamp(min=1)  # [V, d]

        # 3. Combine: system base + node identity → broadcast to batch
        h = (node_emb + sys_feat).unsqueeze(0).expand(n, -1, -1).clone()  # [n, V, d]

        # 4. Add syndrome embedding on check nodes
        inj = self.syndrome_embed(s.long())                         # [n, nc, d]
        h[:, self.num_bits:, :] = h[:, self.num_bits:, :] + inj

        return h.view(*lead, v, d)

    # ----- Forward / inference ------------------------------------------------

    def correction_logits(self, syndromes: torch.Tensor) -> torch.Tensor:
        """Raw logits for P(correction bit = 1).

        Args:
            syndromes: ``[..., num_checks]`` in ``{0, 1}``.

        Returns:
            ``[..., num_bits]`` logits.
        """
        h = self._node_features(syndromes)
        h = self.gnn(h)
        bits = h[..., : self.num_bits, :]
        return self.bit_head(bits).squeeze(-1)

    def log_prob(self, syndromes: torch.Tensor, corrections: torch.Tensor) -> torch.Tensor:
        """``log p(corrections | syndromes)`` with independent Bernoulli bits."""
        lg = self.correction_logits(syndromes)
        y = corrections.float()
        if y.shape != lg.shape:
            if y.shape[-1] != lg.shape[-1]:
                raise ValueError(
                    f"corrections last dim {y.shape[-1]} != num_bits {lg.shape[-1]}"
                )
            y = y.expand_as(lg)
        lp = F.logsigmoid(lg) * y + F.logsigmoid(-lg) * (1.0 - y)
        return lp.sum(dim=-1)

    def forward(
        self,
        syndromes: torch.Tensor,
        corrections: torch.Tensor,
        residual_syndromes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Two-class logits for CE training."""
        lg = self.correction_logits(syndromes)
        *lead, nb = lg.shape
        nc = self.num_checks
        v = nb + nc

        z0 = torch.zeros_like(lg)
        two = torch.stack([z0, lg], dim=-1)
        pad = torch.zeros(*lead, nc, 2, device=lg.device, dtype=lg.dtype)
        full = torch.cat([two, pad], dim=-2)
        suffix_logits = full.unsqueeze(-2)

        targ = torch.zeros(*lead, v, 1, dtype=torch.long, device=lg.device)
        targ[..., :nb, 0] = corrections.long()
        syn_tgt = residual_syndromes if residual_syndromes is not None else syndromes
        targ[..., nb:, 0] = syn_tgt.long()
        return suffix_logits, targ

    @torch.no_grad()
    def generate(self, syndromes: torch.Tensor) -> torch.Tensor:
        """Bernoulli samples from correction logits."""
        lg = self.correction_logits(syndromes)
        lg = lg / max(self.temperature, 1e-8)
        p = torch.sigmoid(lg)
        return torch.bernoulli(p).long()

    def update_corrections(self, syndromes: torch.Tensor, **kwargs) -> None:
        self.corrections = self.generate(syndromes)

    @torch.no_grad()
    def sample_suffix_ids(self, syndromes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compatibility: fake ``[V, G]`` token ids (0/1 on error nodes)."""
        corr = self.generate(syndromes)
        *lead, nb = corr.shape
        v = nb + self.num_checks
        out = torch.zeros(*lead, v, 1, dtype=torch.long, device=corr.device)
        out[..., :nb, 0] = corr
        lp = self.log_prob(syndromes, corr)
        return out, lp


__all__ = ["DecoderConfig", "GraphDecoder"]
