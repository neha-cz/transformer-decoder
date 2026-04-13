"""Edge-aware graph attention (TransformerConv + GatedGCN edge update).

Combines TransformerConv-style edge-in-attention-and-value (Shi et al., 2021)
with GatedGCN-style asymmetric residual edge update (Bresson & Laurent, 2017).

Uses PyG ``MessagePassing`` with ``node_dim=-2`` so that batch dimensions
``[*B, N, d_model]`` are handled natively — no manual flattening needed.
The module is **graph-mode-agnostic**: the caller decides the topology by
choosing what to pass as ``(x, edge_index, edge_attr)``.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class GraphAttention(MessagePassing):
    """Multi-head sparse attention on a graph with optional edge features.

    For each directed edge ``(j -> i)`` in ``edge_index``:

    * **Attention score** (edge-aware)::

        score_ij = Q[i]^T (K[j] + E_ij) / sqrt(d_head)

    * **Message** (edge-aware)::

        msg_ij = alpha_ij * (V[j] + E_ij)

    * **Edge update** (source/target asymmetric delta), via
      :meth:`torch_geometric.nn.MessagePassing.edge_updater` / :meth:`edge_update`::

        delta_e_ij = W_src(x[j]) + W_tgt(x[i]) + W_self(e_ij)

    Both node and edge outputs are **deltas** — the caller is responsible
    for adding residual connections.

    Args:
        d_model: Feature dimension for both nodes and edges.
        n_heads: Number of attention heads.
        update_edge_attr: Whether to compute residual edge updates.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        update_edge_attr: bool = True,
    ):
        super().__init__(aggr="add", flow="source_to_target", node_dim=-2)
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model
        self.update_edge_attr = update_edge_attr
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Node projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # Edge projection (d_edge == d_model)
        self.w_e = nn.Linear(d_model, d_model, bias=False)

        # Edge update projections (asymmetric: source != target)
        if update_edge_attr:
            self.w_src = nn.Linear(d_model, d_model, bias=False)
            self.w_tgt = nn.Linear(d_model, d_model, bias=False)
            self.w_self = nn.Linear(d_model, d_model, bias=False)
        else:
            self.w_src = self.w_tgt = self.w_self = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run sparse multi-head attention on the given graph.

        Args:
            x: Node features ``[*B, N, d_model]``.
            edge_index: ``[2, E]`` COO directed edges.
            edge_attr: Optional edge features ``[*B_e, E, d_model]``.

        Returns:
            ``(x_new, edge_attr_new)`` — updated node and edge features.
        """
        q = self.w_q(x)  # [*B, N, d_model]
        k = self.w_k(x)
        v = self.w_v(x)
        e = self.w_e(edge_attr) if edge_attr is not None else None  # [*B_e, E, d_model]

        out = self.propagate(edge_index, q=q, k=k, v=v, e=e)  # [*B, N, d_model]

        if self.update_edge_attr and edge_attr is not None:
            edge_attr_delta = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)
        else:
            edge_attr_delta = None
        return out, edge_attr_delta

    def message(
        self,
        q_i: torch.Tensor,
        k_j: torch.Tensor,
        v_j: torch.Tensor,
        e: Optional[torch.Tensor],
        index: torch.Tensor,
        size_i: Optional[int],
    ) -> torch.Tensor:
        """Compute attention-weighted messages per edge."""
        H, D = self.n_heads, self.head_dim

        # Reshape to multi-head: [*B, E, d_model] -> [*B, E, H, D]
        q_h = q_i.unflatten(-1, (H, D))
        k_h = k_j.unflatten(-1, (H, D))
        v_h = v_j.unflatten(-1, (H, D))

        # Edge-aware keys and values
        if e is not None:
            e_h = e.unflatten(-1, (H, D))
            k_h = k_h + e_h
            v_h = v_h + e_h

        # Attention scores: [*B, E, H]
        scores = torch.linalg.vecdot(q_h, k_h, dim=-1) * self.scale

        # Sparse softmax per target node, per head
        alpha = softmax(src=scores, index=index, num_nodes=size_i, dim=-2)

        # Weighted values: [*B, E, H, D] -> [*B, E, d_model]
        out = alpha.unsqueeze(-1) * v_h
        return out.flatten(-2)

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        """Output projection after message aggregation (per target node)."""
        return self.w_o(aggr_out)

    def edge_update(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Asymmetric edge update delta (GatedGCN-style), per directed edge."""
        if self.update_edge_attr:
            return self.w_src(x_j) + self.w_tgt(x_i) + self.w_self(edge_attr)
        else:
            return torch.zeros_like(edge_attr)


__all__ = ["GraphAttention"]
