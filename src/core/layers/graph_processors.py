"""Graph processors for QEC decoding on Tanner graphs.

Two processor types share the same interface:

- **AdaptiveSharedGNN** — GCN-style message passing with shared weights.
- **AdaptiveSharedGraphTransformer** — multi-head graph attention with
  edge-aware keys/values, tanner/dual mode alternation, and SwiGLU FFN.

Both use adaptive depth: ``n_iterations = depth_multiplier * code_distance``,
set at ``bind()`` time so a single model scales to different code distances.

Contract:
    - ``bind(tanner_graph, code_distance=...)`` to attach graph topology
    - ``forward(h: [*B, V, d]) -> [*B, V, d]`` to process node features
    - ``_move_graph_tensors(fn)`` to support ``.to(device)``
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

from ...base.tanner import TannerGraph
from .components import RMSNorm


class AdaptiveSharedGNN(nn.Module):
    """GCN with shared weights and adaptive depth.

    A small set of ``n_unique`` layers is iterated
    ``depth_multiplier * code_distance`` times (set at ``bind()``).
    Each layer performs:

        h_i' = h_i + GELU(mean(W_msg @ h_j for j in N(i)) + W_self @ h_i)
        h_i' = h_i' + FFN(RMSNorm(h_i'))

    This mimics classical belief propagation: the same update rule
    applied iteratively until convergence.

    Args:
        d_model: Feature dimension.
        n_unique: Number of distinct layer weight sets.
        n_iterations: Fallback iteration count (used when code_distance
            is not provided at bind time).
        depth_multiplier: Iterations = depth_multiplier * code_distance.
        d_ff: Feed-forward intermediate dimension (default 2 * d_model).
    """

    def __init__(
        self,
        d_model: int,
        n_unique: int = 2,
        n_iterations: int = 16,
        depth_multiplier: int = 2,
        d_ff: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_unique = n_unique
        self.n_iterations = n_iterations
        self.depth_multiplier = depth_multiplier
        d_ff = d_ff or d_model * 2

        self.layers = nn.ModuleList()
        for _ in range(n_unique):
            self.layers.append(nn.ModuleDict({
                'norm': RMSNorm(d_model),
                'msg': nn.Linear(d_model, d_model, bias=False),
                'self_': nn.Linear(d_model, d_model, bias=False),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_ff, bias=False),
                    nn.GELU(),
                    nn.Linear(d_ff, d_model, bias=False),
                ),
                'ffn_norm': RMSNorm(d_model),
            }))
        self.final_norm = RMSNorm(d_model)
        self.edge_index: Optional[torch.Tensor] = None

    def bind(
        self,
        tanner_graph: TannerGraph,
        code_distance: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Attach graph topology and set adaptive iteration count."""
        device = next(self.parameters()).device
        self.edge_index = tanner_graph.edge_index.to(device)
        if code_distance is not None:
            self.n_iterations = self.depth_multiplier * code_distance

    def _move_graph_tensors(self, fn) -> None:
        if self.edge_index is not None:
            self.edge_index = fn(self.edge_index)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.edge_index is None:
            raise RuntimeError("AdaptiveSharedGNN: call .bind(tanner_graph) first.")

        ei = self.edge_index
        src_idx, tgt_idx = ei[0], ei[1]
        V = h.shape[-2]

        for i in range(self.n_iterations):
            layer = self.layers[i % self.n_unique]
            h_normed = layer['norm'](h)

            msg = layer['msg'](h_normed)
            msg_src = msg[..., src_idx, :]
            agg = scatter(msg_src, tgt_idx, dim=-2, dim_size=V, reduce='mean')

            h = h + F.gelu(agg + layer['self_'](h_normed))
            h = h + layer['ffn'](layer['ffn_norm'](h))

        return self.final_norm(h)


class AdaptiveSharedGraphTransformer(nn.Module):
    """GraphTransformer with shared weights and adaptive depth.

    Creates ``n_unique`` GraphTransformerBlocks (with tanner/dual modes) and
    iterates them ``depth_multiplier * code_distance`` times, cycling through
    the block list. Tanner/dual alternation avoids message backflow.
    ``n_iterations`` is set at ``bind()`` time from code_distance.

    Args:
        d_model: Feature dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward intermediate dimension.
        graph_modes: Per-block ``"tanner"`` or ``"dual"``, or one string for all.
        n_unique: Number of distinct shared transformer blocks.
        n_iterations: Fallback iteration count.
        depth_multiplier: Iterations = depth_multiplier * code_distance.
        update_edge_attr: Whether blocks update edge features.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        graph_modes: Union[str, List[str]] = "tanner",
        n_unique: int = 2,
        n_iterations: int = 16,
        depth_multiplier: int = 2,
        update_edge_attr: bool = True,
        **kwargs,
    ):
        super().__init__()
        from .graph_transformer import GraphTransformerBlock

        self.d_model = d_model
        self.n_unique = n_unique
        self.n_iterations = n_iterations
        self.depth_multiplier = depth_multiplier

        # Graph topology — plain attributes, NOT in state_dict
        self.edge_index: Optional[torch.Tensor] = None
        self.dual_edge_index: Optional[torch.Tensor] = None
        self.connecting_node_ids: Optional[torch.Tensor] = None

        if isinstance(graph_modes, str):
            modes = [graph_modes] * n_unique
        else:
            modes = list(graph_modes)
        if len(modes) != n_unique:
            modes = [modes[i % len(modes)] for i in range(n_unique)]

        self.blocks = nn.ModuleList([
            GraphTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                graph_mode=mode,
                update_edge_attr=update_edge_attr,
            )
            for mode in modes
        ])
        self.norm = RMSNorm(d_model)

    def bind(
        self,
        tanner_graph: TannerGraph,
        code_distance: Optional[int] = None,
        **kwargs,
    ) -> None:
        device = next(self.parameters()).device
        self.edge_index = tanner_graph.edge_index.to(device)
        self.dual_edge_index = tanner_graph.dual_edge_index.to(device)
        self.connecting_node_ids = tanner_graph.connecting_node_ids.to(device)
        if code_distance is not None:
            self.n_iterations = self.depth_multiplier * code_distance

    def _move_graph_tensors(self, fn) -> None:
        if self.edge_index is not None:
            self.edge_index = fn(self.edge_index)
        if self.dual_edge_index is not None:
            self.dual_edge_index = fn(self.dual_edge_index)
        if self.connecting_node_ids is not None:
            self.connecting_node_ids = fn(self.connecting_node_ids)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.edge_index is None:
            raise RuntimeError(
                "AdaptiveSharedGraphTransformer: call .bind(tanner_graph) first."
            )
        edge_attr = h.new_zeros(self.edge_index.shape[1], self.d_model)
        for i in range(self.n_iterations):
            block = self.blocks[i % self.n_unique]
            h, edge_attr = block(
                h, self.edge_index, edge_attr,
                self.dual_edge_index, self.connecting_node_ids,
            )
        return self.norm(h)


def build_graph_processor(
    processor_type: str = "transformer",
    d_model: int = 96,
    d_ff: int = 192,
    n_unique: int = 2,
    depth_multiplier: int = 2,
    n_iterations: int = 16,
    n_heads: int = 4,
    graph_modes: Union[str, List[str]] = "tanner",
    update_edge_attr: bool = True,
    **kwargs,
) -> nn.Module:
    """Factory to build a graph processor by type.

    Args:
        processor_type: ``"gnn"`` for AdaptiveSharedGNN or
            ``"transformer"`` for AdaptiveSharedGraphTransformer.
        d_model: Feature dimension.
        d_ff: Feed-forward hidden size.
        n_unique: Number of distinct shared layers/blocks.
        depth_multiplier: Iterations = depth_multiplier * code_distance.
        n_iterations: Fallback iteration count.
        n_heads: Attention heads (transformer only).
        graph_modes: Per-block tanner/dual mode (transformer only).
        update_edge_attr: Whether to update edge features (transformer only).
    """
    if processor_type == "gnn":
        return AdaptiveSharedGNN(
            d_model=d_model,
            n_unique=n_unique,
            n_iterations=n_iterations,
            depth_multiplier=depth_multiplier,
            d_ff=d_ff,
        )
    elif processor_type == "transformer":
        return AdaptiveSharedGraphTransformer(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            graph_modes=graph_modes,
            n_unique=n_unique,
            n_iterations=n_iterations,
            depth_multiplier=depth_multiplier,
            update_edge_attr=update_edge_attr,
        )
    else:
        raise ValueError(
            f"Unknown processor_type: {processor_type!r}. "
            f"Supported: 'gnn', 'transformer'."
        )


__all__ = [
    "AdaptiveSharedGNN",
    "AdaptiveSharedGraphTransformer",
    "build_graph_processor",
]
