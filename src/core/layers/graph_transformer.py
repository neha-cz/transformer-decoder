"""Graph transformer block and full graph transformer stack on a Tanner graph."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from ...base.tanner import TannerGraph
from .components import RMSNorm, SwiGLU
from .graph_attention import GraphAttention


class GraphTransformerBlock(nn.Module):
    """Pre-norm transformer block: GraphAttention + SwiGLU.

    Both graph modes follow the same three-phase pattern:

    1. **Prepare** — select which tensors play the "node" and "edge"
       roles for the attention call.
    2. **Attend** — ``h, ef = GraphAttention(norm(nodes), edge_index, edges)``.
    3. **Route back** — apply ``h`` and ``ef`` as residual updates to
       ``x`` and ``edge_attr`` respectively.

    After attention, a shared FFN + edge-norm is applied regardless of mode.

    **Tanner mode** (default)::

        nodes = x,          edges = edge_attr,  ei = edge_index
        x += h;  edge_attr += ef

    **Dual mode** — swaps node/edge roles::

        nodes = edge_attr,  edges = x[connecting_node_ids],  ei = dual_edge_index
        edge_attr += h;  x += scatter_mean(ef, connecting_node_ids)

    Args:
        d_model: Hidden dimension (same for nodes and edges).
        n_heads: Number of attention heads.
        d_ff: Feed-forward intermediate dimension.
        graph_mode: ``"tanner"`` or ``"dual"``.
        update_edge_attr: Whether to compute edge updates.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        graph_mode: str = "tanner",
        update_edge_attr: bool = True,
    ):
        super().__init__()
        self.graph_mode = graph_mode
        self.d_model = d_model

        self.norm1 = RMSNorm(d_model)
        self.attn = GraphAttention(d_model, n_heads, update_edge_attr)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

        self.update_edge_attr = update_edge_attr
        self.edge_norm = RMSNorm(d_model) if update_edge_attr else None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        dual_edge_index: Optional[torch.Tensor] = None,
        connecting_node_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run one transformer block.

        Args:
            x: Node features ``[*B, num_nodes, d_model]``.
            edge_index: ``[2, E_tanner]`` — required for tanner mode.
            edge_attr: ``[*B_e, E_tanner, d_model]`` — tanner edge
                features / dual-graph node features.
            dual_edge_index: ``[2, E_dual]`` — required for dual mode.
            connecting_node_ids: ``[E_dual]`` integer indices — the
                original tanner node mediating each dual edge.  Required
                for dual mode.

        Returns:
            ``(x_new, edge_attr_new)``
        """
        if self.graph_mode == "dual":
            nf = edge_attr
            ef = x[..., connecting_node_ids, :]
            ei = dual_edge_index
        else:
            nf = x
            ef = edge_attr
            ei = edge_index

        h, ef_delta = self.attn(self.norm1(nf), ei, ef)

        if self.graph_mode == "dual":
            edge_attr = edge_attr + h
            if ef_delta is not None:
                x = x + scatter(
                    ef_delta, connecting_node_ids,
                    dim=-2, dim_size=x.shape[-2], reduce="mean",
                )
        else:
            x = x + h
            if ef_delta is not None:
                edge_attr = edge_attr + ef_delta

        x = x + self.ffn(self.norm2(x))

        if self.update_edge_attr and edge_attr is not None:
            edge_attr = self.edge_norm(edge_attr)

        return x, edge_attr


__all__ = ["GraphTransformerBlock"]
