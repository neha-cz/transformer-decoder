"""Decoder base class (non-stateful) for quantum error correction."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from ..base.dem import DetectorErrorModel
from .player import Player, Config


class Decoder(Player):
    """Decoder (base class) for a :class:`~src.base.dem.DetectorErrorModel`.

    Provides the common interface for feed-forward decoding (syndromes -> corrections
    in one shot). Stateful/iterative variants were previously modeled with
    `StatefulDecoder`, but those have been removed as part of the decoder
    refactor.
    """

    def __init__(self, dem: DetectorErrorModel, **kwargs):
        super().__init__(dem, **kwargs)

    def draw_on(self, draw_kwargs: Dict, **style) -> None:
        """Annotate ``draw_kwargs`` with decoder state visualization.

        Called by ``DetectorErrorModel.draw()`` which builds ``draw_kwargs``
        (the mutable ``nx.draw_networkx`` parameter dict) and passes merged
        styling via ``**style``.
        """
        if self.corrections is None:
            return
        try:
            batch_index = style.get("batch_index", 0)
            colors = style["colors"]
            sizes = style["sizes"]
            node_size = draw_kwargs.get("node_size", sizes["node"])
            nodes = list(self.dem.nodes())
            if not isinstance(node_size, list):
                node_size = [node_size] * len(nodes)
            correction = self.corrections[batch_index]
            edgecolors = [(1.0, 1.0, 1.0)] * len(nodes)
            linewidths = [0.0] * len(nodes)

            node_to_idx = {n: i for i, n in enumerate(nodes)}
            check_boundary = sizes.get("check_boundary", 4.0)

            correction_color = colors["correction_color"]
            if isinstance(correction_color, str):
                from matplotlib.colors import to_rgb

                correction_color = to_rgb(correction_color)
            white = (1.0, 1.0, 1.0)
            error_nodes = self.dem.error_nodes
            for node_idx in range(correction.shape[-1]):
                node = error_nodes[node_idx]
                idx = node_to_idx[node]
                c = float(correction[node_idx])
                c = np.clip(c, 0.0, 1.0)
                edgecolors[idx] = tuple(
                    white[i] * (1 - c) + correction_color[i] * c for i in range(3)
                )
                linewidths[idx] = check_boundary * c

            draw_kwargs["edgecolors"] = edgecolors
            draw_kwargs["linewidths"] = linewidths
            draw_kwargs["node_size"] = node_size
        except (AttributeError, IndexError, RuntimeError, KeyError, ValueError):
            pass

    def update_corrections(self, syndromes: torch.Tensor, **kwargs) -> None:
        """Predict corrections from syndromes and store in ``self.corrections``."""
        raise NotImplementedError("update_corrections not implemented in base class")


__all__ = ["Decoder"]

