"""Minimum Weight Perfect Matching (MWPM) decoder.

Uses PyMatching's Sparse Blossom (C++) backend.
Serves as a classical baseline for evaluating learned decoders.
"""

import torch
import pymatching
from ..base.dem import DetectorErrorModel
from .decoder import Decoder


class MWPMDecoder(Decoder):
    """MWPM decoder on detector Tanner graph.

    Each bit connects to at most two checks (graphlike error model).
    Checks are graph nodes, bits are edges, and edge weights reflect
    error likelihoods.
    """

    def __init__(self, dem: DetectorErrorModel, **kwargs):
        super().__init__(dem, **kwargs)
        tg = self.dem.detector_tanner_graph
        self.num_checks = tg.num_checks
        check_matrix = tg.check_matrix(format='scipy')
        weights = self.dem.get_weights(format='numpy')
        self.matching = pymatching.Matching.from_check_matrix(
            check_matrix, weights=weights,
        )

    def update_corrections(self, syndromes: torch.Tensor, **kwargs) -> None:
        """Decode syndromes into error corrections.

        Args:
            syndromes: ``[*B, num_checks]`` int {0, 1}.
        """
        if syndromes.shape[-1] != self.num_checks:
            raise ValueError(
                f"syndromes last dim ({syndromes.shape[-1]}) must match "
                f"num_checks ({self.num_checks})"
            )
        syndromes_flat = (
            syndromes.reshape(-1, syndromes.shape[-1])
            .round().clamp(0, 1).long().cpu().numpy()
        )
        corrections_flat = self.matching.decode_batch(syndromes_flat)
        self.corrections = (
            torch.from_numpy(corrections_flat)
            .to(syndromes.device, dtype=torch.long)
            .reshape(*syndromes.shape[:-1], -1)
        )


__all__ = ["MWPMDecoder"]
