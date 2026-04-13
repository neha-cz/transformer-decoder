"""Simulator for quantum error correction."""

import builtins
import torch
from typing import Optional
from ..base.dem import DetectorErrorModel
from .player import Player

class Simulator(Player):
    """Simulator for a DetectorErrorModel: error evolution, syndrome/logical computation, and correction application.

    The Simulator holds the **internal error state** (hidden); only syndromes and
    logical observables are exposed, matching the QEC principle that errors
    are not directly observable. It uses the DEM's Markovian error model to
    update errors and the detector/logical Tanner graphs to compute syndromes
    and logicals.

    **Key attributes**
      - `dem`: the `DetectorErrorModel` being simulated (from `Player`).
      - `error_model_data`: pre-computed error weights from `dem.get_error_model_data()`
        (used by `update_errors`).
      - `num_errors`: number of error nodes (from `dem.num_errors`).
      - `errors`: internal error vector, shape `[*batch_shape, num_errors]`, dtype
        ``torch.long`` in ``{0, 1}``; not exposed; updated by `reset_errors`,
        `update_errors`, and `apply_corrections`.
      - `syndromes`: property; detector parity ``H @ errors`` (mod 2), shape
        ``[*batch_shape, num_detectors]``, dtype ``torch.long`` ``{0, 1}``.
      - `logicals`: property; logical parity ``K @ errors`` (mod 2), shape
        ``[*batch_shape, num_logicals]``, dtype ``torch.long`` ``{0, 1}``.
      - `corrections`: last applied correction vector (set by `apply_corrections`).
      - `batch_shape`, `device`: from `Player`.

    **Key methods (with brief examples)**

      - `reset_errors()`: set internal `errors` to all zeros. Call after init
        or to start a new run. 
        Example: `sim.reset_errors()`.

      - `update_errors()`: update `errors` in-place using the DEM's Markovian
        error model (LLR from `error_model_data`, sigmoid to probs, Bernoulli
        sample). 
        Example: `sim.update_errors()`.

      - `syndromes`, `logicals`: read-only properties; compute from current
        `errors` via detector/logical Tanner graphs. 

      - `apply_corrections(corrections)`: set `errors = errors XOR corrections`
        (mod 2). `corrections` must have shape `[*batch_shape, num_errors]`.
        Example: `sim.apply_corrections(corrections)`.

      - `update_corrections(syndromes, range=None)`: build corrections from
        hidden errors (teacher signal). ``syndromes`` satisfies the unified
        Player interface but is not used internally. If ``range`` is ``None``,
        copies all ``errors`` into ``corrections``. If ``range >= 1``, masks to
        bits within that many check-to-bit expansion steps of active syndromes.
        Example: `sim.update_corrections(sim.syndromes)` or
        `sim.update_corrections(sim.syndromes, range=2)`.
    """
    
    def __init__(self, dem: DetectorErrorModel, **kwargs):
        """Initialize Simulator with a DetectorErrorModel.

        Args:
            dem: DetectorErrorModel instance to simulate
            **kwargs: Other keyword arguments (e.g. config) passed to Player.
        """
        super().__init__(dem, **kwargs)

        # Pre-compute error model data (TannerGraphs accessed via dem cached properties)
        self.error_model_data = dem.get_error_model_data()

        # Get node counts
        self.num_errors = dem.num_errors

        # Internal error state (hidden from external observers)
        self.reset_errors()
    
    def update_error_model(self, nodes=None, **kwargs) -> None:
        """Update error model on the DEM and refresh simulator tensor cache.

        Delegates to :meth:`DetectorErrorModel.update_error_model`, then
        recompiles ``error_model_data`` for :meth:`update_errors`.
        """
        self.dem.update_error_model(nodes=nodes, **kwargs)
        self.error_model_data = self.dem.get_error_model_data()

    def reset_errors(self, errors: Optional[torch.Tensor] = None) -> None:
        """Reset or override the internal error vector.

        Args:
            errors: If ``None``, resets to all zeros with shape
                ``[*batch_shape, num_errors]``.  If provided, directly
                sets ``self.errors`` to the given tensor.
        """
        if errors is not None:
            self.errors = errors
        else:
            self.errors = torch.zeros(
                (*self.batch_shape, self.num_errors),
                dtype=torch.long,
                device=self.device,
            )
    
    def update_errors(self) -> None:
        """Update internal error vector using the universal Markovian error model.
        
        Computes log likelihood ratio (LLR) for each error bit based on the current error vector,
        then samples new errors from the resulting probabilities.
        Updates self.errors in-place.
        """
        if self.errors is None:
            self.reset_errors()
        
        # Initialize LLRs for all error bits
        llrs = torch.zeros(self.errors.shape, dtype=torch.float, device=self.device)
        
        # Process each interaction order (1st-order bias, 2nd-order pairs, 3rd-order triples, etc.)
        for order, (indices_tensor, weights_tensor) in self.error_model_data.items():
            # indices_tensor shape: [num_terms, order] - each row is [error_idx, dep_idx1, dep_idx2, ...]
            # weights_tensor shape: [num_terms] - weight coefficients for each interaction term
            
            # Compute context: XOR parity of dependent error bits (mod 2)
            dep_slice = indices_tensor[:, 1:]
            if dep_slice.shape[1] == 0:
                context = torch.zeros(
                    (*self.batch_shape, indices_tensor.shape[0]),
                    dtype=torch.float32,
                    device=self.device
                )
            else:
                # Hard XOR parity of dependent error bits (integer mod 2).
                context = self.errors[..., indices_tensor[:, 1:]].sum(dim=-1) % 2
            
            # Transform context to interaction sign: (1 - 2*context) maps {0->+1, 1->-1}
            # if context=0, weight contributes positively
            # if context=1, weight contributes negatively (flips the sign)
            weights = (1.0 - 2.0 * context.to(weights_tensor.dtype)) * weights_tensor
            
            # Get target error indices (first column of indices_tensor)
            error_idx = indices_tensor[:, 0].view(
                *(1 for _ in self.batch_shape),
                -1
            ).expand(*self.batch_shape, -1)  # [..., num_terms]
            
            # Accumulate weighted contributions to LLRs
            llrs.scatter_add_(-1, error_idx, weights)
        
        # Convert LLRs to probabilities (of error) using sigmoid
        probs = torch.sigmoid(-llrs)
        
        # Sample new errors from Bernoulli distribution and update internal state
        self.errors = torch.bernoulli(probs).long()
    
    @property
    def syndromes(self) -> torch.Tensor:
        """Compute syndrome (detector values) from internal error vector.
        
        Syndrome is computed as: syndrome = H @ error (mod 2)
        where H is the detector-error parity check matrix.
        
        Returns:
            Syndrome tensor with shape ``[..., num_detectors]``, dtype ``long`` ``{0,1}``.
        """
        if self.errors is None:
            self.reset_errors()
        
        return self.dem.detector_tanner_graph.bit_to_check(self.errors)
    
    @property
    def logicals(self) -> torch.Tensor:
        """Compute logical observable values from internal error vector.
        
        Logical is computed as: logical = K @ error (mod 2)
        where K is the logical-error observable matrix.
        
        Returns:
            Logical tensor with shape ``[..., num_logicals]``, dtype ``long`` ``{0,1}``.
        """
        if self.errors is None:
            self.reset_errors()
        
        return self.dem.logical_tanner_graph.bit_to_check(self.errors)
    
    def apply_corrections(self, corrections: torch.Tensor, noisy: bool = False, **kwargs) -> None:
        """Apply error correction to internal error vector.

        Corrections are applied as: error_corrected = error XOR corrections.
        This is the first entry point where the simulator interacts with corrections.
        Corrections must have the same shape as internal errors ([*self.batch_shape, num_errors]).
        """
        if self.errors is None:
            self.reset_errors()
        if corrections.shape != self.errors.shape:
            raise ValueError(
                f"corrections shape {tuple(corrections.shape)} must match errors shape {tuple(self.errors.shape)}"
            )
        self.errors = self.errors ^ corrections.detach().long()

        # If noisy, update the error state using the Markovian error model
        if noisy:
            self.update_errors()

    def update_corrections(self, syndromes: torch.Tensor, range: Optional[int] = None, **kwargs) -> None:
        """Build corrections from hidden errors (teacher signal).

        Satisfies the unified ``Player.update_corrections`` interface so the
        simulator can act as a player (teacher) in the game loop.

        ``syndromes`` is accepted for interface compatibility but not used;
        corrections are derived from the hidden ``errors`` state.

        If ``range`` is ``None``, copies ``errors`` into ``corrections`` (full
        teacher forcing).  If ``range >= 1``, masks to error bits within
        ``range`` check-to-bit expansion steps of any active syndrome.

        Args:
            syndromes: ``[*B, num_detectors]`` (unused, interface only).
            range: ``None`` for full copy, or ``>= 1`` for adjacent masking.
                (Parameter name shadows the builtin ``range``; loops use
                ``builtins.range``.)
        """
        if self.errors is None:
            raise RuntimeError("Errors not initialized. Call reset_errors() first.")

        if range is None:
            self.corrections = self.errors.clone()
            return

        if range < 1:
            raise ValueError(f"range must be >= 1 when not None, got {range}")

        tg = self.dem.detector_tanner_graph
        check_values = tg.bit_to_check(self.errors)  # [..., num_detectors]

        # Alternate check-to-bit and bit-to-check propagation.
        for step_idx in builtins.range(range):
            bit_values = tg.check_to_bit(check_values, mod2=False)
            if step_idx < range - 1:
                check_values = tg.bit_to_check(bit_values, mod2=False)

        mask = (bit_values > 0).long()
        self.corrections = self.errors & mask

    def draw_on(self, draw_kwargs: dict, **style) -> None:
        """Annotate draw_kwargs with active error/syndrome/logical states.

        Called by ``DetectorErrorModel.draw``, which builds ``draw_kwargs``
        and passes the merged style as ``**style``.  Modifies
        ``draw_kwargs['node_color_dict']`` in place to highlight active
        nodes with their active colors from the style dictionary.

        Args:
            draw_kwargs: Mutable dict of ``nx.draw_networkx`` kwargs built by
                         ``DetectorErrorModel.draw``.
            **style: Style keyword arguments forwarded by ``DetectorErrorModel.draw``
                     (includes ``batch_index``, ``colors``, ``sizes``, etc.).
        """
        batch_index = style.get('batch_index', 0)
        colors = style['colors']
        node_color_dict = draw_kwargs['node_color_dict']
        simulator_state = {'error': None, 'detector': None, 'logical': None}
        try:
            if self.errors is not None:
                simulator_state['error'] = self.errors[batch_index].cpu().numpy()
                simulator_state['detector'] = self.syndromes[batch_index].cpu().numpy()
                simulator_state['logical'] = self.logicals[batch_index].cpu().numpy()
        except (AttributeError, IndexError, RuntimeError):
            return
        for node_type, state_vec in simulator_state.items():
            if state_vec is not None and len(state_vec) > 0:
                nodes_list = self.dem._get_nodes_by_type(node_type)
                for idx in range(min(len(state_vec), len(nodes_list))):
                    if int(state_vec[idx]) != 0:
                        node_color_dict[nodes_list[idx]] = colors[f'{node_type}_active']
