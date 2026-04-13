import networkx as nx
import torch
import numpy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from .tanner import TannerGraph


class DetectorErrorModel(nx.Graph):
    """Detector Error Model (DEM) for quantum error correction decoding.

    A DEM is the union of (1) a **Tanner graph** encoding which errors affect
    which detectors and logicals, and (2) an **error model** giving the
    stochastic update of the error vector. This class holds both. It will be 
    used to construct Simulator and Decoder.

    ---------------------------------------------------------------------------
    Mathematical structure
    ---------------------------------------------------------------------------

    **Tanner graph**
      The DEM Tanner graph has three node types:
      - **Error nodes** (bit nodes): candidate error events; index set for the
        physical error vector e.
      - **Detector nodes** (check nodes): syndrome/detection events; index set
        for the detector (syndrome) vector s.
      - **Logical nodes** (check nodes): logical observables; index set for
        the logical error vector l.

      Let E, D, L be the counts of error, detector, and logical nodes. In the
      ideal (noiseless) case:
        s = H @ e,   l = K @ e    (mod 2)
      where:
      - **Detector matrix** H (size D x E over Z2): H[d,i] = 1 iff error i is
        connected to detector d in the graph (edges error-detector define H).
      - **Logical matrix** K (size L x E over Z2): K[l,i] = 1 iff error i is
        connected to logical l (edges error-logical define K).
      So the DEM's bipartite subgraph (errors, detectors) encodes H; the
      subgraph (errors, logicals) encodes K.

    **Error model**
      The error vector evolves as a Markov process: e -> e' with distribution
      p(e'|e). The transition factorizes over error nodes:
        p(e'|e) = product over i of p(e'_i | e)
      Each factor is given by a **log-likelihood ratio (LLR)** w_i(e), also
      called **error weight**:
        p(e'_i | e) = sigmoid( (-1)^(e'_i) * w_i(e) )
      A positive w_i(e) means bit i is more likely 0 (no error).

      **General form of the error weight (LLR):**
        w_i(e) = w_i + sum over j of w_ij * (-1)^(e_j)
                 + sum over j,k of w_ijk * (-1)^(e_j + e_k) + ...
      The DEM stores these coefficients sparsely per error node in
      **error_weights**, a dict keyed by *context* tuples (only other error
      indices), without the current node index:

        ``error_weights = { (): w_i, (j,): w_ij, (j,k): w_ijk, ... }``

      So `()` = bias, `(j,)` = coupling from error j, `(j,k)` = two-body, etc.
      `get_error_model_data()` prepends the current error index to each tuple,
      groups by interaction order, and returns tensors for GPU evaluation.

      **Common compiled patterns (used by `add_error_node`):**
      - Independent error with rate p: `update_method='refresh'` gives
        `{(): llr(p)}`; `update_method='diffuse'` gives `{(i,): llr(p)}`.
      - Copy-from j: `{(j,): 100.0}` (enforces e'_i = e_j).
      - XOR-from j: `{(i, j): 100.0}` (enforces e'_i + e_j = 0 mod 2).

    ---------------------------------------------------------------------------
    Node naming and indexing conventions
    ---------------------------------------------------------------------------
    **Node names**
      - Node identifiers can be any hashable Python object; the DEM does not
        enforce a specific naming scheme.
      - Built-in code (e.g. code builders) typically uses `"E(...)"` for error
        nodes, `"D(...)"` for detector nodes, and `"L(...)"` for logical nodes.
      - After `time_window_extend(layers, ...)`, nodes are relabeled as
        `"<node>[t]"` for time layer t = 0, ..., layers-1.
      - Measurement-error nodes added by `time_window_extend()` are named
        `E(<detector_node>)[t]`.
      - Logical nodes are merged across time: time-labeled logical nodes are
        removed and the original logical name is restored (no `[t]` suffix).
      - The string representation of the node identifier (e.g. `str(node)`) is
        used as the node name when generating node-level prompts in the form
        `"<node_name>: <node_feature_description>"`.

    **Indexing**
      The DEM stores nodes by their (possibly string) identifiers. When
      exporting to tensors, consecutive integer indices are used:
      - **Error nodes**: `get_error_model_data()` and the Tanner graph builders
        map `error_nodes` to indices 0, 1, ..., num_errors - 1 in iteration
        order. The same order is used for `error_weights` and for the
        "current" error index in interaction tuples.
      - **Tanner graphs**: `detector_tanner_graph` and `logical_tanner_graph`
        map bit nodes (errors) to 0, ..., num_bits - 1 and check nodes
        (detectors or logicals) to num_bits, ..., num_bits + num_checks - 1.
        Original DEM node identifiers are not stored in the `TannerGraph`; only
        these integer indices are exposed (e.g. in `edge_index`, `check_matrix`).

    ---------------------------------------------------------------------------
    Key methods (with brief examples)
    ---------------------------------------------------------------------------
    **Node and edge management**
      - `add_error_node(name, error_rate=p, update_method='refresh'|'diffuse', 
        copy_from=..., xor_from=..., error_weights={...})`
        Adds an error node and compiles `error_weights` (e.g. from rate or
        copy_from/xor_from). Example: `dem.add_error_node("E(0)", error_rate=0.01)`.
      - `add_detector_node(name)`, `add_logical_node(name)`
      - `add_detector_edge(error_node, detector_node)`, `add_logical_edge(error_node, logical_node)`
        Adding an edge can auto-create the detector/logical node if missing.
      - `remove_error_node(name)`, `remove_detector_node(name)`, `remove_logical_node(name)`

    **Tanner graph export**
      - `detector_tanner_graph`: property yielding a `TannerGraph` (bits =
        errors, checks = detectors) with integer indices; bi-adjacency = H.
      - `logical_tanner_graph`: property yielding a `TannerGraph` (bits =
        errors, checks = logicals); bi-adjacency = K.
      After mutating the DEM, the caches are cleared automatically where
      possible; otherwise use internal `_invalidate_tanner_cache()` if needed.

    **Error model export**
      - `get_error_model_data()` -> `dict[int, tuple[Tensor, Tensor]]`
        Returns, for each interaction order, (indices, weights) tensors;
        indices have shape (num_terms, order) and include the current error
        index. Example: order 1 gives (indices [[i], ...], weights [w_i, ...]).

    **Time extension**
      - `time_window_extend(layers, measurement_noise=0.0)` -> `DetectorErrorModel`
        Returns a *new* DEM with multiple time layers (repeated syndrome
        rounds), measurement-error nodes between detector layers, and
        copy_from/xor_from linking error nodes across time. Original DEM is
        unchanged. Example: `dem_ext = dem.time_window_extend(3, measurement_noise=0.01)`.

    **Utilities**
      - `validate()` -> `bool`: structure and attribute checks.
      - `clear()`: remove all nodes/edges; resets Tanner graph caches.
      - `draw(ax, simulator=..., decoder=..., **style)`: matplotlib visualization.
    """
    
    def __init__(
        self,
        batch_size: Optional[int] = None,
        batch_shape: Optional[Sequence[int] | torch.Size] = None,
        device: Optional[torch.device | str] = None,
        **attr
    ):
        """Initialize DetectorErrorModel.

        Args:
            batch_size: Batch size for tensor operations
            batch_shape: Shape of the batch for tensor operations (overrides batch_size)
            device: Device for tensor operations ('cpu', 'cuda', or 'mps')
            **attr: Additional graph-level attributes passed to nx.Graph.__init__()
        """
        if 'batch_shape' not in attr:
            if batch_shape is None:
                if batch_size is None:
                    batch_shape = (1,) # default to batch size of 1
                else:
                    batch_shape = (batch_size,)
            attr['batch_shape'] = torch.Size(batch_shape)
        
        if 'device' not in attr:
            if device is None:
                device = 'cpu' # default to cpu
            attr['device'] = torch.device(device)
        
        super().__init__(**attr)
    
    def __repr__(self) -> str:
        """String representation of the DetectorErrorModel."""
        info = f"'{self.code_name}': {self.num_detectors}D-{self.num_errors}E-{self.num_logicals}L"
        if 'layers' in self.graph:
            info += f", t={self.graph['layers']}"
        if self.device != torch.device('cpu'):
            info += f", device='{str(self.device)}'"
        return f"DetectorErrorModel({info})"

    # Public Properties
    @property
    def error_nodes(self) -> List:
        """List of all error node identifiers."""
        return self._get_nodes_by_type('error')
    
    @property
    def detector_nodes(self) -> List:
        """List of all detector node identifiers."""
        return self._get_nodes_by_type('detector')
    
    @property
    def logical_nodes(self) -> List:
        """List of all logical node identifiers."""
        return self._get_nodes_by_type('logical')
    
    @property
    def num_errors(self) -> int:
        """Number of error nodes in the graph."""
        return len(self.error_nodes)
    
    @property
    def num_detectors(self) -> int:
        """Number of detector nodes in the graph."""
        return len(self.detector_nodes)
    
    @property
    def num_logicals(self) -> int:
        """Number of logical nodes in the graph."""
        return len(self.logical_nodes)

    @property
    def device(self) -> torch.device:
        """Device for tensor operations (from graph attributes). Always torch.device."""
        return torch.device(self.graph['device'])

    @property
    def batch_shape(self) -> torch.Size:
        """Batch shape for tensor operations (from graph attributes). Always torch.Size."""
        return torch.Size(self.graph['batch_shape'])

    @property
    def code_name(self) -> str:
        """Dynamically generated name combining code type and distance (e.g. 'surface5', 'repetition3') 
           for repr strings or file names etc."""
        return f"{self.graph['code_type']}{self.graph['code_distance']}"

    # Private Helper Methods
    
    def _get_nodes_by_type(self, node_type: str) -> List:
        """Get all nodes of a specific type.
        
        Args:
            node_type: 'error', 'detector', or 'logical'
        
        Returns:
            List of node identifiers
        """
        return [n for n in self.nodes() if self.nodes[n].get('type') == node_type]
    
    def _create_node_to_index_mapping(self, nodes: List) -> Dict[Any, int]:
        """Create a mapping from node identifiers to consecutive indices.
        
        Args:
            nodes: List of node identifiers
        
        Returns:
            Dictionary mapping node identifier to index
        """
        return {node: idx for idx, node in enumerate(nodes)}
    
    def _validate_node_type(self, node: Any, expected_type: str) -> None:
        """Validate that a node exists and has the expected type.
        
        Args:
            node: Node identifier
            expected_type: Expected node type ('error', 'detector', or 'logical')
        
        Raises:
            KeyError: If node doesn't exist
            ValueError: If node doesn't have the expected type
        """
        if node not in self.nodes():
            raise KeyError(f"Node {node} does not exist")
        actual_type = self.nodes[node].get('type')
        if actual_type != expected_type:
            raise ValueError(f"Node {node} has type '{actual_type}', expected '{expected_type}'")
    
    # Node Addition and Management Methods
    
    def add_error_node(self, error_node: Any, **attr) -> None:
        """Add an error node to the graph.
        
        Args:
            error_node: Node identifier (any hashable type)
            **attr: Additional node attributes
        """
        self.add_node(error_node, type='error', error_weights={}, **attr)
        self.update_error_node(error_node, **attr)

    def update_error_node(
        self,
        error_node: Any,
        error_rate: Optional[float] = None,
        update_method: str = 'refresh',
        copy_from: Optional[Any] = None,
        xor_from: Optional[Any] = None,
        error_weights: Optional[Dict] = None,
        **attr
    ) -> None:
        """Add an error node to the graph.
        
        Args:
            error_node: Node identifier (any hashable type)
            error_rate: Independent error probability/rate (must be in [0, 1] if provided)
            update_method: Error update method for error_rate ('refresh' or 'diffuse')
            copy_from: Error node identifier to copy from
            xor_from: Error node identifier to XOR with
            error_weights: Direct weight specification following simplified format
            **attr: Additional node attributes
        """
        compiled_weights = self.nodes[error_node].get('error_weights', {})
        
        # Compile rate
        if error_rate is not None:
            if update_method not in ('refresh', 'diffuse'):
                raise ValueError(f"update_method must be 'refresh' or 'diffuse', got {update_method}")
            if not (0 <= error_rate <= 1):
                raise ValueError(f"Rate must be in [0, 1], got {error_rate}")
            # Convert probability to LLR, bounded to avoid overflow
            if error_rate <= 0.0:
                llr = 100.0
            elif error_rate >= 1.0:
                llr = -100.0
            else:
                llr = torch.tensor((1 - error_rate)/error_rate).log().clamp(min=-100.0, max=100.0).item()
            if update_method == 'refresh':
                compiled_weights[()] = llr
            else:
                compiled_weights[(error_node,)] = llr
        
        # Compile copy_from
        if copy_from is not None:
            if copy_from not in self.error_nodes:
                raise ValueError(f"copy_from node {copy_from} is not a valid error node")
            compiled_weights[(copy_from,)] = 100.0
        
        # Compile xor_from
        if xor_from is not None:
            if xor_from not in self.error_nodes:
                raise ValueError(f"xor_from node {xor_from} is not a valid error node")
            compiled_weights[(error_node, xor_from)] = 100.0
        
        # Merge with user-provided error_weights (user-provided takes precedence)
        if error_weights is not None:
            compiled_weights.update(error_weights)
        
        # Default: if all parameters are None, set default (error turned off)
        if not compiled_weights:
            compiled_weights = {(): 100.0}
        
        self.nodes[error_node]['error_weights'] = compiled_weights

    def update_error_model(
        self,
        nodes: Optional[Union[Sequence[Any], Callable[[Any, Dict[str, Any]], bool]]] = None,
        **kwargs: Any,
    ) -> None:
        """Update error parameters on one or more error nodes.

        Calls :meth:`update_error_node` for each targeted node.  All keyword
        arguments accepted by :meth:`update_error_node` are supported
        (``error_rate``, ``update_method``, ``copy_from``, ``xor_from``,
        ``error_weights``, etc.).

        Args:
            nodes: Which error nodes to update:
                - ``None`` (default): all error nodes.
                - A sequence of node identifiers: only those nodes.
                - A callable ``(node_name, node_attrs) -> bool``: nodes for
                  which the predicate is true.
            **kwargs: Forwarded to :meth:`update_error_node` for each target.
        """
        if nodes is None:
            targets = self.error_nodes
        elif callable(nodes):
            targets = [n for n in self.error_nodes if nodes(n, self.nodes[n])]
        else:
            targets = list(nodes)

        for node in targets:
            self.update_error_node(node, **kwargs)

    def add_detector_node(self, detector_node: Any, **attr) -> None:
        """Add a detector node to the graph.
        
        Args:
            detector_node: Node identifier (any hashable type)
            **attr: Additional node attributes
        """
        self.add_node(detector_node, type='detector', **attr)
    
    def add_logical_node(self, logical_node: Any, **attr) -> None:
        """Add a logical observable node to the graph.
        
        Args:
            logical_node: Node identifier (any hashable type)
            **attr: Additional node attributes
        """
        self.add_node(logical_node, type='logical', **attr)
    
    def remove_error_node(self, error_node: Any) -> None:
        """Remove an error node from the graph.
        
        Args:
            error_node: Error node identifier
        """
        self._validate_node_type(error_node, 'error')
        self.remove_node(error_node)
    
    def remove_detector_node(self, detector_node: Any) -> None:
        """Remove a detector node from the graph.
        
        Args:
            detector_node: Detector node identifier
        """
        self._validate_node_type(detector_node, 'detector')
        self.remove_node(detector_node)
    
    def remove_logical_node(self, logical_node: Any) -> None:
        """Remove a logical node from the graph.
        
        Args:
            logical_node: Logical node identifier
        """
        self._validate_node_type(logical_node, 'logical')
        self.remove_node(logical_node)
    
    # Edge Management Methods
    
    def add_detector_edge(self, error_node: Any, detector_node: Any, **attr) -> None:
        """Add an edge between an error and a detector node.
        
        Args:
            error_node: Error node identifier
            detector_node: Detector node identifier
            **attr: Edge attributes
        """
        self._validate_node_type(error_node, 'error')
        if detector_node not in self.nodes():
            self.add_detector_node(detector_node)
        self.add_edge(error_node, detector_node, **attr)
    
    def add_logical_edge(self, error_node: Any, logical_node: Any, **attr) -> None:
        """Add an edge between an error and a logical node.
        
        Args:
            error_node: Error node identifier
            logical_node: Logical node identifier
            **attr: Edge attributes
        """
        self._validate_node_type(error_node, 'error')
        if logical_node not in self.nodes():
            self.add_logical_node(logical_node)
        self.add_edge(error_node, logical_node, **attr)
    
    def time_window_extend(self, layers: int, measurement_noise: float | list | torch.Tensor = 0.0) -> 'DetectorErrorModel':
        """Extend the DEM in time by creating multiple layers for multiple rounds of syndrome extraction.
        
        Creates a new DEM instance with time-extended graph structure, leaving the original DEM unchanged.
        
        Args:
            layers: Number of time layers to create (must be >= 1)
            measurement_noise: Measurement error rate(s). Can be:
                - float: Uniform measurement noise for all detectors
                - list/torch.Tensor: Individual measurement noise for each detector (length must match number of detectors)
        
        Returns:
            New DetectorErrorModel instance with time-extended graph
        """
        if layers < 1:
            raise ValueError(f"layers must be >= 1, got {layers}")
        
        # Collect original detector nodes for measurement noise handling
        original_detector_nodes = list(self.detector_nodes)
        original_error_nodes = list(self.error_nodes)
        original_logical_nodes = list(self.logical_nodes)
        
        # Collect which error nodes are connected to logical nodes
        logical_connections = {}  # {logical_node: [error_nodes]}
        for logical_node in original_logical_nodes:
            logical_connections[logical_node] = [
                error_node for error_node in self.neighbors(logical_node)
                if self.nodes[error_node].get('type') == 'error'
            ]
        
        # Handle measurement_noise input
        if isinstance(measurement_noise, (float, int)):
            measurement_noises = [float(measurement_noise)] * len(original_detector_nodes)
        else:
            measurement_noises = list(measurement_noise)
            if len(measurement_noises) != len(original_detector_nodes):
                raise ValueError(
                    f"measurement_noise length ({len(measurement_noises)}) must match number of detectors ({len(original_detector_nodes)})"
                )
            measurement_noises = [float(noise) for noise in measurement_noises]
        
        # Create time layers using NetworkX copy and relabel
        layer_graphs = []
        for t in range(layers):
            # Copy graph and relabel nodes with time label [t]
            layer_graph = self.copy()
            
            # Collect positions before relabeling
            node_positions = {}
            for node in layer_graph.nodes():
                original_pos = layer_graph.nodes[node].get('pos')
                if original_pos is not None:
                    # Shift position by (0, t) for each layer
                    if isinstance(original_pos, (tuple, list)) and len(original_pos) >= 2:
                        node_positions[node] = (original_pos[0], original_pos[1] + t)
                    else:
                        node_positions[node] = original_pos
            
            # Relabel nodes
            mapping = {node: f"{node}[{t}]" for node in layer_graph.nodes()}
            nx.relabel_nodes(layer_graph, mapping, copy=False)
            
            # Update positions after relabeling
            for original_node, new_pos in node_positions.items():
                time_labeled_node = f"{original_node}[{t}]"
                layer_graph.nodes[time_labeled_node]['pos'] = new_pos
            
            layer_graphs.append(layer_graph)
        
        # Merge all layers using compose
        # Inherit graph attributes from original DEM (batch_shape, device, etc.) and add layers attribute
        extended_dem = DetectorErrorModel(**self.graph)
        extended_dem.graph['layers'] = layers
        for layer_graph in layer_graphs:
            extended_dem = nx.compose(extended_dem, layer_graph)
        
        # Remove time-labeled logical nodes and create merged logical nodes
        for logical_node in original_logical_nodes:
            # Get original logical node position
            original_pos = self.nodes[logical_node].get('pos') if logical_node in self.nodes() else None
            
            # Remove all time-labeled logical nodes
            for t in range(layers):
                time_labeled_logical = f"{logical_node}[{t}]"
                if time_labeled_logical in extended_dem.nodes():
                    extended_dem.remove_logical_node(time_labeled_logical)
            
            # Create single logical node (without time label) at original position
            if original_pos is not None:
                extended_dem.add_logical_node(logical_node, pos=original_pos)
            else:
                extended_dem.add_logical_node(logical_node)
            
            # Connect to all corresponding error nodes from all layers
            for error_node in logical_connections.get(logical_node, []):
                for t in range(layers):
                    time_labeled_error = f"{error_node}[{t}]"
                    if time_labeled_error in extended_dem.nodes():
                        extended_dem.add_logical_edge(time_labeled_error, logical_node)
        
        # Update error models for layers t > 0
        for t in range(1, layers):
            for error_node in original_error_nodes:
                time_labeled_node = f"{error_node}[{t}]"
                
                if t < layers - 1:
                    # Intermediate layers: copy from previous layer
                    prev_node = f"{error_node}[{t-1}]"
                    extended_dem.add_error_node(time_labeled_node, copy_from=prev_node)
                else:
                    # Last layer: XOR from second-to-last layer
                    prev_node = f"{error_node}[{t-1}]"
                    extended_dem.add_error_node(time_labeled_node, xor_from=prev_node)
        
        # Add measurement error nodes between consecutive detector layers
        for t in range(layers - 1):
            for i, detector_node in enumerate(original_detector_nodes):
                measurement_error_node = f"E({detector_node})[{t}]"
                detector_t = f"{detector_node}[{t}]"
                detector_t_next = f"{detector_node}[{t+1}]"
                
                node_attr = {} # prepare node attributes
                
                # Set error model for measurement error node
                if t == 0:
                    node_attr['error_rate'] = measurement_noises[i]
                else:
                    node_attr['copy_from'] = f"E({detector_node})[{t-1}]"
                
                # Get detector position and set measurement error position at (0, 0.5) shift
                detector_pos = extended_dem.nodes[detector_t].get('pos')
                if detector_pos is not None and isinstance(detector_pos, (tuple, list)) and len(detector_pos) >= 2:
                    node_attr['pos'] = (detector_pos[0], detector_pos[1] + 0.5)
                
                # Add measurement error node with measurement noise
                extended_dem.add_error_node(measurement_error_node, **node_attr)
                
                # Connect measurement error to both detector layers
                extended_dem.add_detector_edge(measurement_error_node, detector_t)
                extended_dem.add_detector_edge(measurement_error_node, detector_t_next)
        
        return extended_dem
    
    # Graph Structure Access (TannerGraph Interface)

    @property
    def detector_tanner_graph(self) -> TannerGraph:
        """Cached error-detector Tanner graph (bits = errors, checks = detectors).

        Returns:
            :class:`TannerGraph` with error nodes as bits and detectors as checks.
        """
        if not hasattr(self, '_detector_tanner_graph'):
            self._detector_tanner_graph = self._build_tanner_graph('error', 'detector')
        return self._detector_tanner_graph

    @property
    def logical_tanner_graph(self) -> TannerGraph:
        """Cached error-logical Tanner graph (bits = errors, checks = logicals).

        Returns:
            :class:`TannerGraph` with error nodes as bits and logicals as checks.
        """
        if not hasattr(self, '_logical_tanner_graph'):
            self._logical_tanner_graph = self._build_tanner_graph('error', 'logical')
        return self._logical_tanner_graph

    def _invalidate_tanner_cache(self) -> None:
        """Invalidate cached TannerGraph instances (call after graph mutation)."""
        for attr in ('_detector_tanner_graph', '_logical_tanner_graph'):
            if hasattr(self, attr):
                delattr(self, attr)

    def _build_tanner_graph(self, bit_type: str, check_type: str) -> TannerGraph:
        """Build a TannerGraph from the DEM's current graph structure.

        DEM controls the indexing: bit nodes are mapped to [0, num_bits) and
        check nodes to [num_bits, num_bits + num_checks), consistent with the
        TannerGraph indexing convention.

        Args:
            bit_type:   Node type for bit  nodes (e.g. ``'error'``).
            check_type: Node type for check nodes (e.g. ``'detector'`` or ``'logical'``).

        Returns:
            A new ``TannerGraph`` instance.
        """
        bit_nodes = self._get_nodes_by_type(bit_type)
        check_nodes = self._get_nodes_by_type(check_type)
        num_bits = len(bit_nodes)

        # DEM-controlled indexing: bits -> [0, num_bits), checks -> [num_bits, num_nodes)
        node_to_idx = self._create_node_to_index_mapping(bit_nodes + check_nodes)

        # Build edge_index directly (no intermediate nx.Graph)
        edges: list[list[int]] = []
        for check_node in check_nodes:
            chk_idx = node_to_idx[check_node]
            for neighbor in self.neighbors(check_node):
                if self.nodes[neighbor].get('type') == bit_type:
                    bit_idx = node_to_idx[neighbor]
                    edges.append([bit_idx, chk_idx])   # bit -> check
                    edges.append([chk_idx, bit_idx])   # check -> bit

        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).t()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)

        return TannerGraph(num_bits, len(check_nodes), edge_index)
    
    def get_error_model_data(self) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """Get compiled error model as tensors for GPU computation.
        
        Returns:
            Dictionary mapping interaction order to (indices_tensor, weights_tensor) pairs
        """
        # Collect error nodes and create mapping (reuse property and helper)
        error_nodes = self.error_nodes
        error_node_to_idx = self._create_node_to_index_mapping(error_nodes)
        
        # Group weights by interaction order
        weights_by_order = {}  # {order: [(indices_list, weight), ...]}
        
        for error_node in error_nodes:
            error_idx = error_node_to_idx[error_node]
            error_weights = self.nodes[error_node].get('error_weights', {})
            
            for interaction_tuple, weight in error_weights.items():
                # interaction_tuple is in simplified format (without leading error_node)
                # Need to prepend error_idx to get full interaction indices
                # Works for both empty tuple () -> (error_idx,) and non-empty (j, k, ...) -> (error_idx, j, k, ...)
                order = len(interaction_tuple) + 1
                indices = [error_idx] + [error_node_to_idx[node_id] for node_id in interaction_tuple]
                
                if order not in weights_by_order:
                    weights_by_order[order] = []
                weights_by_order[order].append((indices, weight))
        
        # Convert to tensors
        result = {}
        for order, interactions in weights_by_order.items():
            if interactions:
                indices_list = [idx for idx, _ in interactions]
                weights_list = [w for _, w in interactions]
                
                indices_tensor = torch.tensor(indices_list, dtype=torch.long, device=self.device)
                weights_tensor = torch.tensor(weights_list, dtype=torch.float, device=self.device)
                
                result[order] = (indices_tensor, weights_tensor)
        
        return result

    # Error Weights
    def get_weights(self, format: str = 'torch') -> torch.Tensor:
        """Get error weights as a tensor.

        Args:
            format: Output format — one of
                * ``'torch'`` (default): ``torch.Tensor``
                * ``'numpy'``: ``numpy.ndarray``

        Returns:
            Weight vector of length ``num_errors``, or ``None`` if any
            error node lacks a bias weight.
        """
        weights = []
        for node in self.error_nodes:
            error_weights = self.nodes[node]['error_weights']
            if () in error_weights:
                weights.append(error_weights[()])
            elif (node,) in error_weights:
                weights.append(error_weights[(node,)])
            else:
                return None
        if format == 'torch':
            return torch.tensor(weights, dtype=torch.float, device=self.device)
        elif format == 'numpy':
            return numpy.array(weights, dtype=numpy.float32)
        else:
            raise ValueError(f"Invalid format: {format}. Must be 'torch' or 'numpy'.")

    # Prompt Preparation
    def generate_node_prompts(
        self,
        *node_types: str,
        encode_weights: bool = False,
        add_ans: bool = False,
        add_syn: bool = False,
    ) -> List[str]:
        """Generate node prompts for a given list of node types.

        Args:
            node_types: Node types to include (e.g. ``"error"``, ``"detector"``).
            encode_weights: If True, append error weights to error node prompts.
            add_ans: If True, append ``<ans>`` token to each prompt.
            add_syn: If True, append ``<syn>`` placeholder to detector prompts
                (for later syndrome injection via ``torch.where``).

        Returns:
            List of prompt strings, one per node, in the order requested.

        Examples:
            >>> dem.generate_node_prompts("error", encode_weights=True, add_ans=True)
            ['E(0): 2.20 <ans>', 'E(1): 2.20 <ans>', ...]
            >>> dem.generate_node_prompts("detector", add_syn=True, add_ans=True)
            ['D(0): <syn> <ans>', 'D(1): <syn> <ans>', ...]
        """
        nodes = []
        for node_type in node_types:
            nodes.extend(self._get_nodes_by_type(node_type))

        node_prompts = []
        for node in nodes:
            node_type = self.nodes[node].get('type')
            node_prompt = str(node)

            if encode_weights and node_type == 'error':
                error_weights = self.nodes[node].get('error_weights', {})
                if () in error_weights:
                    weight = error_weights[()]
                elif (node,) in error_weights:
                    weight = error_weights[(node,)]
                else:
                    weight = None
                if weight is not None:
                    node_prompt += f": {weight:.2f}"

            if add_syn and node_type == 'detector':
                node_prompt += ": <syn>"

            if add_ans:
                node_prompt += " <ans>"

            node_prompts.append(node_prompt)
        return node_prompts

    def generate_system_prompt(self) -> str:
        """Generate system prompt for the DEM.

        Encodes predetermined system-level information: code type, distance,
        and representative error rate. These are calibration parameters
        available at deployment time, NOT runtime observables — satisfies
        the locality constraint for local decoders.

        Returns:
            System prompt string with tokenizable fields.
        """
        system_prompt = ""
        for key, value in self.graph.items():
            if key == 'code_type':
                system_prompt += f"code type: {value}, "
            elif key == 'code_distance':
                system_prompt += f"code distance: {value}, "
            elif key == 'layers':
                system_prompt += f"layers: {value}, "

        # Include representative error rate from the first error node's bias weight
        error_nodes = self._get_nodes_by_type('error')
        if error_nodes:
            first_node = error_nodes[0]
            weights = self.nodes[first_node].get('error_weights', {})
            bias = weights.get((), weights.get((first_node,), None))
            if bias is not None:
                # Convert LLR back to probability: p = sigmoid(-llr)
                import math
                p = 1.0 / (1.0 + math.exp(bias))
                # Round to 2 decimal places for clean tokenization
                system_prompt += f"error rate: {p:.2f}, "

        return system_prompt

    # Utility Methods
    def validate(self) -> bool:
        """Validate the DEM structure.
        
        Returns:
            True if valid
        """
        # Validate error rates
        for error_node in self.error_nodes:
            rate = self.nodes[error_node].get('rate')
            if rate is not None and not (0 <= rate <= 1):
                return False
            
            # Validate copy_from and xor_from reference valid error nodes
            copy_from = self.nodes[error_node].get('copy_from')
            if copy_from is not None and copy_from not in self.error_nodes:
                return False
            
            xor_from = self.nodes[error_node].get('xor_from')
            if xor_from is not None and xor_from not in self.error_nodes:
                return False
        
        # Validate node types
        valid_types = {'error', 'detector', 'logical'}
        for node in self.nodes():
            node_type = self.nodes[node].get('type')
            if node_type not in valid_types:
                return False
        
        # Validate edge types (error-detector, error-logical only)
        for u, v in self.edges():
            u_type = self.nodes[u].get('type')
            v_type = self.nodes[v].get('type')
            
            # Must be error-detector or error-logical
            if u_type == 'error' and v_type in {'detector', 'logical'}:
                continue
            if v_type == 'error' and u_type in {'detector', 'logical'}:
                continue
            
            # No direct detector-logical edges
            if {u_type, v_type} == {'detector', 'logical'}:
                return False
            
            # No edges within same type (except error-error, which is allowed but unusual)
            if u_type == v_type and u_type != 'error':
                return False
        
        return True
    
    def clear(self) -> None:
        """Clear all nodes and edges and reset to empty graph."""
        self._invalidate_tanner_cache()
        super().clear()
    
    def draw(
        self,
        ax: Optional[plt.Axes] = None,
        simulator: Optional[Any] = None,
        decoder: Optional[Any] = None,
        **style,
    ) -> None:
        """Draw the DEM graph with node colors by type.

        Orchestrates visualization by building a ``draw_kwargs`` dict for
        ``nx.draw_networkx``, then letting simulator and decoder annotate it
        via their ``draw_on`` methods before the final draw call.

        Args:
            ax: Matplotlib axis (if None, creates new figure).
            simulator: Optional Simulator instance whose states are drawn.
            decoder: Optional Decoder instance whose states are drawn.
            **style: Style overrides merged with ``DEM_PLOT_STYLE`` defaults.
                     Supports top-level keys ``batch_index`` (``int`` or
                     ``tuple`` for multi-dim batches, default ``0``),
                     ``pos`` (node positions dict), ``label`` (``None``,
                     ``'name'``, or ``'index'``), and nested dicts
                     ``colors``, ``sizes``, ``legend``, etc.

        Example:
            >>> dem.draw()  # No labels
            >>> dem.draw(label='name')  # Show original labels
            >>> dem.draw(simulator=sim, decoder=dec)
            >>> dem.draw(simulator=sim, batch_index=3) 
            >>> dem.draw(decoder=dec, batch_index=(0, 2))  # Multi-dim batch
            >>> dem.draw(colors={'detector': (1.0, 0.5, 0.5)})
        """
        # Merge style dictionaries: user style > DEM_PLOT_STYLE
        # helper function needed for merging nested dictionaries recursively
        def _merge_style_dicts(default: Dict, custom: Dict) -> Dict:
            result = default.copy()
            for key, value in custom.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = _merge_style_dicts(result[key], value)
                else:
                    result[key] = value
            return result
        plot_style = _merge_style_dicts(DEM_PLOT_STYLE, style) if style else DEM_PLOT_STYLE

        # Extract style values
        colors = plot_style['colors']
        sizes = plot_style['sizes']
        pos = plot_style.get('pos')
        label = plot_style.get('label')

        # Get or create axis
        if ax is None:
            figsize = plot_style['figure']['figsize']
            fig, ax = plt.subplots(figsize=figsize)
            fig.patch.set_facecolor(plot_style['figure']['facecolor'])
        else:
            fig = ax.figure

        # Determine layout: style pos > node pos attribute > bipartite_layout
        if pos is None:
            pos = {n: self.nodes[n].get('pos') for n in self.nodes()}
            if None in pos.values():
                pos = nx.bipartite_layout(self, self.error_nodes)

        # Build base node colors by type
        node_color_dict = {
            node: colors.get(self.nodes[node].get('type'), colors['unknown'])
            for node in self.nodes()
        }

        # Build draw_kwargs for nx.draw_networkx
        draw_kwargs = {
            'pos': pos,
            'node_color_dict': node_color_dict,
            'node_size': sizes['node'],
            'font_size': sizes['font'],
            'font_color': colors['font'],
            'edge_color': colors['edge'],
            'width': sizes['edge_width'],
            'edgecolors': None,
            'linewidths': None,
            'with_labels': False,
            'labels': None,
            'ax': ax,
        }

        with torch.no_grad():
            # Let simulator annotate (modifies node_color_dict for active states)
            if simulator is not None:
                from ..core.simulator import Simulator
                if isinstance(simulator, Simulator):
                    simulator.draw_on(draw_kwargs, **plot_style)
                else: # don't raise, draw nothing
                    pass

            # Let decoder annotate (messages + corrections)
            if decoder is not None:
                from ..core.decoder import Decoder
                from ..core.simulator import Simulator
                if isinstance(decoder, Decoder):
                    decoder.draw_on(draw_kwargs, **plot_style)
                elif isinstance(decoder, Simulator):
                    # Hack: when decoder is actually a Simulator, use Decoder.draw_on
                    # to draw corrections (e.g. Simulator.update_corrections) with decoder style
                    Decoder.draw_on(decoder, draw_kwargs, **plot_style)
                else: # don't raise, draw nothing
                    pass

        # Determine labels from style
        if label == 'name':
            draw_kwargs['with_labels'] = True
        elif label == 'index':
            draw_kwargs['with_labels'] = True
            all_tanner_nodes = self.error_nodes + self.detector_nodes
            node_to_idx = self._create_node_to_index_mapping(all_tanner_nodes)
            draw_kwargs['labels'] = {
                node: str(node_to_idx[node]) for node in all_tanner_nodes
            }

        # Convert node_color_dict to node_color list for nx.draw_networkx
        node_color_dict = draw_kwargs.pop('node_color_dict')
        draw_kwargs['node_color'] = [node_color_dict[n] for n in self.nodes()]

        # Draw the graph
        nx.draw_networkx(self, **draw_kwargs)
        
        # Remove frame surrounding the graph using style settings
        axes_style = plot_style['axes']
        ax.set_frame_on(axes_style['frame_on'])
        if not axes_style['show_ticks']:
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Add legend at the bottom in a row using style settings
        legend_style = plot_style['legend']
        legend_labels = plot_style['labels']
        
        # Create round disk markers without boundaries (like the nodes)
        # Use Line2D with marker from style for round markers matching the nodes
        marker_size = sizes['node'] / legend_style['marker_size_factor']
        
        # Create legend elements using a loop (direct mapping, no name translation)
        node_types = ['detector', 'error', 'logical']
        legend_elements = [
            Line2D(
                [0], [0],
                marker=legend_style['marker'],
                markersize=marker_size,
                markerfacecolor=colors[node_type],
                markeredgecolor=legend_style['marker_edgecolor'],
                linestyle=legend_style['linestyle'],
                label=legend_labels[node_type]
            )
            for node_type in node_types
        ]
        ax.legend(
            handles=legend_elements,
            loc=legend_style['loc'],
            bbox_to_anchor=legend_style['bbox_to_anchor'],
            ncol=legend_style['ncol'],
            frameon=legend_style['frameon'],
            fancybox=legend_style['fancybox'],
            shadow=legend_style['shadow'],
            framealpha=legend_style['framealpha']
        )
        
        # Adjust layout to make room for legend using style settings
        figure_style = plot_style['figure']
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", UserWarning)
                fig.tight_layout(pad=figure_style['tight_layout_pad'])
                if any("Tight layout not applied" in str(w.message) for w in caught):
                    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.08, top=0.95)
        except Exception:
            fig.subplots_adjust(left=0.06, right=0.94, bottom=0.08, top=0.95)
        

# Default style dictionary for DEM visualization
# Can be customized by modifying this dictionary or creating a custom one
DEM_PLOT_STYLE = {
    'batch_index': 0,                # Batch index for simulator/decoder states (int or tuple for multi-dim batches)
    'pos': None,                     # Default positions (None: auto-detect from node attributes or bipartite_layout)
    'label': None,                   # Default label mode (None: no labels, 'name': node labels, 'index': Tanner graph indices)
    'colors': {
        'detector': (0.75, 0.86, 0.98),    # Light blue for detector nodes
        'error': (0.95, 0.8, 0.75),        # Light red for error nodes
        'logical': (0.75, 0.9, 0.79),      # Light green for logical nodes
        'detector_active': (0.25, 0.59, 0.92), # Darker blue for active detector nodes
        'error_active': (0.84, 0.4, 0.25),     # Darker red for active error nodes
        'logical_active': (0.25, 0.7, 0.36),   # Darker green for active logical nodes
        'edge': 'gray',                    # Edge color
        'font': 'black',                   # Font color for labels
        'msg_color': 'black',              # Message text color
        'correction_color': (0.51, 0.35, 0.77), # Correction color
        'unknown': 'gray',                 # Color for unknown node types
    },
    'sizes': {
        'node': 500,                     # Default node size
        'font': 10,                      # Default font size
        'msg_font': 8,                   # Font size for message buffer source index text
        'edge_width': 1.0,               # Default edge width
        'check_boundary': 4.0,           # Thickness of boundary on check nodes
    },
    'legend': {
        'edgecolor': 'black',            # Legend patch edge color
        'linewidth': 0.5,                # Legend patch edge linewidth
        'ncol': 3,                       # Number of columns in legend
        'loc': 'upper center',           # Legend location
        'bbox_to_anchor': (0.5, 0.0),    # Legend bbox anchor (vertically close to graph)
        'frameon': False,                # Show legend frame (set to False for flat style)
        'fancybox': False,               # Use fancy box for legend
        'shadow': False,                 # Show shadow on legend
        'framealpha': 0,                 # Legend frame transparency
        'marker': 'o',                   # Legend marker type (round disk)
        'marker_edgecolor': 'none',      # Legend marker edge color (no boundary)
        'linestyle': 'None',             # Legend line style (no line, just marker)
        'marker_size_factor': 30,        # Factor to scale marker size relative to node size
    },
    'axes': {
        'frame_on': False,               # Show axes frame
        'show_ticks': False,             # Show axes ticks
    },
    'figure': {
        'figsize': (8.0, 6.4),           # Default figure size (width, height) - 80% of original (10, 8)
        'facecolor': 'white',            # Figure face color
        'tight_layout_pad': 0.5,         # Padding for tight_layout
    },
    'labels': {
        'detector': 'Detector',          # Label for detector nodes
        'error': 'Error',                # Label for error nodes
        'logical': 'Logical',            # Label for logical nodes
    },
    'misc': {
        'age_decay_rate': 0.3,           # Decay rate for age-based color mapping (c = exp(-rate*A))
    }
}

