"""Tanner graph as a sparse tensor interface for quantum error correction."""

import torch
import numpy as np
import scipy.sparse
from typing import Dict, Optional
from torch_geometric.utils import scatter


class TannerGraph:
    """Bipartite Tanner graph as a sparse tensor interface.

    A lightweight, GPU-friendly representation of a bipartite graph between
    bit nodes and check nodes. All data is stored as PyTorch tensors; no
    NetworkX dependency at runtime. TannerGraph acts as a "tensor factory":
    it generates graph structure data (e.g. `edge_index`, dual graph, check
    matrix) in torch tensors suitable for use with PyTorch Geometric (PyG)
    for graph-based deep learning.

    Indexing convention (strictly enforced):
        - Bit nodes:   indices [0, 1, ..., num_bits - 1]
        - Check nodes: indices [num_bits, num_bits + 1, ..., num_bits + num_checks - 1]
        - num_nodes  = num_bits + num_checks

    Edge index convention:
        - edge_index: [2, num_edges] COO-format tensor (bidirectional)
        - Even columns (0, 2, 4, ...): bit -> check direction
        - Odd  columns (1, 3, 5, ...): check -> bit direction

    Device is inferred from ``edge_index.device``.

    DetectorErrorModel is responsible for constructing TannerGraph instances
    with consistent indexing.  Derived tensor properties (dual graph, check
    matrix, boundary nodes, ...) are lazily computed and cached on first access.
    """

    def __init__(
        self,
        num_bits: int,
        num_checks: int,
        edge_index: torch.Tensor,
        num_phys_nodes: Optional[int] = None,
    ):
        """Initialize TannerGraph.

        Args:
            num_bits:   Number of bit  nodes (indexed [0, num_bits)).
            num_checks: Number of check nodes (indexed [num_bits, num_bits + num_checks)).
            edge_index: [2, num_edges] COO edge-index tensor following the
                        bidirectional convention described above.  Device is
                        inferred from this tensor.
            num_phys_nodes: Number of physical nodes (bits + physical checks) before
                            boundary extension. Defaults to num_nodes when not extended.
        """
        self.num_bits = num_bits
        self.num_checks = num_checks
        self.edge_index = edge_index
        self.num_phys_nodes = num_phys_nodes if num_phys_nodes is not None else (num_bits + num_checks)

    def __repr__(self) -> str:
        """String representation of the TannerGraph."""
        terms = {'num_bits': self.num_bits, 'num_checks': self.num_checks, 'num_edges': self.num_edges}
        if self.edge_index.device != torch.device('cpu'):
            terms['device'] = f"'{str(self.edge_index.device)}'"
        return f"TannerGraph({', '.join([f'{k}={v}' for k, v in terms.items()])})"

    # ------------------------------------------------------------------
    # Core properties
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        """Total number of nodes (bits + checks)."""
        return self.num_bits + self.num_checks

    @property
    def num_edges(self) -> int:
        """Number of directed edges (each undirected edge counted twice)."""
        return self.edge_index.shape[1]

    @property
    def edge_index_b2c(self) -> torch.Tensor:
        """``[2, num_b2c_edges]`` bit -> check directed edges (even columns of ``edge_index``)."""
        return self.edge_index[:, ::2]

    @property
    def edge_index_c2b(self) -> torch.Tensor:
        """``[2, num_c2b_edges]`` check -> bit directed edges (odd columns of ``edge_index``)."""
        return self.edge_index[:, 1::2]

    @property
    def boundary_bit_nodes(self) -> torch.Tensor:
        """1-D tensor of bit-node indices with degree <= 1 (boundary bits)."""
        if not hasattr(self, '_boundary_bit_nodes'):
            device = self.edge_index.device
            b2c = self.edge_index_b2c
            degrees = torch.zeros(self.num_bits, dtype=torch.long, device=device)
            if b2c.numel() > 0:
                degrees.scatter_add_(
                    0, b2c[0],
                    torch.ones(b2c.shape[1], dtype=torch.long, device=device),
                )
            self._boundary_bit_nodes = torch.where(degrees <= 1)[0]
        return self._boundary_bit_nodes

    @property
    def edge_id_map(self) -> Dict[tuple[int, int], int]:
        """Mapping from directed edge ``(u, v)`` to its column index in
        ``edge_index``, lazily computed and cached."""
        if not hasattr(self, '_edge_id_map'):
            mapping: Dict[tuple[int, int], int] = {}
            for eid in range(self.num_edges):
                u = int(self.edge_index[0, eid])
                v = int(self.edge_index[1, eid])
                mapping[(u, v)] = eid
            self._edge_id_map = mapping
        return self._edge_id_map

    # ------------------------------------------------------------------
    # Check matrix
    # ------------------------------------------------------------------

    def check_matrix(self, format: str = 'torch_sparse'):
        """Return the parity-check matrix **H**.

        ``H[c, b] = 1`` iff bit *b* is connected to check *c*.
        Rows are indexed by checks ``[0 .. num_checks)``, columns by bits
        ``[0 .. num_bits)``.

        Args:
            format: Output format — one of

                * ``'torch_sparse'`` (default): ``torch.sparse_coo_tensor`` (GPU-compatible)
                * ``'torch'``:  dense ``torch.Tensor``
                * ``'scipy'``:  ``scipy.sparse.csc_matrix``
                * ``'numpy'``:  dense ``numpy.ndarray``

        Returns:
            Check matrix *H* in the requested format.
        """
        device = self.edge_index.device
        b2c = self.edge_index_b2c
        bit_idx = b2c[0]
        chk_idx = b2c[1] - self.num_bits

        if format == 'torch_sparse':
            indices = torch.stack([chk_idx, bit_idx])
            values = torch.ones(indices.shape[1], dtype=torch.float, device=device)
            return torch.sparse_coo_tensor(
                indices, values, (self.num_checks, self.num_bits), device=device
            )
        if format == 'torch':
            H = torch.zeros(self.num_checks, self.num_bits, dtype=torch.float, device=device)
            H[chk_idx, bit_idx] = 1.0
            return H
        if format == 'scipy':
            rows = chk_idx.cpu().numpy()
            cols = bit_idx.cpu().numpy()
            data = np.ones(len(rows), dtype=np.int64)
            return scipy.sparse.csc_matrix(
                (data, (rows, cols)), shape=(self.num_checks, self.num_bits)
            )
        if format == 'numpy':
            H = np.zeros((self.num_checks, self.num_bits), dtype=np.int64)
            H[chk_idx.cpu().numpy(), bit_idx.cpu().numpy()] = 1
            return H
        raise ValueError(
            f"Unknown format '{format}'. "
            "Must be 'torch_sparse', 'torch', 'scipy', or 'numpy'."
        )

    # ------------------------------------------------------------------
    # Sparse multiplication  (bit <-> check)
    # ------------------------------------------------------------------

    def bit_to_check(self, x: torch.Tensor, mod2: bool = True) -> torch.Tensor:
        """Compute check values from bit values:  ``y = H @ x``  (mod 2).

        Left-multiplication by the check matrix.  Maps bit-indexed vectors
        to check-indexed vectors.

        When ``mod2=True``, parity is ``sum of incident bits`` reduced mod 2
        (integer or boolean ``x`` only). Floating dtypes are not supported for
        parity.

        When ``mod2=False``, a plain scatter-sum is used (raw accumulation,
        no parity).

        Args:
            x: Bit-value tensor of shape ``(..., num_bits)``.
            mod2: If ``True``, return parity values; otherwise raw sums.

        Returns:
            Check-value tensor of shape ``(..., num_checks)``.
        """
        b2c = self.edge_index_b2c
        bit_idx = b2c[0]
        chk_idx = b2c[1] - self.num_bits

        if mod2 and x.dtype.is_floating_point:
            raise TypeError(
                "bit_to_check(..., mod2=True) expects integer or boolean x; "
                "floating-point relaxed parity is not supported."
            )
        x_src = x[..., bit_idx]
        y = scatter(x_src, chk_idx, dim=-1, dim_size=self.num_checks, reduce='sum')
        return y % 2 if mod2 else y

    def check_to_bit(self, x: torch.Tensor, mod2: bool = True) -> torch.Tensor:
        """Compute bit values from check values:  ``y = H^T @ x``  (mod 2).

        Right-multiplication by H^T.  Maps check-indexed vectors to
        bit-indexed vectors.

        When ``mod2=True``, parity is scatter-sum mod 2 (integer or boolean
        ``x`` only). Floating dtypes are not supported for parity.

        When ``mod2=False``, a plain scatter-sum is used (raw accumulation,
        no parity).

        Args:
            x: Check-value tensor of shape ``(..., num_checks)``.
            mod2: If ``True``, return parity values; otherwise raw sums.

        Returns:
            Bit-value tensor of shape ``(..., num_bits)``.
        """
        c2b = self.edge_index_c2b
        chk_idx = c2b[0] - self.num_bits
        bit_idx = c2b[1]

        if mod2 and x.dtype.is_floating_point:
            raise TypeError(
                "check_to_bit(..., mod2=True) expects integer or boolean x; "
                "floating-point relaxed parity is not supported."
            )
        x_src = x[..., chk_idx]
        y = scatter(x_src, bit_idx, dim=-1, dim_size=self.num_bits, reduce='sum')
        return y % 2 if mod2 else y

    # ------------------------------------------------------------------
    # Graph transforms
    # ------------------------------------------------------------------

    def extend_boundary_checks(self) -> 'TannerGraph':
        """Return a new TannerGraph with virtual checks appended for boundary bits.

        Each boundary bit node (degree <= 1) receives one new check node.
        New check indices start at ``self.num_nodes`` and are contiguous.

        Returns:
            A new ``TannerGraph`` with the additional check nodes and edges.
        """
        boundary_bits = self.boundary_bit_nodes
        num_new = boundary_bits.shape[0]
        if num_new == 0:
            return TannerGraph(self.num_bits, self.num_checks, self.edge_index.clone())

        device = self.edge_index.device
        new_chk = self.num_nodes + torch.arange(num_new, dtype=torch.long, device=device)
        # Interleave (bit->check, check->bit) pairs for each new edge
        new_b2c = torch.stack([boundary_bits, new_chk])        # [2, num_new]
        new_c2b = torch.stack([new_chk, boundary_bits])        # [2, num_new]
        new_edges = torch.stack([new_b2c, new_c2b], dim=-1).reshape(2, -1)  # [2, 2*num_new]

        ext_edge_index = torch.cat([self.edge_index, new_edges], dim=1)
        return TannerGraph(
            self.num_bits,
            self.num_checks + num_new,
            ext_edge_index,
            num_phys_nodes=self.num_nodes,
        )

    # ------------------------------------------------------------------
    # Dual graph  (for BP decoder)
    # ------------------------------------------------------------------

    @property
    def dual_edge_index(self) -> torch.Tensor:
        """Dual-graph edge index, lazily computed and cached.

        Dual graph: nodes are directed edges of the original Tanner graph;
        two dual nodes are connected when the corresponding original edges
        share a node (with the no-backtracking constraint: ``u->v->u``
        paths are excluded).

        For original edges ``e_j = (u->v)`` and ``e_i = (v->w)`` sharing
        node *v* where ``u != w``:  dual edge ``e_j -> e_i`` exists with
        connecting node *v*.

        Returns:
            ``[2, num_dual_edges]`` tensor.
        """
        if not hasattr(self, '_dual_edge_index'):
            self._compute_dual_graph()
        return self._dual_edge_index

    @property
    def connecting_node_ids(self) -> torch.Tensor:
        """Connecting (mediating) node index for each dual edge.

        For dual edge ``e_j -> e_i`` where ``e_j = (u->v)`` and
        ``e_i = (v->w)``, the connecting node is ``v``.

        Returns:
            ``[num_dual_edges]`` integer tensor of tanner node indices.
        """
        if not hasattr(self, '_connecting_node_ids'):
            self._compute_dual_graph()
        return self._connecting_node_ids

    # -- private -------------------------------------------------------

    def _compute_dual_graph(self) -> None:
        """Build dual graph from ``edge_index`` using pure tensor / Python ops."""
        device = self.edge_index.device
        src = self.edge_index[0]
        tgt = self.edge_index[1]

        dual_src: list[int] = []
        dual_tgt: list[int] = []
        connecting: list[int] = []

        for v in range(self.num_nodes):
            incoming = (tgt == v).nonzero(as_tuple=True)[0]
            outgoing = (src == v).nonzero(as_tuple=True)[0]
            if incoming.numel() == 0 or outgoing.numel() == 0:
                continue
            for e_j in incoming:
                u = src[e_j].item()
                for e_i in outgoing:
                    w = tgt[e_i].item()
                    if u != w:  # no backtracking
                        dual_src.append(e_j.item())
                        dual_tgt.append(e_i.item())
                        connecting.append(v)

        if dual_src:
            self._dual_edge_index = torch.tensor(
                [dual_src, dual_tgt], dtype=torch.long, device=device
            )
            self._connecting_node_ids = torch.tensor(
                connecting, dtype=torch.long, device=device
            )
        else:
            self._dual_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            self._connecting_node_ids = torch.empty((0,), dtype=torch.long, device=device)
