"""Code constructors for quantum error correction."""

import torch
from .dem import DetectorErrorModel

# ---------------------------- Repetition code ----------------------------
def repetition_code(
    code_distance: int,
    error_rate: float | list | torch.Tensor = 0.0,
    update_method: str = 'refresh',
    batch_size: int = 1,
    device: torch.device | str = 'cpu') -> DetectorErrorModel:
    """Create a DetectorErrorModel for a 1D repetition code.
    
    The Tanner graph structure is: E0-D0-E1-D1-E2-D2-...-En-1-L0
    where Ei are error nodes, Di are detector nodes, and L0 is the logical node.
    
    Args:
        code_distance: Number of error nodes (n)
        error_rate: Error rate(s) for error nodes. Can be:
            - float: Uniform error rate for all error nodes
            - list/torch.Tensor: Individual error rates for each error node (length must match code_distance)
        update_method: Error update method for error_rate ('refresh' or 'diffuse')
        batch_size: Batch size for simulation (stored in dem.graph; shared by simulator/decoder).
        device: Device for tensor operations.

    Returns:
        DetectorErrorModel instance representing the repetition code

    Example:
        >>> dem = repetition_code(code_distance=4, error_rate=0.1)
        >>> dem = repetition_code(code_distance=4, error_rate=[0.1, 0.2, 0.15, 0.1])
        >>> dem = repetition_code(code_distance=4, batch_size=32, device='cuda')
    """
    dem = DetectorErrorModel(
        batch_size=batch_size,
        device=device,
        code_type='repetition',
        code_distance=code_distance
    )

    # Handle error_rate input
    if isinstance(error_rate, (float, int)):
        # Uniform error rate
        error_rates = [float(error_rate)] * code_distance
    else:
        # Iterable (list, torch.Tensor, etc.)
        error_rates = list(error_rate)
        if len(error_rates) != code_distance:
            raise ValueError(
                f"error_rate length ({len(error_rates)}) must match code_distance ({code_distance})"
            )
        error_rates = [float(rate) for rate in error_rates]
    
    # Add error nodes with their rates
    for i in range(code_distance):
        dem.add_error_node(
            f'E({i})',
            error_rate=error_rates[i],
            update_method=update_method,
            pos=(i,0)
        )
    
    # Add detector nodes and connect them
    # For n error nodes, we have n-1 detector nodes
    for i in range(code_distance - 1):
        detector_node = f'D({i})'
        dem.add_detector_node(detector_node, pos=(i+0.5,0))
        # Connect error node i and i+1 to detector i
        dem.add_detector_edge(f'E({i})', detector_node)
        dem.add_detector_edge(f'E({i+1})', detector_node)
    
    # Add logical node and connect to last error node
    dem.add_logical_node('L(0)', pos=(code_distance-0.5,0))
    dem.add_logical_edge(f'E({code_distance-1})', 'L(0)')
    
    return dem

# ---------------------------- Surface code ----------------------------
def surface_code(
    code_distance: int,
    error_rate: float | list | torch.Tensor = 0.0,
    update_method: str = 'refresh',
    batch_size: int = 1,
    device: torch.device | str = 'cpu') -> DetectorErrorModel:
    """Create a DetectorErrorModel for a surface code.
    
    The surface code is defined on a d x d grid where:
    - Error nodes E(i,j) for i, j in range(d)
    - Bulk detectors D(i,j) on even plaquettes check 4 errors
    - Boundary detectors check 2 errors along top and bottom boundaries
    - Logical observable L0 checks all errors in the last row
    
    Args:
        code_distance: Code distance (d), defines a d x d grid
        error_rate: Error rate(s) for error nodes. Can be:
            - float: Uniform error rate for all error nodes
            - torch.Tensor: 2D tensor of shape (d, d) with individual error rates
        update_method: Error update method for error_rate ('refresh' or 'diffuse')
        batch_size: Batch size for simulation (stored in dem.graph; shared by simulator/decoder).
        device: Device for tensor operations.

    Returns:
        DetectorErrorModel instance representing the surface code

    Example:
        >>> dem = surface_code(code_distance=3, error_rate=0.1)
        >>> dem = surface_code(code_distance=3, error_rate=torch.ones(3, 3) * 0.1)
        >>> dem = surface_code(code_distance=3, batch_size=32, device='cuda')
    """
    dem = DetectorErrorModel(
        batch_size=batch_size,
        device=device,
        code_type='surface',
        code_distance=code_distance
    )

    # Handle error_rate input
    if isinstance(error_rate, (float, int)):
        # Uniform error rate
        error_rates = [[float(error_rate)] * code_distance for _ in range(code_distance)]
    elif isinstance(error_rate, torch.Tensor):
        # 2D tensor
        if error_rate.shape != (code_distance, code_distance):
            raise ValueError(
                f"error_rate shape {error_rate.shape} must be ({code_distance}, {code_distance})"
            )
        error_rates = error_rate.tolist()
    else:
        raise TypeError(f"error_rate must be float or torch.Tensor, got {type(error_rate)}")
    
    # Add error nodes E(i,j) with their rates
    for i in range(code_distance):
        for j in range(code_distance):
            dem.add_error_node(
                f'E({i},{j})',
                error_rate=error_rates[i][j],
                update_method=update_method,
                pos=(i,j)
            )
    
    # Add bulk detectors D(i,j) on even plaquettes
    # Even plaquettes: i and j have same parity (both even or both odd)
    # Exclude boundaries: i < d-1 and j < d-1
    for i in range(code_distance - 1):
        for j in range(code_distance - 1):
            if (i + j) % 2 == 0:  # Same parity (even plaquette)
                detector_node = f'D({i},{j})'
                dem.add_detector_node(detector_node, pos=(i+0.5,j+0.5))
                # Check 4 errors: E(i,j), E(i+1,j), E(i,j+1), E(i+1,j+1)
                dem.add_detector_edge(f'E({i},{j})', detector_node)
                dem.add_detector_edge(f'E({i+1},{j})', detector_node)
                dem.add_detector_edge(f'E({i},{j+1})', detector_node)
                dem.add_detector_edge(f'E({i+1},{j+1})', detector_node)
    
    # Add boundary detectors along bottom boundary (j=-1)
    # D(i,0) for i=1,3,... (odd i) checks E(i,0), E(i+1,0)
    for i in range(1, code_distance - 1, 2):  # i=1,3,5,... (odd)
        detector_node = f'D({i},-1)'
        dem.add_detector_node(detector_node, pos=(i+0.5,-0.5))
        dem.add_detector_edge(f'E({i},0)', detector_node)
        dem.add_detector_edge(f'E({i+1},0)', detector_node)
    
    # Add boundary detectors along top boundary (j=d)
    # D(i,d-1) for i=(d-1)%2 + 0,2,... checks E(i,d-1), E(i+1,d-1)
    for i in range((code_distance - 1) % 2, code_distance - 1, 2):  # i=(d-1)%2 + 0,2,...
        detector_node = f'D({i},{code_distance})'
        dem.add_detector_node(detector_node, pos=(i+0.5,code_distance-0.5))
        dem.add_detector_edge(f'E({i},{code_distance-1})', detector_node)
        dem.add_detector_edge(f'E({i+1},{code_distance-1})', detector_node)
    
    # Add logical observable L0 that checks all errors in the last row
    dem.add_logical_node('L0', pos=(code_distance, (code_distance-1)/2))
    for j in range(code_distance):
        dem.add_logical_edge(f'E({code_distance-1},{j})', 'L0')
    
    return dem
