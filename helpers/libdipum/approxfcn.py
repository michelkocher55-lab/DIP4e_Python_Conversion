from typing import Any
import numpy as np
from scipy.interpolate import RegularGridInterpolator
# from tqdm import tqdm


class FuzzyApproximator:
    def __init__(self, interpolator: Any, grid: Any, num_inputs: Any):
        """__init__."""
        self.interpolator = interpolator
        self.grid = grid  # Store just in case, though interpolator has it
        self.num_inputs = num_inputs

    def __call__(self, *inputs: Any):
        """__call__."""
        # inputs: Z1, Z2... arrays

        # Broadcast inputs against each other
        try:
            inputs_b = np.broadcast_arrays(*inputs)
        except ValueError:
            # Fallback if inputs likely scalars but passed strangely
            inputs_b = [np.asarray(i) for i in inputs]

        shape = inputs_b[0].shape

        # Stack: (TotalPoints, num_inputs)
        # Flatten and stack
        points = np.stack([inp.ravel() for inp in inputs_b], axis=-1)

        res = self.interpolator(points)
        return res.reshape(shape)


def approxfcn(F: Any, range_val: Any):
    """
    Approximation function using lookup table.
    Returns a picklable FuzzyApproximator object.
    """
    range_val = np.asarray(range_val)
    num_inputs = range_val.shape[0]

    max_table_elements = 10000
    max_table_dim = 100

    # Calculate dimension size per input
    table_dim = min(
        int(np.floor(max_table_elements ** (1 / num_inputs))), max_table_dim
    )

    # Create grids
    grids = []
    for k in range(num_inputs):
        grids.append(np.linspace(range_val[k, 0], range_val[k, 1], table_dim))

    # Meshgrid for evaluation
    # Use meshgrid (dense)
    mesh_grids = np.meshgrid(*grids, indexing="ij")

    # Flatten for F evaluation
    flat_inputs = [g.flatten() for g in mesh_grids]

    # Evaluate F
    print(f"Evaluating fuzzy system on {table_dim}^{num_inputs} grid...")
    table_flat = F(*flat_inputs)
    table = table_flat.reshape([table_dim] * num_inputs)

    # Create interpolator
    interp = RegularGridInterpolator(
        tuple(grids), table, bounds_error=False, fill_value=None
    )

    return FuzzyApproximator(interp, grids, num_inputs)
