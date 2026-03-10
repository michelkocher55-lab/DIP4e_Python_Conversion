from typing import Any
import numpy as np
from .lambdafcns import lambdafcns
from .implfcns import implfcns
from .aggfcn import aggfcn
from .defuzzify import defuzzify


def fuzzysysfcn(inmf: Any, outmf: Any, vrange: Any, op: Any = np.fmin):
    """
    Creates a fuzzy system function.

    Parameters:
    inmf: Input MFs (list of lists).
    outmf: Output MFs (list).
    vrange: Output range.
    op: Operator (default np.fmin).

    Returns:
    F (callable): Function F(Z1, Z2...) returning scalar output.
                  Supports scalar or array inputs.
    """

    # Precompute Lambda functions (rule strengths)
    L = lambdafcns(inmf, op)

    def fuzzyOutput(*inputs: Any):
        """fuzzyOutput."""
        # Check input types.
        # If arrays, we need to loop element-wise or handle vectorized logic.
        # defuzzify integrates over v, implying scalar operations per pixel.
        # Vectorizing defuzzify logic (integration) over an array of inputs implies
        # constructing Qv for each pixel. This is heavy.

        # MATLAB loops:
        # for k = 1:numel(Z{1}) ...

        inputs_np = [np.asarray(inp) for inp in inputs]
        shape = inputs_np[0].shape

        # Flatten
        inputs_flat = [inp.ravel() for inp in inputs_np]
        N = inputs_flat[0].size
        out = np.zeros(N)

        for k in range(N):
            # Extract scalar inputs for this pixel
            current_inputs = [inp[k] for inp in inputs_flat]

            # Compute Q for this pixel
            Q = implfcns(L, outmf, *current_inputs)

            # Aggregate
            Qa = aggfcn(Q)

            # Defuzzify
            out[k] = defuzzify(Qa, vrange)

        return out.reshape(shape)

    return fuzzyOutput
