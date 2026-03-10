from typing import Any
import numpy as np
import pickle
from .bellmf import bellmf
from .triangmf import triangmf
from .fuzzysysfcn import fuzzysysfcn
from .approxfcn import approxfcn


def makefuzzyedgesys(save_path: Any = "fuzzyedgesys.pkl"):
    """
    Creates the fuzzy edge detection system G.

    Returns:
    G (callable): The fuzzy system approximation.
    """

    # Input MFs
    # zero = @(z) bellmf(z, -0.3, 0) -- Wait, bellmf(z,a,b).
    # MATLAB: bellmf(z, [a, b, c]) usually?
    # My bellmf.py takes(z, a, b).
    # In 'makefuzzyedgesys.m' line 14: zero = @(z) bellmf(z, -0.3, 0);
    # This implies a=-0.3, b=0?
    # BUT wait. bellmf usually defines width.
    # Let's check my bellmf.py implementation again.
    # It delegates to smf.
    # smf(z, a, b).
    # bellmf use:
    # if z < b: smf(z, a, b)
    # if z >= b: smf(2*b - z, a, b)
    # So 'b' is the center/peak?
    # And 'a' is the foot?
    # MATLAB code: bellmf(z, -0.3, 0).
    # a=-0.3, b=0.
    # z < 0: smf(z, -0.3, 0).
    # smf rise from -0.3 to 0.
    # z=0 -> 1.
    # Symmetry around 0.

    def zero(z: Any):
        """zero."""
        return bellmf(z, -0.3, 0)

    # not_used = @(z) onemf(z); -> Always 1?
    # I don't have onemf. I'll implement a lambda.
    def not_used(z: Any):
        """not_used."""
        return np.ones_like(z, dtype=float)

    # Output MFs
    # black = @(z) triangmf(z,0,0,0.75);
    def black(z: Any):
        """black."""
        return triangmf(z, 0, 0, 0.75)

    # white = @(z) triangmf(z,0.25,1,1);
    def white(z: Any):
        """white."""
        return triangmf(z, 0.25, 1, 1)

    # Rules
    # inmf matrix 4x4
    inmf = [
        [zero, not_used, zero, not_used],
        [not_used, not_used, zero, zero],
        [not_used, zero, not_used, zero],
        [zero, zero, not_used, not_used],
    ]

    # Outmf
    # {white, white, white, white, black}
    outmf = [white, white, white, white, black]

    vrange = [0, 1]

    # Create F
    print("Creating Fuzzy System F...")
    F = fuzzysysfcn(inmf, outmf, vrange)

    # Approx F
    print("Approximating F as G (this may take a moment)...")
    # range [-1, 1] for 4 inputs
    ranges = np.array([[-1, 1], [-1, 1], [-1, 1], [-1, 1]])
    G = approxfcn(F, ranges)

    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(G, f)
        print(f"Saved system to {save_path}")

    return G


if __name__ == "__main__":
    makefuzzyedgesys()
