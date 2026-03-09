from typing import Any
import numpy as np


def defuzzify(Qa: Any, vrange: Any):
    """
    Defuzzify using center-of-gravity.

    Parameters:
    Qa (callable): Aggregation function Qa(v).
    vrange (tuple/list): [min, max] of output range.

    Returns:
    out (float): Defuzzified output.
    """
    v1, v2 = vrange
    # Integration grid
    v = np.linspace(v1, v2, 100)
    Qv = Qa(v)

    sum_Qv = np.sum(Qv)

    if sum_Qv == 0:
        return np.mean(vrange)

    out = np.sum(v * Qv) / sum_Qv
    return out
