
import numpy as np

def aggfcn(Q):
    """
    Aggregation function.
    
    Parameters:
    Q (list of callables): Implication functions.
    
    Returns:
    Qa (callable): Aggregated output function Qa(v).
    """
    def aggregate(v):
        res = Q[0](v)
        for i in range(1, len(Q)):
            res = np.fmax(res, Q[i](v))
        return res
    return aggregate
