
import math
import numpy as np

def paddedsize(AB, CD=None, PARAM=None):
    """
    Computes padded sizes useful for FFT-based filtering.
    
    Parameters:
    AB (tuple/list/int): Size vector [M, N] or single dimension match.
    CD (tuple/list/int, optional): Size vector to pad relative to.
    PARAM (str, optional): 'pwr2' to pad to next power of 2.
    
    Returns:
    PQ (tuple): [P, Q]
    """
    # Handle AB input flexibility
    if np.isscalar(AB):
        AB = [AB, AB]
    else:
        AB = np.asarray(AB).flatten()
        
    # Case 1: PQ = PADDEDSIZE(AB) -> 2*AB
    if CD is None and PARAM is None:
        return (2 * AB[0], 2 * AB[1])
        
    # Case 2: PQ = PADDEDSIZE(AB, 'PWR2')
    # Check if CD is 'PWR2'
    if isinstance(CD, str): 
        if CD.upper() == 'PWR2':
             m = np.max(AB)
             P = 2**math.ceil(math.log2(2*m))
             return (P, P)
        else:
             # Unknown string param
             raise ValueError("Unknown parameter")
             
    # Case 3: PQ = PADDEDSIZE(AB, CD)
    # PQ = AB + CD - 1 -> next even integer
    if CD is not None and not isinstance(CD, str):
        if np.isscalar(CD): CD = [CD, CD]
        else: CD = np.asarray(CD).flatten()
        
        # Check for 3rd arg
        if PARAM is not None:
             if isinstance(PARAM, str) and PARAM.upper() == 'PWR2':
                 m = max(np.max(AB), np.max(CD))
                 P = 2**math.ceil(math.log2(2*m))
                 return (P, P)
        
        # Normal padding
        # PQ = AB + CD - 1
        # P = next even int >= PQ
        P = AB[0] + CD[0] - 1
        Q = AB[1] + CD[1] - 1
        
        P = 2 * math.ceil(P / 2)
        Q = 2 * math.ceil(Q / 2)
        
        return (int(P), int(Q))
        
    return (2 * AB[0], 2 * AB[1])
