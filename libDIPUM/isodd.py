
import numpy as np

def isodd(A):
    """
    Determines which elements of an array are odd numbers.
    
    Parameters:
    A (ndarray or scalar): Input numbers.
    
    Returns:
    D (ndarray or bool): True where elements are odd.
    """
    A = np.asarray(A)
    # MATLAB: D = 2*floor(A/2) ~= A
    # Python: A % 2 != 0 logic works for integers.
    # But if A contains floats like 1.2?
    # MATLAB: floor(1.2/2) = 0. 2*0 = 0. 0 != 1.2 -> True.
    # MATLAB: floor(1.0/2) = 0. 2*0 = 0. 0 != 1.0 -> True.
    # WAIT! 
    # isodd(1) -> 2*floor(0.5) = 0. 0 != 1 -> True. Correct.
    # isodd(1.2) -> 2*floor(0.6) = 0. 0 != 1.2 -> True. Odd?
    # isodd(2) -> 2*1 = 2. 2 != 2 -> False. Even.
    # isodd(2.5) -> 2*floor(1.25)=2. 2 != 2.5 -> True. Odd?
    
    # Let's replicate exact formula logic to be safe for non-integers if used.
    # D = 2*floor(A/2) != A
    return 2 * np.floor(A / 2) != A
