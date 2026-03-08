
import numpy as np

def twomodegauss(m1, sig1, m2, sig2, A1, A2, k):
    """
    Generates a two-mode Gaussian function.
    
    Parameters:
    m1, sig1: Mean and std dev of first mode.
    m2, sig2: Mean and std dev of second mode.
    A1, A2: Amplitudes of the modes.
    k: Offset ("floor").
    
    Returns:
    p (ndarray): 256-element vector normalized so sum(p) = 1.
    """
    
    # Constants
    # MATLAB: c1 = A1 * (1 / ((2 * pi) ^ 0.5) * sig1);
    # This formula looks like A1 * (1 / (sqrt(2pi) * sig1))? 
    # Or A1 * (1/sqrt(2pi)) * sig1? 
    # Brackets in MATLAB: (1 / ((2 * pi) ^ 0.5) * sig1). 
    # ((2*pi)^0.5) is denom. Then * sig1.
    # So (1/sqrt(2pi)) * sig1. 
    # Note: Standard Gaussian PDF amplitude is 1/(sig * sqrt(2pi)).
    # If this is trying to match specific code:
    # A1 * (1/sqrt(2pi) * sig1) would mean sig1 is in numerator?
    # Let's verify standard PDF vs this.
    # If A1 is meant to be peak value relative or area?
    
    # Re-reading MATLAB code line 31: 
    # c1 = A1 * (1 / ((2 * pi) ^ 0.5) * sig1);
    # Operations: A1 * ( ... ). Inside: 1 / denom * sig1.
    # Division and multiplication have same precedence, evaluated left to right.
    # (1 / ((2*pi)^0.5)) * sig1. 
    
    c1 = A1 * (1 / ((2 * np.pi) ** 0.5) * sig1)
    k1 = 2 * (sig1 ** 2)
    
    c2 = A2 * (1 / ((2 * np.pi) ** 0.5) * sig2)
    k2 = 2 * (sig2 ** 2)
    
    z = np.linspace(0, 1, 256)
    
    p = k + c1 * np.exp(-((z - m1) ** 2) / k1) + \
        c2 * np.exp(-((z - m2) ** 2) / k2)
        
    p = p / np.sum(p)
    
    return p
