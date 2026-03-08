
import numpy as np
import pickle
import os
from scipy.ndimage import convolve

def fuzzyfilt(f):
    """
    Fuzzy edge detector.
    
    Parameters:
    f (ndarray): Input image.
    
    Returns:
    g (ndarray): Filtered image.
    """
    f = f.astype(float)
    if f.max() > 1.0:
        f = f / 255.0
        
    # Load fuzzy system
    # Assuming fuzzyedgesys.pkl is in current dir or libDIPUM
    sys_path = 'fuzzyedgesys.pkl'
    # Check if exists, else look in libDIPUM
    if not os.path.exists(sys_path):
        import sys
        # Try to find it in the module path if possible, or just assume local run
        # For now, assume it was generated in the current working directory or lib
        pass
        
    try:
        with open(sys_path, 'rb') as fd:
            G = pickle.load(fd)
    except FileNotFoundError:
        print(f"Error: {sys_path} not found. Run makefuzzyedgesys.py first.")
        return np.zeros_like(f)
        
    # Compute differences
    # MATLAB: z1 = imfilter(f,[0 -1 1],'conv','replicate');
    # MATLAB 'conv' flips kernel. [0 -1 1] -> [1 -1 0].
    # Center is -1.
    # Output at x: 1*f(x-1) - 1*f(x). (Left - Center).
    
    # Scipy convolve flips kernel. So using [0, -1, 1] gives same result.
    k1 = np.array([[0, -1, 1]])
    z1 = convolve(f, k1, mode='nearest')
    
    # z2 = imfilter(f,[0; -1; 1],'conv','replicate');
    # Vertical column vector.
    k2 = np.array([[0], [-1], [1]])
    z2 = convolve(f, k2, mode='nearest')
    
    # z3 = imfilter(f,[1; -1; 0],'conv','replicate');
    k3 = np.array([[1], [-1], [0]])
    z3 = convolve(f, k3, mode='nearest')
    
    # z4 = imfilter(f,[1 -1 0],'conv','replicate');
    k4 = np.array([[1, -1, 0]])
    z4 = convolve(f, k4, mode='nearest')
    
    # Apply G
    print("Applying fuzzy system to pixels...")
    # G expects arrays z1, z2, z3, z4
    # G returns array of same shape
    g = G(z1, z2, z3, z4)
    
    return g
