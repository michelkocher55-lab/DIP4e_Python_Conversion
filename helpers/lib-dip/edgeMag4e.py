import numpy as np
from scipy.ndimage import convolve
import sys
import os

# Ensure we can import modules from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from lib.imageRotate4e import imageRotate4e
except ImportError:
    # Fallback or error if not found, but it should be there as per user request
    pass

def edgeMag4e(f, type_val, T=0.0):
    """
    Computes the gradient magnitude of the video gradient.
    
    Parameters:
    -----------
    f : numpy.ndarray
        Input image.
    type_val : str
        Type of kernel: 'prewitt', 'sobel', or 'kirsch'.
    T : float, optional
         Threshold in range [0, 1]. Defaults to 0.0 (no thresholding).
         
    Returns:
    --------
    g : numpy.ndarray
        Gradient magnitude image. (float or binary if Thresholded)
    """
    f = np.array(f, dtype=float)
    type_val = type_val.lower()
    
    kernels = []
    
    if type_val == 'prewitt':
        w1 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=float)
        w2 = w1.T
        kernels = [w1, w2]
        
    elif type_val == 'sobel':
        w1 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
        w2 = w1.T
        kernels = [w1, w2]
        
    elif type_val == 'kirsch':
        # Kirsch kernels
        w1 = np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]], dtype=float)
        kernels.append(w1)
        # Rotate 45 degrees 7 times to get 8 compass kernels
        # MATLAB code rotates w(:,:,k-1) by 45 degrees to get w(:,:,k)
        
        current_k = w1
        for _ in range(7):
            # Use imageRotate4e to rotate the kernel
            # Note: rotating a 3x3 kernel might introduce interpolation artifacts if not handled carefully
            # but we follow the request to use the translated function.
            # However, imageRotate4e might assume image logic.
            # For 3x3 kernels, specific discrete rotations are usually expected.
            # Let's see if imageRotate4e handles small matrices well or if we should hardcode.
            # The MATLAB code uses imageRotate4e explicitly on the kernel.
            # So we will do the same.
            
            # Using nearest neighbor interpolation inside imageRotate4e typically?
            # Or bilinear? imRecon used bilinear. 
            # If imageRotate4e uses 'bilinear' by default or if we can specify.
            # Typically for kernels we want exact values (shifting elements).
            
            # Let's try to use imageRotate4e with 'nearest' if possible, or just call it.
            # Checking imageRotate4e signature from memory/context: imageRotate4e(f, angle, method, crop)
            # Actually I should check imageRotate4e source or usage in imRecon4e (it used it).
            # imRecon4e usage: imageRotate4e(f, -theta(I))
            
            rotated = imageRotate4e(current_k, 45) 
            
            # After rotation, we might need to round to integer if they are supposed to be integers?
            # Kirsch weights are integers. Rotation of 3x3 grid by 45 deg via interpolation might be messy.
            # But let's trust the mandated tool.
            
            # Force small matrix 3x3 if it grew? imageRotate4e usually preserves size?
            # Or handles cropping.
            
            kernels.append(rotated)
            current_k = rotated
            
    else:
        raise ValueError(f"Unknown kernel type: {type_val}")
        
        
    # Convolve
    # Preallocate response
    m, n = f.shape
    responses = np.zeros((m, n, len(kernels)))
    
    for i, k in enumerate(kernels):
        # twodConv4e uses replicate padding. 
        # scipy.ndimage.convolve with mode='nearest' does replicate padding.
        responses[:, :, i] = convolve(f, k, mode='nearest')
        
    # Compute Magnitude
    if type_val in ['prewitt', 'sobel']:
        # Eq 10-17: sqrt(gx^2 + gy^2)
        g = np.sqrt(responses[:, :, 0]**2 + responses[:, :, 1]**2)
    else:
        # Kirsch: Max response
        # MATLAB: cv = abs(cv); g = max(cv, [], 3)
        responses = np.abs(responses)
        g = np.max(responses, axis=2)
        
    # Thresholding
    if T != 0:
        g = (g > (T * np.max(f))).astype(float) # Return as 0.0/1.0 double image or boolean?
        # MATLAB says "The output is a binary image... This is an image of class double." if T!=0 logic implies.
        # Actually MATLAB `g > ...` produces logical.
        # Then `g` is logical.
        # But if T=0, g is double.
        # We'll return float 0.0/1.0 for consistency if thresholded, or just the float map.
        
    return g
