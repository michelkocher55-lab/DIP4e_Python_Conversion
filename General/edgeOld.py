
import numpy as np
from skimage import feature, filters
from scipy import ndimage

def edge(I, method='sobel', threshold=None, sigma=2):
    """
    Find edges in an intensity image.
    Mimics MATLAB's edge function.
    
    Parameters:
    I : ndarray
        Input image.
    method : str
        Edge detection method: 'sobel', 'prewitt', 'roberts', 'log', 'canny'.
    threshold : float or None
        Threshold value. If None, it is determined automatically.
        If 0, it might imply return all edges or very low threshold.
    sigma : float
        Standard deviation for 'log' / 'canny'. Default is 2 for LoG, 1 for Canny in MATLAB (but we use explicit args).
        MATLAB edge(..., sigma) for Canny/LoG.
        
    Returns:
    BW : ndarray (bool)
        Binary image containing 1s where edges are found.
    """
    
    I = np.asarray(I, dtype=float)
    # Ensure I is [0, 1] or appropriate range if needed, but filters usually work on any range.
    # Normalizing usually helps with default thresholding.
    if I.max() > 1:
        I = I / I.max() # Simple normalization for thresholding logic
    
    method = method.lower()
    
    if method == 'sobel':
        # skimage.filters.sobel returns the gradient magnitude
        mag = filters.sobel(I)
        if threshold is None:
            # Automatic threshold logic (MATLAB uses heuristic logic, e.g. based on RMS noise)
            # skimage recommends filters.threshold_otsu or similar?
            # MATLAB default is often heuristic. A simple one is 4 * mean(gradient)?
            # Or use skimage's basic approach?
            # Let's use a heuristic: threshold = sqrt(4 * mean(mag^2)) approx?
            # Simple fallback: 
            threshold = 0.04 # Ad-hoc default if normalized
            # Better:
            threshold = 0.1 * mag.max()
        elif threshold == 0:
             # If 0 is passed explicitly, it might mean "very sensitive".
             # But strictly 0 threshold means everything > 0 is edge.
             pass 
             
        BW = mag > threshold
        # Thinning? MATLAB's sobel edge does thinning by default.
        # skimage sobel returns magnitude.
        # We need NMS (Non-Maximum Suppression) to get thin edges like MATLAB.
        # Converting magnitude to thin edges is not trivial without direction.
        # But Figure 10.1 likely just wants the binary mask or the magnitude?
        # MATLAB `edge` returns BINARY image with lines 1 pixel wide (thinned).
        # Implementing thinning on Sobel magnitude is hard without direction.
        # Canny includes NMS.
        # Maybe we just return the thresholded magnitude for now, or use Canny with sigma=0 (approx)?
        # Actually, for 'sobel', MATLAB does perform thinning.
        # If we can't easily do thinning, we might return thick edges.
        
    elif method == 'prewitt':
        mag = filters.prewitt(I)
        if threshold is None: threshold = 0.1 * mag.max()
        BW = mag > threshold
        
    elif method == 'roberts':
        mag = filters.roberts(I)
        if threshold is None: threshold = 0.1 * mag.max()
        BW = mag > threshold
        
    elif method == 'canny':
        # sigma defaults to 1 in MATLAB Canny
        if sigma is None: sigma = 1.0
        # low_threshold, high_threshold.
        # MATLAB takes [low, high] as threshold.
        # If scalar threshold provided, it's high, low is 0.4*high.
        low_threshold = None
        high_threshold = None
        
        if threshold is not None:
            if np.isscalar(threshold):
                if threshold == 0:
                    # Automatic
                    pass
                else:
                    high_threshold = threshold
                    low_threshold = 0.4 * threshold
            elif len(threshold) == 2:
                low_threshold, high_threshold = threshold
        
        BW = feature.canny(I, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
        
    elif method == 'log':
        # Laplacian of Gaussian
        if sigma is None: sigma = 2.0
        
        # MATLAB uses a specific LoG filter generation or internal logic.
        # gaussian_laplace from scipy calculates Laplacian of Gaussian.
        # Note: MATLAB's fspecial('log') and scipy's gaussian_laplace might have scaling differences.
        # MATLAB documentation says fspecial('log',...) returns:
        # h(x,y) = c * (x^2 + y^2 - 2*sigma^2) * exp(...)
        # We need to ensure we are consistent if we want to use the same threshold values (e.g. 0.0009).
        
        # Let's use the helper fspecial_log if we want exact matching or stick to ndimage and handle scaling.
        # To strictly replicate Figure 10.22, using the exact filter constructed in the script (or similar) is best.
        # But 'edge' function is generic. 
        # Let's use ndimage.gaussian_laplace. It computes derivative.
        
        # IMPORTANT: MATLAB's edge function for LoG automatically determines threshold if not provided.
        # If threshold is 0, it returns all zero crossings.
        
        log_img = ndimage.gaussian_laplace(I, sigma=sigma)
        
        # Zero Crossing Detection
        # A zero crossing occurs if a pixel has a different sign than its neighbor.
        # We check 4-connectivity or 8-connectivity. MATLAB usually checks horizontal and vertical.
        
        # 1. Sign change Horizontal
        # 2. Sign change Vertical
        
        # We also need to check the "strength" of the crossing if threshold > 0.
        # Strength = |val - neighbor| ??? 
        # MATLAB: "The sensitivity threshold... ignores all edges that are not stronger than thresh."
        # For LoG, this usually means the absolute difference of the LoG values across the zero crossing.
        
        # Create zero crossing mask
        rows, cols = I.shape
        BW = np.zeros((rows, cols), dtype=bool)
        
        # Threshold logic
        if threshold is None:
            # Heuristic default approx 0.75 * mean_abs_dev? 
            # Or similar to MATLAB.
            threshold = 0.75 * np.mean(np.abs(log_img)) 
        
        # Vectorized Zero Crossing with Threshold
        
        # Horizontal (compare [:, :-1] and [:, 1:])
        diff_h = log_img[:, :-1] - log_img[:, 1:]
        # Check signs oppposite
        sign_change_h = (log_img[:, :-1] * log_img[:, 1:]) < 0
        # Check strength
        strong_h = np.abs(diff_h) > threshold
        
        # Mark the 'positive' side or both? 
        # MATLAB edge usually marks single pixels. 
        # Typically marks the pixel closer to zero, or simply the one on the left/top.
        BW[:, :-1] |= (sign_change_h & strong_h)
        
        # Vertical (compare [:-1, :] and [1:, :])
        diff_v = log_img[:-1, :] - log_img[1:, :]
        sign_change_v = (log_img[:-1, :] * log_img[1:, :]) < 0
        strong_v = np.abs(diff_v) > threshold
        
        BW[:-1, :] |= (sign_change_v & strong_v)
        
        # Note generally "closed loops" are formed. 
        # The simple shift approach works well for regular grids.
        
        return BW
        
    else:
        raise ValueError(f"Unknown method details: {method}")
        
    return BW
