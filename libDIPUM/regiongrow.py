
import numpy as np
from skimage.morphology import reconstruction
from skimage.measure import label, regionprops

def regiongrow(f, S, T):
    """
    Image segmentation using region growing.
    
    Parameters:
        f: Input image (float or uint8).
        S: Seed array (same size as f) with 1s at seed locations, OR a scalar seed value.
        T: Threshold (scalar or array).
        
    Returns:
        g: Segmented image (labeled).
        NR: Number of regions.
        SI: Final seed image.
        TI: Thresholded image before connectivity.
    """
    f = f.astype(float)
    
    # Check if S is scalar
    if np.isscalar(S):
        SI = (f == S)
        S1 = np.array([S])
    else:
        # S is a matrix.
        # "Eliminate duplicate, connected seed locations".
        # We simulate bwmorph('shrink', Inf) by reducing each component to a single point.
        # Actually, if S is passed from Figure1046, it might already be shrunken.
        # But regiongrow logic implies it handles it.
        # Efficient way to reduce to single points:
        # Label S, then take one coord per label.
        
        S_bool = S > 0
        if not np.any(S_bool):
             # No seeds
             return np.zeros_like(f), 0, S_bool, np.zeros_like(f, dtype=bool)
             
        labeled_S = label(S_bool)
        props = regionprops(labeled_S)
        
        SI = np.zeros_like(S_bool)
        S1_values = []
        
        for p in props:
            # Pick the first coordinate (centroid might be float)
            # coords returns (row, col)
            r, c = p.coords[0]
            SI[r, c] = True
            S1_values.append(f[r, c])
            
        S1 = np.array(S1_values)

    TI = np.zeros_like(f, dtype=bool)
    
    # Loop over seed values
    # "S = abs(f - seedvalue) <= T"
    # To optimize: if many seeds have same value, we can group them.
    # unique_seeds = np.unique(S1)
    # But strictly following MATLAB logic:
    
    # Optimization: iterate over unique values to avoid redundant TI updates
    unique_seeds = np.unique(S1)
    
    for val in unique_seeds:
         mask = np.abs(f - val) <= T
         TI |= mask
         
    # Reconstruct
    # imreconstruct(SI, TI)
    # SI is marker, TI is mask.
    # Note: skimage reconstruction is (seed, mask).
    
    # "Use function imreconstruct with SI as the marker image to obtain the regions... in TI"
    # Wait. imreconstruct(marker, mask).
    # SI (seeds) is marker. TI (thresholded) is mask.
    # Reconstruction dilates SI constrained by TI.
    
    recon = reconstruction(SI, TI, method='dilation')
    
    # Label geometry
    # bwlabel -> label
    # In MATLAB bwlabel works on binary. recon is binary (float 0/1 or bool).
    g, NR = label(recon > 0, return_num=True)
    
    return g, NR, SI, TI
