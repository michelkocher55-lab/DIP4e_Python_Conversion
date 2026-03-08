
import numpy as np

def otsuthresh(h):
    """
    Computes Otsu's optimum threshold from a histogram.
    
    Parameters:
    h (array-like): Histogram.
    
    Returns:
    T (float): Optimum threshold in range [0, 1].
    SM (float): Separability measure in range [0, 1].
    """
    h = np.array(h).flatten()
    
    # Normalize histogram to unit area
    if np.sum(h) > 0:
        h = h / np.sum(h)
        
    # All possible intensities represented in histogram (1-based index emulation or 0-based?)
    # MATLAB: c = (1:numel(h))' -> 1, 2, ... N
    # We will use 0-based indices 0, 1, ... N-1 for calculations, but replicate math.
    
    num_bins = len(h)
    c = np.arange(num_bins) + 1 # Use 1-based to match MATLAB intermediate math if needed? 
    # Actually, let's stick to the logic:
    # m = cumsum(c .* h)
    # If we use 0-based, T will be index.
    # MATLAB returns T in range [0, 1].
    # T = (T - 1) / (numel(h) - 1). (Where T was 1-based index).
    # If we use 0-based index k: T_norm = k / (num_bins - 1).
    
    # Let's use 1-based 'c' to match MATLAB steps exactly, then convert.
    
    # Cumulative sums P1
    P1 = np.cumsum(h)
    
    # Cumulative means m
    m = np.cumsum(c * h)
    
    # Global mean mG
    mG = m[-1]
    
    # Between-class variance
    # sigSquared = ((mG*P1 - m).^2)./(P1.*(1 - P1) + eps);
    eps = np.finfo(float).eps
    sigSquared = ((mG * P1 - m)**2) / (P1 * (1 - P1) + eps)
    
    # Find maximum of sigSquared
    maxSigsq = np.max(sigSquared)
    
    # Indices where max occurs
    # MATLAB: find(sigSquared == maxSigsq) -> 1-based indices
    indices = np.where(sigSquared == maxSigsq)[0] + 1
    
    # Average them
    T_index = np.mean(indices)
    
    # Normalize T to [0, 1]
    # T = (T - 1)/(numel(h) - 1);
    T = (T_index - 1) / (num_bins - 1)
    
    # Separability measure
    # SM = maxSigsq/(sum(((c - mG).^2).*h) + eps);
    # Total variance: sum((c - mG)^2 * h)
    variance_total = np.sum(((c - mG)**2) * h)
    SM = maxSigsq / (variance_total + eps)
    
    return T, SM
