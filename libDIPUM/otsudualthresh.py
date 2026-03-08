
import numpy as np

def otsudualthresh(h):
    """
    Computes two optimum thresholds using Otsu's method.
    
    Parameters:
    h: histogram (1D array). Must be normalized to unit area (sum(h) == 1).
    
    Returns:
    T1: First threshold (normalized [0, 1]).
    T2: Second threshold (normalized [0, 1]).
    S: Separability measure.
    """
    h = np.array(h).flatten()
    
    # Check normalization with some tolerance
    if not np.isclose(np.sum(h), 1.0, atol=1e-5):
         # Matlab throws error. We can try to normalize or raise error.
         # "The input histogram must be normalized to unit area"
         raise ValueError("The input histogram must be normalized to unit area")
         
    L = len(h)
    
    # 0-based indices 0 to L-1
    # Matlab: i = (1:numel(h))' -> 1..L
    # We will use 0..L-1 and shift logic accordingly for T calculation in range [0, 1].
    # But for calculation of mean 'm', we should probably consistent with intensities 1..L or 0..L-1?
    # Matlab code: m1 = sum(i(1:k1).*h(1:k1))...
    # If we use 0-based `i` array:
    range_i = np.arange(L) # 0, 1 ... L-1.
    
    # To match MATLAB exactly, we might want to use 1..L range for 'i' and then normalize result?
    # MATLAB: T1 = (T1 - 1)/(L - 1).
    # If we use 0..L-1 range for i:
    # T1_index comes from range 0..L-1.
    # T1_norm = T1_index / (L - 1).
    # Let's verify M first.
    # M = sum(i * h).
    # If i starts at 0, M is mean intensity in 0..L-1.
    # If i starts at 1, M is mean in 1..L.
    # The term (m1 - M)^2 would be same relative difference?
    # P1 * (m1 - M)^2.
    # Let m1_0 be mean with 0-based. m1_1 be mean with 1-based.
    # m1_1 = m1_0 + 1.
    # M_1 = M_0 + 1.
    # (m1_1 - M_1) = (m1_0 + 1 - (M_0 + 1)) = m1_0 - M_0.
    # So the VARIANCE calculation is INDEPENDENT of the shift.
    # So `maxsigsq` will be the same.
    # The indices k1, k2 will be the same (relative to array start).
    # So we can use 0-based `i` safely.
    
    i_vals = np.arange(L)
    M = np.sum(i_vals * h)
    
    # Pre-compute cumulative sums for speed
    P_cum = np.cumsum(h)
    m_cum = np.cumsum(i_vals * h)
    
    # Helper to get P and m for a range (inclusive of indices)
    # Range [0, k1] -> P1
    # Range [k1+1, k2] -> P2
    # Range [k2+1, L-1] -> P3
    
    # Indices k1, k2.
    # MATLAB loop:
    # k1 = 1 to L-3. (1-based index of end of first interval)
    # k2 = k1+1 to L-2. (1-based index of end of second interval)
    
    # Python indices:
    # k1 from 0 to L-4.
    # k2 from k1+1 to L-3.
    
    # Initialize
    maxsigsq = -1.0
    best_k1 = []
    best_k2 = []
    
    eps = np.finfo(float).eps
    
    # We can use nested loops or vectorization.
    # L is usually 256. Loops are 256*256/2 = 32k. Very fast in Python too.
    # Let's use loops for clarity and direct translation.
    
    # Pre-allocating/using variables
    # We need to iterate valid split points.
    
    # In MATLAB: k1 is the last index of region 1.
    # Region 1: 1 .. k1.
    # Region 2: k1+1 .. k2.
    # Region 3: k2+1 .. L.
    
    # Python corresponding indices (0-based):
    # Region 1: 0 .. k1. (So k1 is index inclusive).
    # Region 2: k1+1 .. k2.
    # Region 3: k2+1 .. L-1.
    
    # Constraints:
    # Region 1 must have at least 1 bin? usually.
    # Region 2 must have at least 1 bin.
    # Region 3 must have at least 1 bin.
    # MATLAB loop `k1 = 1:L-3` ensures:
    # if k1=1: region 1 is index 1.
    # max k2=L-2. Region 2 is 2..L-2. Region 3 is L-1..L (at least 2 bins? No L-1 is index L-1.
    # Wait, 1..L. L-2 is the index.
    # Next starts L-1. End is L. So Region 3 is Indices L-1, L. (2 bins).
    # If loop goes to L-3. Then max k2 is L-2.
    # Why L-3?
    # If k1 = L-3. k2 must start at L-2.
    # If k2 ends at L-2. Region 3 starts at L-1.
    # Seems to guarantee at least 1 or 2 bins.
    
    # Python loops:
    # k1 from 0 to L-4. (inclusive)
    # k2 from k1+1 to L-3. (inclusive)
    
    # Let's just collect all sig_squared values and find max at the end, like MATLAB code
    # sig_squared = np.zeros((L, L)) # Wasteful but matches logic
    # Actually we only care about max.
    
    # Iterating
    for k1 in range(0, L-3): # 0 to L-4 (range excludes end)
        # P1 = sum(h[0:k1+1]) -> P_cum[k1]
        P1 = P_cum[k1]
        m1 = m_cum[k1] / (P1 + eps)
        
        for k2 in range(k1+1, L-2): # k1+1 to L-3
            # P2 = sum(h[k1+1 : k2+1])
            P2 = P_cum[k2] - P_cum[k1]
            m2 = (m_cum[k2] - m_cum[k1]) / (P2 + eps)
            
            # P3 = sum(h[k2+1 : end])
            P3 = P_cum[-1] - P_cum[k2]
            m3 = (m_cum[-1] - m_cum[k2]) / (P3 + eps)
            
            curr_sig_sq = P1*((m1 - M)**2) + P2*((m2 - M)**2) + P3*((m3 - M)**2)
            
            if curr_sig_sq > maxsigsq:
                maxsigsq = curr_sig_sq
                best_k1 = [k1]
                best_k2 = [k2]
            elif np.isclose(curr_sig_sq, maxsigsq):
                best_k1.append(k1)
                best_k2.append(k2)
                
    # Average the best indices
    # MATLAB: T1 = sum(I)/numel(I). I are indices (1-based).
    # best_k1 are 0-based indices.
    # T1_index = mean(best_k1).
    if len(best_k1) == 0:
        # Should not happen
        return 0, 0, 0
        
    T1_idx = np.mean(best_k1)
    T2_idx = np.mean(best_k2)
    
    # Normalize to [0, 1]
    # MATLAB: T1 = (T1 - 1)/(numel(h) - 1); where T1 was 1-based.
    # Here T1_idx is 0-based.
    # So T1 = T1_idx / (L - 1).
    
    T1 = T1_idx / (L - 1)
    T2 = T2_idx / (L - 1)
    
    # Separability
    # S = maxsigsq/(sum(((i - M).^2).*h) + eps);
    variance_total = np.sum(((i_vals - M)**2) * h)
    S = maxsigsq / (variance_total + eps)
    
    return T1, T2, S
