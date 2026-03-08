import numpy as np
import pywt
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.fft import fft, fftshift

def haarTimeFrequency(N):
    """
    Create time-frequency plot for the Haar wavelet of dimension N.
    Generates Heisenberg boxes.
    
    Parameters:
    -----------
    N : int
        Dimension (must be power of 2).
    """
    # Check power of 2
    if not np.log2(N).is_integer():
        print("N must be a power of 2")
        return

    LN = int(np.log2(N))
    
    # Create identity matrix
    I = np.eye(N)
    
    # Compute Haar Transform Matrix X
    # Each column i of X is the Haar transform of the i-th identity vector.
    # MATLAB: wavedec returns [cAn, cDn, cDn-1, ..., cD1] concatenated.
    # pywt.wavedec with level=LN returns [cAn, cDn, ..., cD1].
    # But max level for N length is log2(N).
    # pywt returns lists of coeffs.
    
    X = np.zeros((N, N))
    
    for i in range(N):
        # Using mode='periodization' matches MATLAB's 'per'
        coeffs = pywt.wavedec(I[:, i], 'haar', mode='periodization', level=LN)
        # Concatenate coeffs to get a single vector
        # Note: pywt returns [cA, cD_level, cD_level-1, ... cD_1]
        # Checking MATLAB wavedec order: [cAn, cDn, cDn-1, ..., cD1].
        # It seems consistent.
        X[:, i] = np.concatenate(coeffs)
        
    # X rows are the basis functions in time domain.
    # Why?
    # Let w = X @ v. w is coeffs, v is time signal.
    # w_i = (Row_i of X) dot v.
    # If v is an impulse at k, w_i = X_ik.
    # The code analyzes rows of X: X(i, :). This is the i-th coefficient's sensitivity map.
    # i.e. The Basis Function corresponding to coefficient i.
    
    # FFT of Basis
    # MATLAB: F = fft(transpose(X)); F = transpose(F);
    # In Python, fft(X, axis=?)
    # transpose(X) has columns as basis functions.
    # fft(..., axis=0) computes fft down columns.
    # So F_cols = fft(X.T, axis=0).
    # F = F_cols.T -> Rows are FFTs.
    # Equivalent to fft(X, axis=1) if X rows are basis functions?
    # MATLAB fft(MATRIX) works on columns.
    # transpose(X) -> columns are basis functions.
    # fft(transpose(X)) -> transform of each basis function.
    # transpose(ans) -> rows are transforms.
    # So yes, we want FFT of each ROW of X.
    # numpy fft is last axis by default. So fft(X) works on rows.
    
    F = fft(X, axis=1)
    
    # Center each row
    for i in range(N):
        F[i, :] = np.abs(fftshift(F[i, :]))
        
    t = np.arange(1, N + 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw bounding box
    ax.add_patch(Rectangle((0, 0), N, 1, fill=False, edgecolor='k'))
    
    for i in range(N):
        # Time Domain stats
        f2 = np.abs(X[i, :])**2
        E_t = np.sum(f2)
        if E_t == 0: continue
            
        tm = np.sum(t * f2) / E_t
        ts = np.sqrt(np.sum(((t - tm)**2) * f2) / E_t)
        
        # Frequency Domain stats
        # MATLAB: FROW = F(i, N/2 + 1 : N)
        # N/2+1 in 1-based is index N/2 in 0-based.
        # Range N/2 to N-1 (inclusive of N-1?). MATLAB 1:N.
        # indices: N/2, N/2+1, ... N-1.
        
        # Frequencies: W = (0 : N/2) / N
        # 0 to 0.5.
        
        # Picking Highest Freq: FROW(1 + N/2) = F(i, 1). 
        # MATLAB F(i,1) is DC (shifted?). 
        # Wait, F was fftshifted.
        # If N=8. Shifted: [-4, -3, -2, -1, 0, 1, 2, 3].
        # Indices: 0..7.
        # Center (DC) is at index N/2 = 4. (0,1,2,3, 4).
        # Positive freqs are 4, 5, 6, 7. (0, 1, 2, 3).
        # MATLAB code logic:
        # F(i, N/2+1 : N). With 1-based.
        # Indices N/2..N-1 in 0-based.
        # These are the positive frequencies [0, 1, 2, 3...]
        # Then it does something weird: FROW(1 + N/2) = F(i, 1).
        # Appends/Overwrites?
        # FROW is length N/2. (indices 1 to N/2).
        # FROW(end+1)? No, 1+N/2 is index OUTSIDE FROW size if FROW is size N/2.
        # MATLAB: FROW = F(i, N/2 + 1 : N). length is N - (N/2 + 1) + 1 = N/2.
        # Indices of FROW are 1..N/2.
        # FROW(1 + N/2) = ... writes to index N/2 + 1. Expands array by 1.
        # So it takes the positive half, and appends F(i,1) (which is -Nyquist typically in shifted?).
        # In shifted: index 0 is -Nyquist.
        # So it grabs 0..PositiveMax, and adds -Nyquist magnitude to end (treated as +Nyquist?).
        # Yes, magnitude is symmetric for real signals.
        
        start_idx = N // 2
        # F is shifted. Center is at start_idx.
        # Positive side: start_idx to end.
        
        F_row_part = F[i, start_idx:].copy() # Includes DC at 0 relative
        
        # Append F[i, 0] (which corresponds to -0.5, same mag as +0.5)
        val_at_neg_nyquist = F[i, 0]
        F_row_part = np.append(F_row_part, val_at_neg_nyquist)
        
        W = np.arange(len(F_row_part)) / N # 0/N, 1/N, ... (N/2)/N
        
        F2 = np.abs(F_row_part)**2
        E_f = np.sum(F2)
        
        if E_f == 0:
            fm, fs = 0, 0
        else:
            fm = np.sum(W * F2) / E_f
            fs = np.sqrt(np.sum(((W - fm)**2) * F2) / E_f)
            
        # Draw Box
        # Rectangle('Position', [tm - ts, fm - fs, 2*ts, 2*fs])
        # fm is normalized freq [0, 0.5].
        # tm is time samples [1, N].
        
        color = 'b' # Default blue
        # Maybe varying colors?
        rect = Rectangle((tm - ts, fm - fs), 2*ts, 2*fs, 
                         linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.plot(tm, fm, 'k.', markersize=2) # Centroid
        
    ax.set_xlim(0, N+1)
    ax.set_ylim(0, 0.6) # Freqs are up to 0.5
    ax.set_title(f'Haar Time-Frequency Plane (N={N})')
    ax.set_xlabel('Time')
    ax.set_ylabel('Normalized Frequency')
    plt.grid(True, alpha=0.3)
    
    print("Displaying Time-Frequency Plot. Close to continue.")
    plt.show()

# If run directly
if __name__ == "__main__":
    haarTimeFrequency(32)
