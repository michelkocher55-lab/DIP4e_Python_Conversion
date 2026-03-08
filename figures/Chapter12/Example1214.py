
import numpy as np
import matplotlib.pyplot as plt
import sys
from lib.principalComponents4e import principalComponents4e

def get_choice():
    print("Example from the book (1), Larger example (2) : ")
    try:
        val = input()
        return int(val)
    except:
        return 1

def eigsort(C):
    # Sort eigenvalues/vectors descending
    vals, vecs = np.linalg.eig(C)
    idx = np.argsort(vals)[::-1]
    D = np.diag(vals[idx])
    V = vecs[:, idx]
    return V, D

def Example1114(Choix=None):
    if Choix is None:
        Choix = get_choice()
        
    print(f"Running Example1114 (Choix={Choix})...")

    # Data
    if Choix == 1:
        N = 4
        X = np.array([
            [0, 1, 1, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)
        
    elif Choix == 2:
        N = 200
        t = np.linspace(0, 1, N)
        f0 = 3
        Sigma = 0.4
        f1 = np.sin(2 * np.pi * f0 * t)
        f2 = np.cos(2 * np.pi * f0 * t)
        
        # Noise
        # randn(size(t))
        r1 = Sigma * np.random.randn(N)
        r2 = Sigma * np.random.randn(N)
        r3 = Sigma * np.random.randn(N)
        
        X = np.vstack([
            f1 + r1,
            f2 + r2,
            f1 + f2 + r3
        ])
        
    else:
        print("Invalid choice.")
        return

    # Mean and Covariance (Normalized by N)
    # X rows are variables.
    mx = np.mean(X, axis=1).reshape(-1, 1)
    
    # Cov
    # np.cov default is ddof=1 (N-1).
    # We need ddof=0 (N).
    CX = np.cov(X, ddof=0)
    
    print("Mean mx:")
    print(mx)
    print("Covariance CX (Norm by N):")
    print(CX)
    
    # Eigenvalues
    V, D = eigsort(CX)
    print("Eigenvalues D:")
    print(np.diag(D))
    
    # Hotelling Transform
    A = V.T # Eigenvectors in rows
    
    # Center X
    # X - mx
    X_centered = X - mx 
    
    # Y
    Y = A @ X_centered
    
    if Choix == 1:
        print("Y:")
        print(Y)
        
    my = np.mean(Y, axis=1).reshape(-1, 1)
    CY = np.cov(Y, ddof=0)
    print("Mean Y:")
    print(my) # Should be 0
    print("Cov Y:")
    print(CY) # Should be diagonal D
    
    # Complete Reconstruction
    # XHat = A' * Y + mx
    XHatComplete = A.T @ Y + mx
    EComplete = X - XHatComplete
    
    if Choix == 1:
        print("XHatComplete:")
        print(XHatComplete)
        print("EComplete:")
        print(EComplete)
        
    MSEComplete = np.mean(EComplete**2, axis=1) # Mean over samples (columns)
    MSECompleteSum = np.sum(MSEComplete)
    print(f"MSECompleteSum: {MSECompleteSum}")
    
    # Partial Reconstruction (Keep 2 components)
    A2 = A[:2, :] # First 2 rows
    Y2 = A2 @ X_centered
    
    XHat2 = A2.T @ Y2 + mx
    E2 = X - XHat2
    
    if Choix == 1:
        print("Y2:")
        print(Y2)
        print("XHat2:")
        print(XHat2)
        print("E2:")
        print(E2)
        
    mE = np.mean(E2, axis=1)
    CE = np.cov(E2, ddof=0)
    MSE2 = np.mean(E2**2, axis=1)
    MSE2Sum = np.sum(MSE2)
    print(f"MSE2Sum: {MSE2Sum}")
    
    # Use of principalComponents (Norm by N-1)
    # X' is passed. Rows=Samples.
    print("Running principalComponents4e (Norm by N-1)...")
    try:
        P = principalComponents4e(X.T, 2)
        print(f"Result type: {type(P)}")
        print(f"Result keys/dir: {dir(P)}")
        if hasattr(P, 'ems'):
             print("P.ems:", P.ems)
        else:
             print("Error: P object has no 'ems' attribute.")
             if isinstance(P, dict):
                 print("P is a dict. Keys:", P.keys())
    except Exception as e:
        print(f"Error calling principalComponents4e: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Display (Choix 2)
    if Choix == 2:
        fig = plt.figure(figsize=(10, 8))
        
        # 1. X'
        plt.subplot(2, 2, 1)
        plt.plot(X.T)
        plt.title(f"X, lambda={np.diag(D)}")
        plt.grid(True)
        # Store axis limits?
        
        # 2. Y2'
        plt.subplot(2, 2, 2)
        plt.plot(Y2.T)
        plt.title("Y2")
        plt.grid(True)
        
        # 3. Xrec'
        plt.subplot(2, 2, 3)
        plt.plot(XHat2.T)
        plt.title("X_rec")
        plt.grid(True)
        
        # 4. Error
        plt.subplot(2, 2, 4)
        plt.plot(E2.T)
        plt.title(f"Err, MSE={MSE2}, Sum={MSE2Sum:.4f}")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"Example1114_Choix{Choix}.png")
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            Example1114(int(sys.argv[1]))
        except:
            Example1114()
    else:
        Example1114()
