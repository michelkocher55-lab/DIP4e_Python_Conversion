import numpy as np

def perceptronClassifier4e(X, w, r=None):
    """
    Perceptron classifier for two classes.
    
    Parameters
    ----------
    X : numpy.ndarray
        (Dim, NumPatterns). Augmented patterns (last row 1s).
    w : numpy.ndarray
        (Dim, 1) or (Dim,). Weight vector.
    r : numpy.ndarray, optional
        (1, NumPatterns) or (NumPatterns,). True classes (+1 or -1).
        
    Returns
    -------
    rout : numpy.ndarray
        Classifications (+1, -1, or 0).
    numError : int
        Number of classification errors (if r provided).
    recogRate : float
        Percent correct (if r provided).
    """
    
    # Ensure w is column vector (Dim, 1) or shape (Dim,)
    # X is (Dim, NumPatterns)
    
    # Check orientation
    # MATLAB: w = w(:)
    w = np.array(w).flatten()
    
    # X dot w
    # (Dim, N).T dot (Dim,) -> (N,)
    # or w.T dot X -> (1, N) or (N,)
    
    c = np.dot(w, X) # (N,) result if w is (Dim,)
    
    rout = np.zeros_like(c)
    rout[c > 0] = 1
    rout[c < 0] = -1
    rout[c == 0] = 0
    
    numError = None
    recogRate = None
    
    if r is not None:
        r = np.array(r).flatten()
        # Compute errors
        # sum(rout ~= r)
        incorrect = (rout != r)
        numError = np.sum(incorrect)
        np_patterns = X.shape[1]
        recogRate = ((np_patterns - numError) / np_patterns) * 100.0
        
    return rout, numError, recogRate
