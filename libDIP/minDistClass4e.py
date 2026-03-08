import numpy as np

def minDistClass4e(X, mode, M=None, R=None):
    """
    Minimum distance classifier.
    
    Parameters:
    -----------
    X : numpy.ndarray
        n-by-np matrix whose columns are pattern vectors.
        n is dimensionality, np is number of vectors.
    mode : str
        'train' or 'classify'.
    M : numpy.ndarray, optional
        n-by-nc matrix whose columns are mean vectors. Required for 'classify'.
    R : numpy.ndarray, optional
        nc-by-np binary matrix containing target vectors (class membership).
        Required for 'train'. Optional for 'classify' (to compute rate).
        
    Returns:
    --------
    output : dict
        Dictionary containing:
        - 'M': Mean vectors (if 'train' or 'classify').
        - 'Class': Predicted class indices (1-based to match MATLAB or 0-based? Let's use 0-based for Python).
                   Wait, MATLAB typically uses 1-based. Let's return 0-based strictly for Python 
                   but maybe note it. The prompt implies transcoding.
        - 'RecogRate': Recognition accuracy percentage.
    """
    X = X.astype(float)
    n, np_count = X.shape
    
    output = {}
    
    # Determine dimensionality/classes
    nc = 0
    if M is not None:
        nc = M.shape[1]
    elif R is not None:
        nc = R.shape[0]
        
    if mode == 'train':
        if R is None:
            raise ValueError("Input R (Targets) is required for 'train' mode.")
        
        # Compute mean vector of each class
        M_out = np.zeros((n, nc))
        num_patterns = np.zeros(nc)
        
        # R is nc-by-np (one-hot ish)
        # Iterate over classes to handle empty classes correctly?
        # MATLAB loops over patterns? No:
        # for I=1:np, classid = find(R(:,I)==1) ...
        # Vectorized approach in Python:
        # class_indices = np.argmax(R, axis=0) assuming single label per col.
        
        class_indices = np.argmax(R, axis=0)
        
        for k in range(nc):
            # patterns belonging to class k
            w = np.where(class_indices == k)[0]
            num_patterns[k] = len(w)
            if len(w) > 0:
                M_out[:, k] = np.mean(X[:, w], axis=1)
            else:
                M_out[:, k] = 0 # Or keep zero
        
        output['M'] = M_out
        
        # Compute recognition rate on training data
        classes, correct_recog = md_classify(X, M_out, R)
        output['Class'] = classes
        output['RecogRate'] = correct_recog
        
    elif mode == 'classify':
        if M is None:
            raise ValueError("Input M (Means) is required for 'classify' mode.")
        
        output['M'] = M
        if R is not None:
            classes, correct_recog = md_classify(X, M, R)
            output['Class'] = classes
            output['RecogRate'] = correct_recog
        else:
            classes, _ = md_classify(X, M, None)
            output['Class'] = classes
            
    else:
        raise ValueError(f"Unknown mode: {mode}")
        
    return output

def md_classify(X, M, R=None):
    """
    Helper to classify patterns.
    """
    n, np_count = X.shape
    nc = M.shape[1]
    
    # Classify by calculating distance to each mean
    D = np.zeros((nc, np_count))
    
    for i in range(nc):
        # Mean vector i
        mean_vec = M[:, i].reshape(-1, 1) # (n, 1)
        # Distance squared
        diff = X - mean_vec
        D[i, :] = np.sum(diff**2, axis=0)
        
    # Find min distance index
    # MATLAB: minCol = min(D); ... find first min ...
    # Python: np.argmin returns first occurrence of min.
    class_ids = np.argmin(D, axis=0)
    
    correct_recog = 0
    if R is not None:
        # R is nc-by-np. True class indices.
        true_class_ids = np.argmax(R, axis=0)
        
        # Compare
        matches = (class_ids == true_class_ids)
        num_correct = np.sum(matches)
        correct_recog = (num_correct / np_count) * 100.0
        
    return class_ids, correct_recog
