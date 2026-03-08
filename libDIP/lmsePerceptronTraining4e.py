import numpy as np

def lmsePerceptronTraining4e(input_data):
    """
    Training of two-class LMS perceptron.
    
    Parameters:
    -----------
    input_data : dict
        Structure with fields:
        'X': numpy.ndarray (n+1) x np
             Pattern vectors. Last row must be 1s.
        'R': numpy.ndarray (np,)
             Class membership (1 or -1).
        'Alpha': float, optional (default 0.5)
             Learning rate (0, 2).
        'DelE': float, optional (default 0.001)
             Convergence error threshold.
        'Nepochs': int, optional (default 100)
             Max epochs.
        'W0': numpy.ndarray, optional
             Initial weights (n+1) x 1. Default random [0,1].
             
    Returns:
    --------
    output : dict
        Structure with fields:
        'W': Weights at convergence.
        'Error': List of squared errors per epoch.
        'ActualEpochs': Number of epochs run.
    """
    
    # Check inputs
    if 'X' not in input_data:
        raise ValueError("Pattern vectors must be provided in input_data['X']")
    X = np.array(input_data['X'])
    
    if 'R' not in input_data:
        raise ValueError("Pattern class membership must be provided in input_data['R']")
    R = np.array(input_data['R']).flatten()
    
    # Defaults
    alpha = input_data.get('Alpha', 0.5)
    del_e = input_data.get('DelE', 0.001)
    n_epochs = input_data.get('Nepochs', 100)
    
    n_dim, n_patterns = X.shape
    
    if 'W0' in input_data:
        w = np.array(input_data['W0'])
        if w.ndim == 1:
            w = w.reshape(-1, 1)
    else:
        # Random initial weights [0, 1]
        rng = np.random.default_rng()
        w = rng.random((n_dim, 1))
        
    w_old = w.copy()
    lmse_error = 0.0
    error_history = []
    
    actual_epochs = 0
    
    for i in range(n_epochs):
        actual_epochs = i + 1
        
        # Shuffle patterns?
        # MATLAB code iterates 1:np sequentially.
        # Online learning usually benefits from shuffling, but to match MATLAB exactly
        # we will iterate sequentially as written in the source.
        
        for j in range(n_patterns):
            x_j = X[:, j].reshape(-1, 1)
            r_j = R[j]
            
            # errorTerm = R(J) - W'*X(:,J)
            # w.T is (1, n), x_j is (n, 1) -> scalar
            # But wait, w uses w_old? No, MATLAB code:
            # errorTerm = (R(J) - W'*X(:,J));
            # W = Wold + alpha*errorTerm*X(:,J);
            # Wold = W;
            # So inside the loop, it uses 'W' for calculation, then updates 'W', then updates 'Wold'.
            # Wait, line 85: errorTerm uses W.
            # line 86: update W using Wold.
            # line 87: Wold = W.
            # Initialize W = Wold = W0.
            # It seems it updates incrementally.
            # In MATLAB, for J=1, W is W0. It computes error based on W0.
            # Then W_new = Wold (W0) + update.
            # Then Wold = W_new.
            # Then W = W_new.
            # So error is computed on current weights. Update is applied.
            
            prediction = np.dot(w.T, x_j).item()
            error_term = r_j - prediction
            
            # Update
            w = w_old + alpha * error_term * x_j
            w_old = w.copy()
            
            # Accumulate squared error
            # lmseError = lmseError + 0.5*(errorTerm^2);
            lmse_error += 0.5 * (error_term ** 2)
            
        # End of pattern loop
        
        mean_sq_error = lmse_error / n_patterns
        error_history.append(mean_sq_error)
        
        if lmse_error <= del_e:
            break
            
        lmse_error = 0.0
        
    output = {
        'W': w,
        'Error': np.array(error_history),
        'ActualEpochs': actual_epochs
    }
    
    return output
