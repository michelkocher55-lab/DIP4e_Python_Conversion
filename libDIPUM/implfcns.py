
import numpy as np

def implfcns(L, outmf, *inputs):
    """
    Implication functions for a fuzzy system.
    
    Parameters:
    L (list of callables): Rule strength functions.
    outmf (list of callables): Output MFs.
    *inputs: Input values Z1, Z2...
    
    Returns:
    Q (list of callables): Implication functions Q[i](v).
    """
    num_rules = len(L)
    lambdas = np.zeros(num_rules)
    
    # Compute rule strengths
    # If inputs are arrays, lambdas will be arrays?
    # MATLAB code: lambdas(i) = L{i}(Z{:}).
    # If Z are arrays, L returns array. lambdas should be list or array of arrays.
    
    # Let's assume we handle scalar inputs for now inside the closure or
    # keep lambdas as list of results (which might be scalars or arrays).
    lambdas = [func(*inputs) for func in L]
    
    Q = []
    
    # Create implication functions
    for i in range(num_rules):
        def make_impl(idx, lam):
            def implication(v):
                # Min of lambda and output MF
                # lam might be array if inputs were arrays.
                # v is scalar (in defuzzify loop) or array (linspace).
                return np.fmin(lam, outmf[idx](v))
            return implication
        Q.append(make_impl(i, lambdas[i]))
        
    # Else rule if needed
    if len(outmf) == num_rules + 1:
        # Compute lambda_e = min(1 - rule_strengths)
        # Assuming lambdas is list of arrays/scalars
        # 1 - lambdas
        lambdas_arr = np.array(lambdas)
        if lambdas_arr.ndim == 1:
            lambda_e = np.min(1 - lambdas_arr)
        else:
             # If arrays, min across rules axis (0)
             lambda_e = np.min(1 - lambdas_arr, axis=0)
             
        def make_else(lam_e):
            def else_rule(v):
                return np.fmin(lam_e, outmf[-1](v))
            return else_rule
            
        Q.append(make_else(lambda_e))
        
    return Q
