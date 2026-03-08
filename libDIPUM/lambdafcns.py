
import numpy as np

def lambdafcns(inmf, op=np.fmin):
    """
    Lambda functions for a set of fuzzy rules.
    
    Parameters:
    inmf (list of lists of callables): M x N matrix of input MFs.
          M rules, N inputs. 
          inmf[i][j] is the MF for rule i, input j.
    op (callable): Operator to combine antecedents (default np.fmin for 'min').
    
    Returns:
    L (list of callables): List of rule strength functions.
        L[i](*inputs) returns the strength of rule i.
    """
    num_rules = len(inmf)
    L = []
    
    for i in range(num_rules):
        # Create a closure for the rule
        def make_rule_strength(rule_idx):
            def rule_strength(*inputs):
                # inputs is a tuple of (Z1, Z2, ... ZN)
                # each Zn can be scalar or array.
                
                # Apply first input MF
                mf0 = inmf[rule_idx][0]
                strength = mf0(inputs[0])
                
                # Combine with rest
                for j in range(1, len(inputs)):
                     mfj = inmf[rule_idx][j]
                     val = mfj(inputs[j])
                     strength = op(strength, val)
                     
                return strength
            return rule_strength
            
        L.append(make_rule_strength(i))
        
    return L
