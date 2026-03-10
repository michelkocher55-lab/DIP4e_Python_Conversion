import numpy as np

def makeMark4e(sd, k):
    """
    Generates a K-element pseudorandom watermark vector after seeding
    the random number generator with SD.
    
    Parameters:
    -----------
    sd : int
        Seed for the random number generator.
    k : int
        Length of the watermark vector.
        
    Returns:
    --------
    m : numpy.ndarray
        Watermark vector of length k (mean 0, stdev 1).
    """
    # Create a seeded generator
    rng = np.random.default_rng(sd)
    
    # Generate vector of mean 0 and stdev 1 (standard normal)
    m = rng.standard_normal(k)
    
    return m
