
import numpy as np
import sys
# Try importing standard libraries if needed. 
# skimage.util.img_as_float is useful for conversion.

def imnoise2(f, type_noise, a=None, b=None):
    """
    Outputs noisy image and random matrix with given PDF.
    
    Parameters:
    f: Input image (ndarray) or shape tuple. 
       If f is an ndarray, it acts as the image to add noise to.
       If f is a tuple (M, N), it's treated as image dimensions for generating noise only (fn returned might be None or just the noise on zeros?).
       MATLAB version takes 'f' as image. To generate just noise, pass ones(M,N) or zeroes.
       
       We will emulate MATLAB behavior: f is expected to be the image.
       
    type_noise: String defining distribution.
    a, b: Parameters.
    
    Returns:
    fn: Noisy image (float, [0,1]).
    R: Noise pattern (float).
    """
    
    # Pre-processing input f
    # Convert to float [0, 1]
    # We'll expect f to be ndarray.
    
    f = np.asarray(f)
    if f.dtype.kind != 'f':
        # Simple scaling if integer
        if f.max() > 1.0:
            f_float = f.astype(float) / 255.0 # Naive scaling? Or assume user handles it?
            # MATLAB `im2double` scales integer types.
            # safe conversion:
            min_val = f.min()
            max_val = f.max()
            # If standard uint8
            if f.dtype == np.uint8:
                f_float = f.astype(float) / 255.0
            elif f.dtype == np.uint16:
                 f_float = f.astype(float) / 65535.0
            else:
                # heuristic
                f_float = (f - min_val) / (max_val - min_val) if max_val > min_val else f.astype(float)
        else:
            f_float = f.astype(float)
    else:
        f_float = f
        
    M, N = f.shape[:2] # Handling 2D for now, if RGB, iterate? 
    # MATLAB description implies 2D usually ("M-by-N random matrix").
    # If F is RGB, imnoise2 usually adds noise to all channels or treating MxNx3?
    # The code `R = a + b*randn(M,N)` would fail dimensionality if f is 3D.
    # The MATLAB code `[M,N] = size(f)` returns M, N*3 if RGB and simple size? No, returns [M, N, P].
    # But `rand(M,N)` creates 2D.
    # If f is RGB, `f + R` adds the *same* 2D noise plane to all channels? 
    # MATLAB broadcasting would error if sizes mismatch M,N,3 vs M,N.
    # The MATLAB `imnoise2` seems designed for grayscale ("input grayscale image F").
    # We will assume grayscale (2D).
    
    type_noise = type_noise.lower()
    
    R = None
    fn = None
    
    if type_noise == 'uniform':
        if a is None: a = 0.0
        if b is None: b = 1.0
        
        R = a + (b - a) * np.random.rand(M, N)
        fn = f_float + R
        
    elif type_noise == 'gaussian':
        if a is None: a = 0.0
        if b is None: b = 1.0
        
        R = a + b * np.random.randn(M, N)
        fn = f_float + R
        
    elif type_noise == 'salt & pepper':
        if a is None: a = 0.05
        if b is None: b = 0.05
        # a=Ps (salt prob), b=Pp (pepper prob)
        
        R = saltpepper(M, N, a, b)
        fn = f_float.copy()
        fn[R == 1] = 1.0 # Salt
        fn[R == 0] = 0.0 # Pepper
        
    elif type_noise == 'lognormal':
        if a is None: a = 1.0
        if b is None: b = 0.25
        
        # R = exp(b*randn + a)
        R = np.exp(b * np.random.randn(M, N) + a)
        fn = f_float + R
        
    elif type_noise == 'rayleigh':
        if a is None: a = 0.0
        if b is None: b = 1.0
        
        # R = a + (-b * log(1 - rand))^0.5
        R = a + np.sqrt(-b * np.log(1.0 - np.random.rand(M, N)))
        fn = f_float + R
        
    elif type_noise == 'exponential':
        if a is None: a = 1.0
        
        if a <= 0:
             raise ValueError("Parameter a must be positive for exponential type.")
             
        k = -1.0 / a
        R = k * np.log(1.0 - np.random.rand(M, N))
        fn = f_float + R
        
    elif type_noise == 'erlang':
        if a is None: a = 2.0
        if b is None: b = 5.0
        
        if b != int(b) or b <= 0:
            raise ValueError("Parameter b must be a positive integer for Erlang.")
        
        b = int(b)
        k = -1.0 / a
        R = np.zeros((M, N))
        for j in range(b):
            R += k * np.log(1.0 - np.random.rand(M, N))
            
        fn = f_float + R
        
    else:
        raise ValueError('Unknown distribution type.')
        
    # Scale/Clip fn to [0, 1]
    # MATLAB: fn = max(min(fn,1),0);
    fn = np.clip(fn, 0.0, 1.0)
    
    return fn, R

def saltpepper(M, N, a, b):
    # Check probabilities
    if (a + b) > 1:
        raise ValueError("The sum (Ps + Pp) must not exceed 1.")
        
    R = 0.5 * np.ones((M, N))
    X = np.random.rand(M, N)
    
    # Pepper (0) with prob b (Pp)
    R[X <= b] = 0
    # Salt (1) with prob a (Ps), range (b, a+b]
    R[(X > b) & (X <= a + b)] = 1
    
    return R
