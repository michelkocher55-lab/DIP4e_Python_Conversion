import numpy as np
from lib.twodConv4e import twodConv4e

def edgeAngle4e(f, type='sobel', T=0.0):
    """
    Computes the angle of the image gradient.

    Parameters:
    -----------
    f : numpy.ndarray
        Input image.
    type : str, optional
        Kernel type: 'prewitt', 'sobel', or 'kirsch'. Default is 'sobel'.
    T : float, optional
        Threshold in range [0, 1] representing percentage of 360 degrees.
        Default is 0.0 (no thresholding).

    Returns:
    --------
    ang : numpy.ndarray
        Angle image values in range [0, 360] degrees.
        If T > 0, returns a boolean array where ang > T*360.
    """
    # Preliminaries
    f = np.array(f, dtype=float)
    
    # Define kernels
    if type == 'prewitt':
        wx = np.array([[-1, -1, -1],
                       [ 0,  0,  0],
                       [ 1,  1,  1]])
        wy = wx.T
    elif type == 'sobel':
        wx = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]])
        wy = wx.T
    elif type == 'kirsch':
        # Kirsch kernels (8 directions)
        # N, NW, W, SW, S, SE, E, NE
        # But MATLAB implementation generates them by rotating 45 degrees.
        # Let's match the MATLAB order:
        # k=1: N (implied starting point in loop)
        # The MATLAB code does:
        # w(:,:,1) = [-3 -3 -3; -3 0 -3; 5 5 5]  <-- This looks like North (if 5s are positive/max response direction?)
        # Wait, standard Kirsch North has 5s on top? No, usually it's about the gradient direction.
        # Let's check the code:
        # w(:,:,1) = [-3 -3 -3;-3 0 -3; 5 5 5];
        # imageRotate4e(w, 45) -> rotates counter-clockwise.
        
        # We can implement this by manually defining the 8 kernels to avoid imageRotate4e dependency complexity here
        # and for better performance.
        
        # k=1 (Base): [-3 -3 -3; -3 0 -3; 5 5 5] (Matches MATLAB base)
        k1 = np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]])
        
        # k=2 (45 deg CCW): E? No, let's trace rotation.
        # 45 deg CCW rotation of a matrix.
        # If we stick to the cyclic permutation of the 8 border elements.
        # [-3 -3 -3]      [-3 -3  5]
        # [-3  0 -3]  ->  [-3  0  5]
        # [ 5  5  5]      [-3 -3  5]
        # Let's generate them properly.
        
        w_all = np.zeros((3, 3, 8))
        
        # Define the boundary elements in clockwise order starting from top-left (0,0)
        # 0,0  0,1  0,2
        # 1,0       1,2
        # 2,0  2,1  2,2
        # Indices: (0,0), (0,1), (0,2), (1,2), (2,2), (2,1), (2,0), (1,0)
        # Values in k1: -3, -3, -3, -3, 5, 5, 5, -3 ... wait
        # k1:
        # -3 -3 -3
        # -3  0 -3
        #  5  5  5
        
        # For Kirsch, it's easier to just compute max response.
        # Let's define the 8 masks explicitly to be safe and match MATLAB's rotation logic exactly if possible,
        # or use standard Kirsch definition. MATLAB code creates subsequent kernels by rotating the previous one by 45 degrees.
        
        # Let's trust the standard Kirsch set order if it matches.
        # Actually, let's just implement the rotation logic on the 3x3 grid directly, it's safer.
        
        # Flatten the 3x3 border
        # border indices: 0 1 2 5 8 7 6 3 (if flattened row major 0..8)
        # (0,0) (0,1) (0,2) (1,2) (2,2) (2,1) (2,0) (1,0)
        border_vals = [-3, -3, -3, -3, 5, 5, 5, -3] # From k1
        
        # The MATLAB code rotates w by 45 degrees.
        # imageRotate4e positive angle = counter-clockwise.
        # Rotating the image (kernel) by 45 deg CCW shifts the pixels.
        
        w_list = []
        w_list.append(k1)
        
        # Construct the other 7 by rotating the values in the border
        # A 45 degree rotation of the kernel structure corresponds to shifting the border 
        # elements by 1 position.
        # If we rotate image CCW, pixels move CCW.
        # So values at (0,0) moves to (1,0)? No, (1,0) is below.
        # Rotation 45 deg CCW:
        # top-left -> mid-left
        # top-mid -> top-left
        # etc.
        # So the sequence of values shifts clockwise?
        
        current_border = border_vals
        
        # Standard Kirsch masks are cyclic shifts of [-3 -3 -3 -3 -3  5  5  5] usually.
        # Here we have three 5s.
        
        # Let's manually define them to ensure correctness.
        # K1 (0 deg relative to this definition):
        # -3 -3 -3
        # -3  0 -3
        #  5  5  5
        
        # K2 (45):
        # -3 -3 -3
        #  5  0 -3
        #  5  5 -3
        
        # K3 (90):
        #  5 -3 -3
        #  5  0 -3
        #  5 -3 -3
        
        # K4 (135):
        #  5  5 -3
        #  5  0 -3
        # -3 -3 -3
        
        # K5 (180):
        #  5  5  5
        # -3  0 -3
        # -3 -3 -3
        
        # K6 (225):
        # -3  5  5
        # -3  0  5
        # -3 -3 -3
        
        # K7 (270):
        # -3 -3  5
        # -3  0  5
        # -3 -3  5
        
        # K8 (315):
        # -3 -3 -3
        # -3  0  5
        # -3  5  5
        
        # I'll put these in a list.
        # Order: N, NW, W, SW, S, SE, E, NE (CCW rotation)
        # Note: Previous manual list was CW. 
        # N=0, NE=1(my old 7?), E=2, SE=3, S=4, SW=5, W=6, NW=7.
        # Let's explicitly define them to be sure.
        
        # K0 (N): 5s Bottom, Flipped-> 5s Top. Top Bright. Up Gradient.
        k0 = np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]])
        
        # K1 (NW): Rotate K0 45 CCW. 5s move Right (CCW along perimeter).
        # 5s at Bottom-Right.
        k1 = np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])
        
        # K2 (W): Rotate K1 45 CCW. 5s at Right.
        k2 = np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]])
        
        # K3 (SW): 5s Top-Right.
        k3 = np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]])
        
        # K4 (S): 5s Top.
        k4 = np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]])
        
        # K5 (SE): 5s Top-Left.
        k5 = np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])
        
        # K6 (E): 5s Left.
        k6 = np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]])
        
        # K7 (NE): 5s Bot-Left.
        k7 = np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]])

        kernels_kirsch = [k0, k1, k2, k3, k4, k5, k6, k7]
        
    else:
        raise ValueError("Unknown kernel type. Valid values are 'prewitt', 'sobel', 'kirsch'.")

    # Compute angle image
    if type in ['prewitt', 'sobel']:
        gy = twodConv4e(f, wy)
        gx = twodConv4e(f, wx)
        
        # Compute angle in degrees.
        # MATLAB: ang = atan2(gy,gx)*(-180/pi);
        # The minus sign is for x-axis/angle convention (y axis down is positive in image coordinates)
        # atan2(y, x) gives angle from x-axis.
        # If gy is positive (down), and gx is positive (right).
        # We want standard math geometric angle?
        # MATLAB comment: "The minus is to correpond to x-axis and angle convention (see Fig. 10.12)."
        ang = np.arctan2(gy, gx) * (-180 / np.pi)
        
        # Map to 0-360 range if needed?
        # MATLAB atan2 returns [-180, 180].
        # The prompt says "ANG is the angle image whose values are in the range 0 to 360 degrees."
        # So we need to normalize.
        # The MATLAB code for Prewitt/Sobel: ang = atan2(gy,gx)*(-180/pi);
        # Wait, does the MATLAB code normalize?
        # The provided MATLAB code:
        # ang = atan2(gy,gx)*(-180/pi);
        # It does NOT seem to normalize to [0, 360] explicitly in the Prewitt/Sobel block.
        # However, for Kirsch, it produces 0, 45, ..., 315.
        # Let's check if the user wants strictly 0-360.
        # "ANG is the angle image whose values are in the range 0 to 360 degrees."
        # If atan2 returns negative, we should add 360.
        
        ang = np.mod(ang, 360) 
        
    else: # Kirsch
        # Convolve with each kernel
        K = 8
        if f.ndim == 3:
            M, N, C = f.shape
            cv = np.zeros((M, N, C, K))
             # This might get complicated for color, usually edge detection is on gray.
             # Assuming gray for standard edge detection, but let's handle if twodConv handles it.
             # If f is color, twodConv4e handles it channel-wise.
             # But max response across kernels?
             # Let's assume grayscale input or process channels independently.
             # For simplicity and typical usage, we'll assume 2D or handle 3D by iterating.
        else:
            M, N = f.shape
            cv = np.zeros((M, N, K))
            
        for k in range(K):
            cv[..., k] = twodConv4e(f, kernels_kirsch[k])
            
        cv = np.abs(cv)
        
        # Find max response index
        # MATLAB:
        # ang = angle associated with max kernel.
        # k=1 -> 0 deg?
        # MATLAB code:
        # k=1; angle = 45*(k - 1);  -> 0
        # ...
        # maxValues tracking.
        
        # In Python we can use argmax.
        max_idx = np.argmax(cv, axis=-1) # Indices 0..7
        ang = max_idx * 45.0
        
    # Thresholding
    if T != 0.0:
        ang = ang > (T * 360.0)
        
    return ang
