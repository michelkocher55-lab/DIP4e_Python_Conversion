
import numpy as np
from skimage.transform import rotate
from skimage.util import img_as_float
from helpers.libdip.imPad4e import imPad4e
from helpers.libdip.intScaling4e import intScaling4e

def imRecon4e(f, theta):
    """
    Image reconstruction from projections.
    
    g = imRecon4e(f, theta)
    
    Parameters:
    f: input image
    theta: array of angles in degrees
    
    Returns:
    g: Reconstructed image, scaled to [0, 1].
    """
    
    f = img_as_float(f)
    M, N = f.shape[:2]
    
    # Ensure theta is iterable
    if np.isscalar(theta):
        theta = [theta]
    theta = np.asarray(theta).flatten()
    
    # Pad f
    # Beam is normal to angle. 
    # Theta logic matches MATLAB: theta + 90
    theta_adj = theta + 90
    
    # Diagonal size
    D_diag = np.sqrt(M**2 + N**2)
    # Padding amount D
    D = int(np.ceil((D_diag - max(M, N) + 1) / 2))
    
    f_pad = imPad4e(f, D, D) # default 'zeros', 'both'
    
    # Initialize accumulator g same size as padded f
    g = np.zeros_like(f_pad)
    
    # Rotate, Project, Backproject
    NL = len(theta_adj)
    
    for i in range(NL):
        angle = theta_adj[i]
        
        # Rotate image by -angle (equivalent to rotating sensors by angle)
        # skimage rotate uses degrees, counter-clockwise.
        # resize=False matches 'crop'
        rot = rotate(f_pad, -angle, resize=False, mode='constant', cval=0)
        
        # Projection: sum rows
        p = np.sum(rot, axis=1)
        
        # Smear projection across image (Backprojection)
        # p is 1D array of length rows. 
        # Make it (rows, 1) and tile across cols.
        temp = np.tile(p[:, np.newaxis], (1, g.shape[1]))
        
        # Rotate g to insert projection
        g_rot = rotate(g, -angle, resize=False, mode='constant', cval=0)
        
        # Insert
        g_rot = g_rot + temp
        
        # Rotate back
        g = rotate(g_rot, angle, resize=False, mode='constant', cval=0)
        
        # Progress (optional, print)
        # print(f"Processing angle {i+1}/{NL}...")

    # Crop back
    # MATLAB: g(D+1:D+M, D+1:D+N)
    # Python: g[D:D+M, D:D+N]
    g = g[D:D+M, D:D+N]
    
    # Scale to [0, 1]
    g = intScaling4e(g, 'full')
    
    return g
