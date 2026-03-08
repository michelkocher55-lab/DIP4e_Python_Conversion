
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.measure import find_contours
from scipy.spatial.distance import pdist, squareform
from libDIPUM.diameter import diameter
from libDIPUM.x2majoraxis import x2majoraxis
import os
import itertools

def Equation12_4():
    # Equation12_4.m

    # Paths
    # legbone.tif location check
    paths = [
        dip_data('legbone.tif'),
        'legbone.tif' 
    ]
    img_path = None
    for p in paths:
        if os.path.exists(p):
            img_path = p
            break
            
    if img_path is None:
        print("Error: legbone.tif not found.")
        return

    print(f"Reading image from {img_path}...")
    X = imread(img_path)
    # Ensure boolean
    if X.dtype != bool:
        X = X > 0

    # Diameter
    # diameter returns a list of results. We assume 1 region.
    print("Computing Diameter (efficient)...")
    S_list = diameter(X)
    if not S_list:
        print("No regions found.")
        return
        
    S = S_list[0]
    
    # x2majoraxis (For demo, though not plotted in subplot 1/2)
    # [C, Theta] = x2majoraxis (S.MajorAxis, X);
    # My python diameter returns MajorAxis as [[r1, c1], [r2, c2]]
    # x2majoraxis expects A, B.
    print("Aligning to Major Axis...")
    C, Theta = x2majoraxis(S.MajorAxis, X)

    # Diameter, brute force ...
    print("Computing Diameter (brute force)...")
    # B = bwboundaries (X);
    contours = find_contours(X, 0.5)
    # contours returns float coordinates. Using pixels [row, col]
    # We want integer pixel coordinates of the boundary?
    # bwboundaries finds the set of pixels on the boundary. 
    # find_contours finds standard marching squares contours.
    # For accurate pixel-to-pixel distance, we should get boundary pixels.
    from skimage.segmentation import find_boundaries
from libDIPUM.data_path import dip_data
    boundary_mask = find_boundaries(X, mode='inner')
    pts_r, pts_c = np.where(boundary_mask)
    B = np.column_stack((pts_r, pts_c))
    
    # Brute force on boundary pixels
    # MATLAB: Couples = nchoosek (1:N, 2); D = hypot(...);
    # We use pdist.
    dists = pdist(B, 'euclidean')
    DMax = np.max(dists)
    
    # Find indices
    dist_mat = squareform(dists) # Can be large
    idx_flat = np.argmax(dist_mat)
    idx_i, idx_j = np.unravel_index(idx_flat, dist_mat.shape)
    
    BestFrom = B[idx_i]
    BestTo = B[idx_j]
    
    # Check consistency
    print(f"Diameter (Efficient): {S.Diameter:.4f}")
    print(f"Diameter (Brute Force): {DMax:.4f}")
    
    # Display
    fig = plt.figure(figsize=(12, 6))
    
    # Subplot 1: Brute Force
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(X, cmap='gray')
    ax1.plot(BestFrom[1], BestFrom[0], 'or', markersize=8) # Col, Row
    ax1.plot(BestTo[1], BestTo[0], 'or', markersize=8)
    # Green face color?
    ax1.plot(BestFrom[1], BestFrom[0], 'og', markersize=4) 
    ax1.plot(BestTo[1], BestTo[0], 'og', markersize=4)
    ax1.set_title(f"Brute force, N={len(B)*(len(B)-1)//2}, D={DMax:.2f}")
    
    # Subplot 2: Calculated
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(X, cmap='gray')
    
    # Major Axis (Red)
    # S.MajorAxis is [[r1 c1], [r2 c2]]
    # Plot line: plot([c1, c2], [r1, r2])
    maj = S.MajorAxis
    ax2.plot([maj[0, 1], maj[1, 1]], [maj[0, 0], maj[1, 0]], 'r-', linewidth=2)
    
    # Minor Axis (Green)
    mino = S.MinorAxis
    ax2.plot([mino[0, 1], mino[1, 1]], [mino[0, 0], mino[1, 0]], 'g-', linewidth=2)
    
    # Basic Rectangle (Blue)
    # S.BasicRectangle: [[r1 c1], [r2 c2], [r3 c3], [r4 c4]]
    rect = S.BasicRectangle
    # Close the loop
    rect_plot = np.vstack((rect, rect[0]))
    ax2.plot(rect_plot[:, 1], rect_plot[:, 0], 'b-', linewidth=1)
    
    ax2.set_title(f"X, Maj(r), Min(g), Rect(b), D={S.Diameter:.2f}")
    
    plt.tight_layout()
    plt.savefig('Equation12_4.png')
    plt.show()

if __name__ == "__main__":
    Equation12_4()
