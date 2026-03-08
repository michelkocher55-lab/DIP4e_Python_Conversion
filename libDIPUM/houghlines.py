
import numpy as np
import math

def houghlines(BW, theta, rho, peaks, fill_gap=20, min_length=40):
    """
    Extract line segments based on the Hough transform.
    
    Parameters:
        BW: Binary image (edge map).
        theta: Array of angles returned by hough_line (radians).
        rho: Array of distances returned by hough_line.
        peaks: Indices of peaks in the accumulator (size [N, 2] -> [rho_idx, theta_idx] or similar).
               Note: Figure1031.m passes P from houghpeaks. 
               In MATLAB houghpeaks returns [r, c] which are indices into the accumulator.
               So peaks[:, 0] is rho_idx, peaks[:, 1] is theta_idx.
        fill_gap: Max gap to merge lines.
        min_length: Min length to keep.
        
    Returns:
        lines: List of dictionaries [{'point1': (r1, c1), 'point2': (r2, c2), 'theta': t, 'rho': r}, ...].
               Points are (row, col).
    """
    
    lines = []
    rows, cols = BW.shape
    
    # Get all edge pixel locations
    # y (row), x (col)
    y_idxs, x_idxs = np.nonzero(BW)
    
    # Pre-calculate trig for all edge pixels against all peaks? 
    # Or just iterate peaks.
    
    for k in range(len(peaks)):
        rho_idx = peaks[k][0]
        theta_idx = peaks[k][1]
        
        theta_val = theta[theta_idx]
        rho_val = rho[rho_idx]
        
        # Check tolerance based on rho resolution
        rho_res = rho[1] - rho[0] if len(rho) > 1 else 1.0

        # Theta in degrees (from new hough.py), convert to radians for calc
        theta_rad = np.deg2rad(theta_val)
        
        # Calculate rho for all points
        # x_idxs * cos(t) + y_idxs * sin(t)
        proj_rho = x_idxs * np.cos(theta_rad) + y_idxs * np.sin(theta_rad)
        
        # Select pixels close to rho_val (within 0.5 bin size usually)
        mask = np.abs(proj_rho - rho_val) < (0.5 * rho_res)
        
        if not np.any(mask):
            continue
            
        current_y = y_idxs[mask]
        current_x = x_idxs[mask]
        
        # Now we have pixels.
        # Rotation logic from houghlines.m to align with vertical axis for gap filling.
        # MATLAB: omega = (90 - theta_deg) * pi / 180 = (pi/2 - theta_rad)
        
        omega = np.deg2rad(90 - theta_val)
        
        # 1D sort approach along the line
        # pos = x * (-sin t) + y * (cos t) ?
        # Wait, if we use MATLAB logic:
        # T = [cos(w) sin(w); -sin(w) cos(w)]
        # xy = [r-1 c-1] * T (row, col)
        # x_rot = row * cos(w) - col * sin(w) ? No row vector mul.
        # [r c] @ [[c -s], [s c]].
        # x_sorted = sort(xy(:,1)) -> sorted primarily by row-ish coordinate?
        # Let's stick to the 1D projection concept which is robust.
        # Line direction vector is (-sin(theta), cos(theta)).
        # pos = x * (-sin) + y * (cos).
        
        pos_along_line = -current_x * np.sin(theta_rad) + current_y * np.cos(theta_rad)
        
        # Sort
        sort_idx = np.argsort(pos_along_line)
        pos_sorted = pos_along_line[sort_idx]
        x_sorted = current_x[sort_idx]
        y_sorted = current_y[sort_idx]
        
        # Find gaps
        diffs = np.diff(pos_sorted)
        # Indices where gap > fill_gap
        # Note: fill_gap is in pixels distance
        gap_idxs = np.where(diffs > fill_gap)[0]
        
        # Segments are defined by [start_idx, end_idx]
        # gap_idxs gives index i such that diff between i and i+1 is big.
        # So split after i.
        
        segment_starts = [0] + list(gap_idxs + 1)
        segment_ends = list(gap_idxs) + [len(pos_sorted) - 1]
        
        for s, e in zip(segment_starts, segment_ends):
            # Check min length
            # Length can be measured by (end_pos - start_pos) or distance between endpoints
            p1 = np.array([y_sorted[s], x_sorted[s]]) # row, col
            p2 = np.array([y_sorted[e], x_sorted[e]])
            
            # Distance
            dist = np.linalg.norm(p1 - p2)
            
            if dist >= min_length:
                lines.append({
                    'point1': (p1[0], p1[1]), # row, col
                    'point2': (p2[0], p2[1]),
                    'theta': theta_val,
                    'rho': rho_val,
                    'length': dist
                })
                
    return lines
