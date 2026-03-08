
import numpy as np
import scipy.sparse as sp
from scipy.ndimage import convolve, rotate
from scipy.fftpack import fft2, ifft2, fftshift
from scipy.sparse.linalg import eigsh
from skimage.draw import line
from skimage.measure import label
from skimage.transform import resize
import math

# -------------------------------------------------------------------------
# Filter Bank Generation (DoG / DOOG)
# -------------------------------------------------------------------------
# Same as before, standard functions...

def gaussian_2d(X, mean, cov):
    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)
    diff = X - mean
    exponent = -0.5 * np.sum((diff @ inv_cov) * diff, axis=1)
    return np.exp(exponent)

def doog1(sig, r, th_deg, N):
    no_pts = N
    rng = np.arange(-(N/2)+0.5, (N/2)+0.5)
    x, y = np.meshgrid(rng, rng)
    phi = np.deg2rad(th_deg)
    sigy = sig
    sigx = r * sig
    R = np.array([[np.cos(phi), -np.sin(phi)],
                  [np.sin(phi),  np.cos(phi)]])
    C = R @ np.diag([sigx, sigy]) @ R.T
    X_vec = np.column_stack((x.ravel(), y.ravel()))
    Gb = gaussian_2d(X_vec, np.array([0, 0]), C)
    Gb = Gb.reshape(N, N)
    m = R @ np.array([0, sig])
    Ga = gaussian_2d(X_vec, m/2, C)
    Ga = Ga.reshape(N, N)
    Gb_rot = np.rot90(Ga, 2)
    H = Ga - Gb_rot
    return H

def doog2(sig, r, th_deg, N):
    no_pts = N
    rng = np.arange(-(N/2)+0.5, (N/2)+0.5)
    x, y = np.meshgrid(rng, rng)
    phi = np.deg2rad(th_deg)
    sigy = sig
    sigx = r * sig
    R = np.array([[np.cos(phi), -np.sin(phi)],
                  [np.sin(phi),  np.cos(phi)]])
    C = R @ np.diag([sigx, sigy]) @ R.T
    X_vec = np.column_stack((x.ravel(), y.ravel()))
    Gb = gaussian_2d(X_vec, np.array([0, 0]), C)
    Gb = Gb.reshape(N, N)
    m = R @ np.array([0, sig])
    Ga = gaussian_2d(X_vec, m, C)
    Ga = Ga.reshape(N, N)
    Gc = np.rot90(Ga, 2)
    G = -Ga + 2*Gb - Gc
    return G

def make_filterbank_odd2(num_ori, filter_scales, wsz, enlong=3):
    enlong = enlong * 2
    num_scale = len(filter_scales)
    ori_incr = 180.0 / num_ori
    ori_offset = ori_incr / 2.0
    total_filters = num_scale * num_ori
    FB = np.zeros((wsz, wsz, total_filters))
    idx = 0
    for m in range(num_scale):
        for n in range(num_ori):
            scale = filter_scales[m]
            angle = ori_offset + n * ori_incr
            f = doog1(scale, enlong, angle, wsz)
            a = np.sum(np.abs(f))
            if a > 0: f = f / a
            FB[:, :, idx] = f
            idx += 1
    return FB

def make_filterbank_even2(num_ori, filter_scales, wsz, enlong=3):
    enlong = enlong * 2
    num_scale = len(filter_scales)
    ori_incr = 180.0 / num_ori
    ori_offset = ori_incr / 2.0
    total_filters = num_scale * num_ori
    FB = np.zeros((wsz, wsz, total_filters))
    idx = 0
    for m in range(num_scale):
        for n in range(num_ori):
            scale = filter_scales[m]
            angle = ori_offset + n * ori_incr
            f = doog2(scale, enlong, angle, wsz)
            a = np.sum(np.abs(f))
            if a > 0: f = f / a
            FB[:, :, idx] = f
            idx += 1
    return FB

def fft_filt_2(img, FB):
    H, W = img.shape
    M, M_filt, K = FB.shape
    out = np.zeros((H, W, K))
    from scipy.signal import fftconvolve
    for k in range(K):
        filt = FB[:, :, k]
        res = fftconvolve(img, filt, mode='same')
        out[:, :, k] = res
    return out

def quadedgep(I, par=None, threshold=0.2):
    I = I.astype(float)
    r, c = I.shape
    if par is None:
        par = [0, 0, 0, 0]
    def_par = [8, 1, 20, 3]
    p_final = []
    for j in range(4):
        val = par[j]
        if val == 0: val = def_par[j]
        p_final.append(val)
    n_filter, n_scale, winsz, enlong = p_final
    j_sz = winsz / 2
    if not (j_sz > int(j_sz) + 0.1): winsz = int(winsz) + 1
    else: winsz = int(winsz)

    scales = [n_scale] if np.isscalar(n_scale) else n_scale
    FBo = make_filterbank_odd2(n_filter, scales, winsz, enlong)
    FBe = make_filterbank_even2(n_filter, scales, winsz, enlong)
    
    n_pad = int(np.ceil(winsz / 2))
    col_pre = np.fliplr(I[:, 1:n_pad+1])
    col_post = np.fliplr(I[:, c-n_pad:c-1])
    f = np.concatenate([col_pre, I, col_post], axis=1)
    row_pre = np.flipud(f[1:n_pad+1, :])
    row_post = np.flipud(f[f.shape[0]-n_pad:f.shape[0]-1, :])
    f = np.concatenate([row_pre, f, row_post], axis=0)
    
    FIo_full = fft_filt_2(f, FBo)
    FIe_full = fft_filt_2(f, FBe)
    FIo = FIo_full[n_pad:n_pad+r, n_pad:n_pad+c, :]
    FIe = FIe_full[n_pad:n_pad+r, n_pad:n_pad+c, :]
    
    mag = np.sqrt(np.sum(FIo**2, axis=2) + np.sum(FIe**2, axis=2))
    mag_a = np.sqrt(FIo**2 + FIe**2)
    max_id = np.argmax(mag_a, axis=2)
    R_idx, C_idx = np.indices((r, c))
    mage = FIe[R_idx, C_idx, max_id]
    mage = (mage > 0).astype(float) - (mage < 0).astype(float)
    
    ori_incr = np.pi / n_filter
    ori_offset = ori_incr / 2.0
    thetas = ori_offset + np.arange(n_filter) * ori_incr
    mago = FIo[R_idx, C_idx, max_id]
    ori = thetas[max_id]
    ori = ori * (mago > 0) + (ori + np.pi) * (mago < 0)
    
    gy = mag * np.cos(ori)
    gx = -mag * np.sin(ori)
    
    mag_th = mag.max() * threshold
    eg = (mag > mag_th)
    
    diff_v = (mage[1:, :] != mage[:-1, :])
    zeros_row = np.zeros((1, c), dtype=bool)
    h = eg & np.vstack([diff_v, zeros_row])
    
    diff_h = (mage[:, 1:] != mage[:, :-1])
    zeros_col = np.zeros((r, 1), dtype=bool)
    v = eg & np.hstack([diff_h, zeros_col])
    
    Y, X = np.where(h | v)
    h_vals = h[Y, X]; v_vals = v[Y, X]
    Y_float = Y + 1 + h_vals * 0.5
    X_float = X + 1 + v_vals * 0.5
    
    gx_1 = gx[Y, X]; gy_1 = gy[Y, X]
    Y_n = Y + h_vals.astype(int); X_n = X + v_vals.astype(int)
    Y_n = np.clip(Y_n, 0, r-1); X_n = np.clip(X_n, 0, c-1)
    gx_2 = gx[Y_n, X_n]; gy_2 = gy[Y_n, X_n]
    gx_out = gx_1 + gx_2
    gy_out = gy_1 + gy_2
    
    return X_float, Y_float, gx_out, gy_out, [n_filter, n_scale, winsz, enlong], threshold, mag, mage

def computeEdges(imageX, parametres=None, threshold=0.02):
    I = imageX
    if parametres is None: parametres = [4, 3, 21, 3]
    ex, ey, egx, egy, eg_par, eg_th, emag, ephase = quadedgep(I, parametres, threshold)
    from skimage.feature import canny
    edges_canny = canny(imageX) * 1.0
    edges2 = emag * edges_canny
    edges2 = edges2 * (edges2 > threshold)
    res = {'emag': emag, 'ephase': ephase, 'imageEdges': edges2}
    return res

# -------------------------------------------------------------------------
# Graph Construction (OPTIMIZED)
# -------------------------------------------------------------------------

def computeW(imageX, dataW=None, emag=None, ephase=None, **kwargs):
    if dataW is None: pass 
    r = kwargs.get('sampleRadius', 10)
    sample_rate = kwargs.get('sample_rate', 0.2)
    edgeVariance = kwargs.get('edgeVariance', 0.1)
    
    # Check if dataW structure logic is needed (MATLAB compat)
    # If parameters passed directly as kwargs, use them.
    
    nr, nc = imageX.shape
    N = nr * nc
    
    sigma = emag.max() * edgeVariance
    variance_factor = 2 * sigma**2
    if variance_factor == 0: variance_factor = 1e-10

    # Vectorized computeW
    # Instead of pairs, loop over offsets (dy, dx)
    
    I_list = []
    J_list = []
    W_list = []
    
    # Indices grid
    indices = np.arange(N).reshape(nr, nc)
    
    # For each possible offset in window
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            if dy == 0 and dx == 0:
                # Self loop, affinity 1.0??
                # Typically N-cut graph has self-affinity 1 or 0?
                # Code has W[u,u] based on something? 
                # cimgnbmap includes self. affinityic makes w=1 if i==j.
                I_list.append(np.arange(N))
                J_list.append(np.arange(N))
                W_list.append(np.ones(N))
                continue
                
            # Random sample rate check (per OFFSET, approximation to per-pixel)
            # If we skip an offset, we skip it for ALL pixels.
            # Ideally sample_rate is per pixel pair.
            # But that breaks vectorization.
            # Compromise: Check sample_rate per offset? Or generate random mask?
            # Random mask is better.
            
            # Line points
            # rr, cc = line(0, 0, dy, dx) returns points starting at 0,0 ending at dy,dx
            # Shifted to every pixel!
            line_y, line_x = line(0, 0, dy, dx)
            # These are relative offsets from (0,0).
            
            # Construct shifts for the whole line
            # We want max(emag) along this line for every pixel.
            # emag is (nr, nc).
            # stack shifted emag
            
            max_edge_val = np.zeros((nr, nc))
            
            # Optim: Iterate points on the line
            if len(line_y) == 0: continue
            
            # Initialize with start point (0,0 shift) -> emag itself
            # But line includes start point.
            
            pass_valid = True
            
            # Shift and check bounds
            # For vectorization, we only care about the overlap area where (y+dy, x+dx) is valid.
            # Valid region for source pixels:
            # y in [max(0, -dy), min(nr, nr-dy))
            # x in [max(0, -dx), min(nc, nc-dx))
            
            y_start = max(0, -dy); y_end = min(nr, nr-dy)
            x_start = max(0, -dx); x_end = min(nc, nc-dx)
            
            if y_start >= y_end or x_start >= x_end: continue
            
            # Crop to valid source region
            valid_h = y_end - y_start
            valid_w = x_end - x_start
            
            # Indices of source pixels
            src_indices = indices[y_start:y_end, x_start:x_end].flatten()
            
            # Indices of dest pixels
            dst_indices = indices[y_start+dy:y_end+dy, x_start+dx:x_end+dx].flatten()
            
            # Random sampling mask
            n_valid = len(src_indices)
            if sample_rate < 1.0:
                 mask = np.random.rand(n_valid) < sample_rate
                 src_indices = src_indices[mask]
                 dst_indices = dst_indices[mask]
                 # We also need pixel coords for these selected ones to compute max_edge
                 # Recomputing coords of selected src
                 sel_rows = src_indices // nc # Wait, src_indices is index in flattened array
                 sel_rows = src_indices // nc # No, reshape?
                 # src_indices are global indices.
                 # Let's get r, c
                 r_src = src_indices % nr # Fortran? No, numpy default C?
                 # Wait, MATLAB uses Fortran order. My indices/setup uses C order?
                 # `indices = np.arange(N).reshape(nr, nc)` -> C order logic (0,1.. at row 0).
                 # If I map to MATLAB, I should be careful.
                 # But self-consistent Python is fine as long as shape is consistent.
                 # r_src = src_indices // nc (Row)
                 # c_src = src_indices % nc (Col)
                 r_src = src_indices // nc
                 c_src = src_indices % nc
            else:
                 # Full block
                 # Coords relative to crop
                 # y grid: y_start..y_end
                 pass
            
            # If sampled set is empty
            if len(src_indices) == 0: continue

            # Optimized Phase-Sensitive Line Scan
            # We iterate steps along the line.
            # For each step k -> k+1, check if phase changes.
            # If changed, candidate edge val = (mag[k] + mag[k+1]).
            # Barrier = max(candidate values).
            
            # Initialize barrier with 0 (no edge)
            temp_max_barrier = np.zeros((valid_h, valid_w))
            
            # Base frame (k=0)
            # We need to track current Phase and Mag to compare with next
            
            # Initial point (k=0)
            ly_prev = line_y[0]; lx_prev = line_x[0]
            mag_prev = emag[y_start+ly_prev : y_end+ly_prev, x_start+lx_prev : x_end+lx_prev]
            phase_prev = ephase[y_start+ly_prev : y_end+ly_prev, x_start+lx_prev : x_end+lx_prev]
            
            for k in range(1, len(line_y)):
                ly_curr = line_y[k]
                lx_curr = line_x[k]
                
                # Slices for k
                mag_curr = emag[y_start+ly_curr : y_end+ly_curr, x_start+lx_curr : x_end+lx_curr]
                phase_curr = ephase[y_start+ly_curr : y_end+ly_curr, x_start+lx_curr : x_end+lx_curr]
                
                # Check phase crossing
                # Using != for float phases might be risky? 
                # ephase from quadedgep is based on `max_id`? 
                # No, `quadedgep` output `ephase` is NOT `mage` (phase map). 
                # Wait. `quadedgep` output signature: `ex, ey, egx, egy, eg_par, eg_th, emag, ephase`.
                # In `quadedgep.m`:
                # `edgemap.ephase = ephase;`
                # But where is `ephase` calculated?
                # Line 7: `[ex... ephase, g] = quadedgep(...)`.
                # Recursion? No.
                # In `quadedgep.m`:
                # `mage` is calculated (line 73).
                # `ephase` is NOT explicitly returned in the list `[x,y,gx,gy,par,threshold,mag,mage,g,FIe,FIo,mago]`
                # Wait. `computeEdges.m` line 7:
                # `[ex,ey,egx,egy,eg_par,eg_th,emag,ephase , g ] = quadedgep(...)`.
                # But `quadedgep` definition line 25:
                # `function [x,y,gx,gy,par,threshold,mag,mage,g,FIe,FIo,mago] = quadedgep(...)`.
                # Matching outputs:
                # ex=x, ey=y, egx=gx, egy=gy, eg_par=par, eg_th=threshold, emag=mag, ephase=mage.
                # So `ephase` IS `mage`.
                # In Python `quadedgep` return:
                # `return X_float, Y_float, gx_out, gy_out, [...params], threshold, mag, mage`.
                # In `computeEdges`:
                # `ex, ey, egx, egy, eg_par, eg_th, emag, ephase = quadedgep(...)`.
                # So `ephase` holds `mage`.
                # `mage` calculation: `mage = (mage>0) - (mage<0)`. Values are 1.0 or -1.0 (or 0).
                # So strict inequality check `!=` works.
                
                crossing = (phase_prev != phase_curr)
                
                # Edge value at this crossing
                # z = mag1 + mag2
                z = mag_prev + mag_curr
                
                # Update barrier where crossing occurs
                # We want max over the line
                # Only update where crossing is True
                
                # We can zero out non-crossings
                z_masked = z * crossing.astype(float)
                
                temp_max_barrier = np.maximum(temp_max_barrier, z_masked)
                
                # Update prev
                mag_prev = mag_curr
                phase_prev = phase_curr
                
            # Final barrier scaling
            # MATLAB C++: maxori = 0.5 * maxori
            barrier = temp_max_barrier.flatten() * 0.5
            
            # Generate Source indices for the crop
            # Grid of indices
            # Global indices of source crop
            # Global indices = (r)*nc + c (if C-order)
            # Create meshgrid for crop
            
            # Use precomputed `indices` grid
            crop_src_indices = indices[y_start:y_end, x_start:x_end].flatten()
            crop_dst_indices = indices[y_start+dy:y_end+dy, x_start+dx:x_end+dx].flatten()
            
            if sample_rate < 1.0:
                 mask = np.random.rand(len(crop_src_indices)) < sample_rate
                 final_src = crop_src_indices[mask]
                 final_dst = crop_dst_indices[mask]
                 final_barrier = barrier[mask]
            else:
                 final_src = crop_src_indices
                 final_dst = crop_dst_indices
                 final_barrier = barrier

            # Affinity
            w = np.exp( - (final_barrier**2) / variance_factor )
            
            I_list.append(final_src)
            J_list.append(final_dst)
            W_list.append(w)
            
    Ii = np.concatenate(I_list)
    Jj = np.concatenate(J_list)
    Ww = np.concatenate(W_list)
    
    W = sp.coo_matrix((Ww, (Ii, Jj)), shape=(N, N)).tocsc()
    
    # Normalize
    if W.max() > 0:
        W = W / W.max()
        
    return W

def ncut(W, nbEigenValues=8):
    N = W.shape[0]
    d = np.array(W.sum(axis=1)).flatten()
    offset = 0.5
    d = d + 2 * offset
    W = W + sp.diags(offset * np.ones(N))
    D = sp.diags(d)
    
    # D^-1/2 W D^-1/2
    d_inv_sqrt = 1.0 / np.sqrt(d + 1e-10)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_W = D_inv_sqrt @ W @ D_inv_sqrt
    
    vals, vecs = eigsh(norm_W, k=nbEigenValues, which='LA')
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    eigenvectors = D_inv_sqrt @ vecs
    for i in range(eigenvectors.shape[1]):
        v = eigenvectors[:, i]
        nrm = np.linalg.norm(v)
        if nrm > 0: eigenvectors[:, i] = (v / nrm) * np.sqrt(N)
        if eigenvectors[0, i] != 0:
             eigenvectors[:, i] *= -np.sign(eigenvectors[0, i])
    return eigenvectors, vals

def discretisation(EigenVectors):
    # Normalize rows
    rows_norm = np.linalg.norm(EigenVectors, axis=1, keepdims=True)
    rows_norm[rows_norm==0] = 1
    Y = EigenVectors / rows_norm
    k = EigenVectors.shape[1]
    
    try:
        from kmeans import kmeans
    except ImportError:
        try:
            from .kmeans import kmeans
        except ImportError:
            import kmeans
            
    # Local kmeans(X, k). Returns 1-based labels.
    labels_1based, _ = kmeans(Y, k, replicates=20)
    labels = labels_1based - 1
    
    N = Y.shape[0]
    Discrete = np.zeros((N, k))
    Discrete[np.arange(N), labels] = 1
    return Discrete, EigenVectors

def ICgraph(I, **kwargs):
    edge_res = computeEdges(I)
    W = computeW(I, emag=edge_res['emag'], ephase=edge_res['ephase'], **kwargs)
    return W, edge_res['imageEdges']

def NcutImage(I, nbSegments=10, **kwargs):
    W, imageEdges = ICgraph(I, **kwargs)
    vecs, vals = ncut(W, nbSegments)
    discrete, _ = discretisation(vecs)
    labels_flat = np.argmax(discrete, axis=1) + 1
    SegLabel = labels_flat.reshape(I.shape)
    return SegLabel, discrete, vecs, vals, W, imageEdges
