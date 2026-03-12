import numpy as np
from DIP4eFigures.levelSetHeavimp4e import levelSetHeavimp4e
from DIP4eFigures.levelSetCurvature4e import levelSetCurvature4e

def levelSetForce4e(type_str, paramcell, normcell=['Fu', 'Cu']):
    """
    Computes scalar force field for level-set segmentation.
    
    F = levelsetForce4e(type_str, paramcell, normcell)
    """
    
    type_str = type_str.lower()
    
    if type_str == 'binary':
        f = paramcell[0]
        a = paramcell[1]
        b = paramcell[2]
        F = a*f + b*(1 - f)
        
        if len(normcell) > 0 and normcell[0] == 'Fn':
            F = F / (np.max(np.abs(F)) + np.finfo(float).eps)
            
    elif type_str == 'gradient':
        f = paramcell[0]
        p = paramcell[1]
        lam = paramcell[2]
        
        # MATLAB gradient: [gx, gy] (x=cols, y=rows)
        # Python gradient: [gy, gx] (0=rows, 1=cols)
        gy, gx = np.gradient(f)
        fnorm = np.sqrt(gx**2 + gy**2)
        F = 1.0 / (1 + lam * (fnorm**p))
        
        if len(normcell) > 0 and normcell[0] == 'Fn':
            F = F / (np.max(np.abs(F)) + np.finfo(float).eps)
            
    elif type_str == 'geodesic':
        # Geodesic force (Eq. 11-102 in DIP4E):
        # F = -div2D(W .* (grad(phi) / ||grad(phi)||)) - c*W
        phi = paramcell[0]
        c = paramcell[1]
        W = np.array(paramcell[2], dtype=float)

        # MATLAB gradient: [phiy, phix]
        phiy, phix = np.gradient(phi)
        phinorm = np.sqrt(phix**2 + phiy**2)

        phixN = np.zeros_like(phix)
        phiyN = np.zeros_like(phiy)
        mask = phinorm > 0
        phixN[mask] = phix[mask] / phinorm[mask]
        phiyN[mask] = phiy[mask] / phinorm[mask]

        Fx = W * phixN
        Fy = W * phiyN

        # div2D(Fx, Fy) with x=cols, y=rows:
        # div = dFx/dx + dFy/dy
        dFx_dy, dFx_dx = np.gradient(Fx)
        dFy_dy, dFy_dx = np.gradient(Fy)
        div = dFx_dx + dFy_dy

        F = -div - c * W

        if len(normcell) > 0 and normcell[0] == 'Fn':
            F = F / (np.max(np.abs(F)) + np.finfo(float).eps)
        
    elif type_str == 'regioncurve':
        # Check param length
        if len(paramcell) == 3:
            f = paramcell[0]
            phi = paramcell[1]
            mu = paramcell[2]
            nu = 0
            lambda1 = 1
            lambda2 = 1
        else:
            f = paramcell[0]
            phi = paramcell[1]
            mu = paramcell[2]
            nu = paramcell[3]
            lambda1 = paramcell[4]
            lambda2 = paramcell[5]
            
        # Compute c1 (inside) and c2 (outside)
        # MATLAB: idxIn = find(phi >= 0). 
        # Note: Original CV paper has inside < 0. MATLAB implementation comments say:
        # "In original algo, phi on or inside is positive... opposite... we follow original and then reconcile sign."
        # Wait, MATLAB code says: `idxIn = find(phi >= 0)`.
        # And later `F = -F`.
        # If standard convention (my levelsetFunction) is phi < 0 inside.
        # Then `(phi >= 0)` selects OUTSIDE (or background).
        # Let's align with the goal: Chan-Vese minimizes variance inside vs outside.
        # If I use `phi < 0` for Inside.
        
        # Let's verify MATLAB logic again:
        # "phi values on or inside the contour are defined as positive" (Original Algo convention).
        # "This is the opposite of ours" (ours is neg inside).
        # Code: `idxIn = find(phi >= 0)`. So it computes c1 over the POSITIVE region.
        # `idxIn` -> Region P.
        # `idxOut` -> Region N.
        # `c1` -> Mean of P. `c2` -> Mean of N.
        # `F = ... - lambda1*(f - c1)^2 + lambda2*(f - c2)^2`.
        # This force F drives the boundary towards minimizing fitting error.
        # Finally `F = -F`.
        
        # My implementation:
        # I use `phi < 0` as Inside (Region I). `phi >= 0` as Outside (Region O).
        # Chan Vese Energy: lambda1 * int_I (f-c1)^2 + lambda2 * int_O (f-c2)^2.
        # Euler Lagrange Force term (for dPhi/dt = ... - F):
        # - lambda1(f-c1)^2 + lambda2(f-c2)^2. (If phi becomes more positive, we go outside).
        
        HS, _ = levelSetHeavimp4e(phi, 1)
        # HS is approx 1 where phi > 0 (Outside), 0 where phi < 0 (Inside).
        
        # c_outside (corresponds to HS=1)
        # c_inside (corresponds to HS=0)
        
        EPS = np.finfo(float).eps
        
        # Calculate means
        # c1 (Inside, phi<0). (HS is 0). (1-HS is 1).
        # Wait. MATLAB `HS = 0.5*(1 + atan(phi))`. phi>0 -> HS->1.
        # So sum(f * HS) is sum over OUTSIDE.
        # MATLAB calculates `c1 = sum(f*HS)`. So c1 is OUTSIDE mean.
        # `c2 = sum(f*(1-HS))`. So c2 is INSIDE mean.
        
        # Formula: `F = ... - lambda1*(f - c1)^2 + lambda2*(f - c2)^2`.
        # Note indices: 1 goes with c1 (Outside). 2 goes with c2 (Inside).
        
        c1 = np.sum(f * HS) / (np.sum(HS) + EPS) # Outside Mean
        c2 = np.sum(f * (1 - HS)) / (np.sum(1 - HS) + EPS) # Inside Mean
        
        # Curvature
        mode_curv = 'Cu'
        if len(normcell) == 2:
            mode_curv = normcell[1]
            
        V = levelSetCurvature4e(phi, mode_curv)
        
        # Force
        # MATLAB: F = mu*V + nu - lambda1*(f - c1).^2 + lambda2*(f - c2).^2;
        F = mu * V + nu - lambda1 * (f - c1)**2 + lambda2 * (f - c2)**2
        
        # "Change sign because phi in original paper is negative of ours"
        # If 'ours' (MATLAB) is Neg Inside.
        # Original (Chan Vese) is Pos Inside.
        # If we computed F assuming Pos Inside convention (but using our phi), 
        # and our c1 was 'High Phi' (Outside) and c2 was 'Low Phi' (Inside).
        # Then `F` driving expansion of High Phi region...
        # I'll trust the MATLAB `F = -F` inversion to align math with the convention used.
        F = -F
        
        # Normalize
        if len(normcell) > 0 and normcell[0] == 'Fn':
            F = F / (np.max(np.abs(F)) + EPS)
            
    else:
        # Default or unknown
        F = np.zeros_like(paramcell[0])
        
    return F
