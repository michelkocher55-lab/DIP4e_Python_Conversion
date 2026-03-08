
import numpy as np

def basisImages(A, gray=0.5, space=1):
    """
    Displays the basis images of a transformation matrix.
    
    Parameters:
        A: NxN transformation matrix.
        gray: Intensity of the border (0 to 1).
        space: Spacing in pixels between basis images.
        
    Returns:
        DisplayStore: Image containing the grid of basis images.
    """
    A = np.asarray(A)
    M, N = A.shape
    
    if M < 2 or M != N or space < 0 or gray < 0 or gray > 1:
        raise ValueError('A must be square with at least 2 rows, GRAY must be between 0 and 1, and SPACE must be non-negative!')
        
    # Preallocate arrays
    # basisImages_full is used to store the seamless montage for normalization
    # but the loop logic fills 'basisImages' which is N*N x N*N
    # MATLAB: basisImages = ones(N*N);
    # Actually, the MATLAB logic computes *all* basis images then normalizes them together.
    
    # We will compute each basis image, place it in a large array, then normalize.
    # The grid is N x N images. Each image is N x N pixels.
    # Total size: N*N x N*N.
    basis_grid = np.zeros((N*N, N*N))
    
    # Prepare display structure
    # Dimensions:
    # N basis images wide + (N-1) spaces + 2 borders?
    # MATLAB: display = ones(N*N + 2*N + space*(N-1))
    # Let's trace C = N+2+space.
    # The cell size includes the basis image (N) + 2 border pixels.
    # Spacing is between cells.
    # The loop places cells at: C*(i-1)
    
    # Let's just follow the loop logic.
    C = N + 2 + space
    
    # Output can contain multiple parts (Real, Imag)
    DisplayStore = []
    
    # Check if complex
    is_complex = np.iscomplexobj(A)
    parts = [0, 1] if is_complex else [0]
    
    for part in parts:
        # Compute and scale basis images
        basis_grid_part = np.zeros((N*N, N*N))
        
        for i in range(N): # 0 to N-1
            for j in range(N): # 0 to N-1
                # Outer product of row i and row j of A
                # params i, j correspond to u, v in basis vector definition
                # S = A(i,:).T * A(j,:)
                # A[i,:] is 1D array (row).
                # outer: np.outer(A[i,:], A[j,:])
                
                # MATLAB: S = real(transpose(A(i,:))*A(j,:));
                # A(i,:) is row i. Transpose makes it Col vector.
                # Col * Row -> Matrix.
                # So it IS outer product.
                
                term = np.outer(A[i,:], A[j,:])
                
                if part == 0:
                    S = np.real(term)
                else:
                    S = np.imag(term)
                
                # Place in grid
                # MATLAB: x = N*i - N + 1 (1-based) -> N*(i-1) + 1
                # Python: x = N*i
                x_start = N * i
                y_start = N * j
                basis_grid_part[x_start:x_start+N, y_start:y_start+N] = S
                
        # Normalize the whole grid
        # mat2gray
        min_v = basis_grid_part.min()
        max_v = basis_grid_part.max()
        if max_v - min_v > 1e-10:
            basis_grid_norm = (basis_grid_part - min_v) / (max_v - min_v)
        else:
            basis_grid_norm = np.zeros_like(basis_grid_part) # Or ones?
            
        # Add borders and spacing
        # Size calculation
        # Each cell is (N + 2) size.
        # There are N cells.
        # Spacing 'space' between cells (N-1 spaces).
        # Total size = N * (N+2) + (N-1) * space
        full_size = N * (N + 2) + (N - 1) * space
        display_img = np.ones((full_size, full_size)) # background ?
        # MATLAB initializes with ones? No, explicitly logic sets border.
        # But `display` init was ones.
        
        # Fill
        for i in range(N):
            for j in range(N):
                # Coords
                r_start = C * i
                c_start = C * j
                
                # Fill cell area with border color
                # Cell size is N+2
                display_img[r_start:r_start+N+2, c_start:c_start+N+2] = gray
                
                # Fill inner with basis image
                img_chunk = basis_grid_norm[i*N:(i+1)*N, j*N:(j+1)*N]
                
                display_img[r_start+1:r_start+1+N, c_start+1:c_start+1+N] = img_chunk
                
        DisplayStore.append(display_img)
        
    # Concatenate if multiple
    if len(DisplayStore) == 1:
        return DisplayStore[0]
    else:
        # Side by side
        return np.concatenate(DisplayStore, axis=1)

