from typing import Any
import numpy as np
from lib.tmat4e import tmat4e
from skimage.util import view_as_blocks


def blockTransformTest4e(f: Any, type_transform: Any, blockSize: Any, truncate: Any):
    """
    Compute 'type_transform' block transforms of image F with blocks of size BLOCKSIZE x
    BLOCKSIZE, truncate TRUNCATE % of the smallest magnitude coefficients,
    and return the inverse block transform of the result.

    Parameters:
    -----------
    f : numpy.ndarray
        Input image.
    type_transform : str
        Transform type (e.g., 'DCT', 'DFT', 'WHT', etc.).
    blockSize : int
        Size of the blocks (blockSize x blockSize).
    truncate : float
        Proportion (0 to 1, or percent?) MATLAB code implies logic "z".
        MATLAB: y(index(1:uint32(z*s*s))) = 0;
        So truncate is a fraction [0, 1].

    Returns:
    --------
    g : numpy.ndarray
        Processed image (uint8).
    """

    f = f.astype(float)
    rows, cols = f.shape

    # Pad if necessary for view_as_blocks
    # This isn't in original MATLAB but view_as_blocks requires perfect division.
    # MATLAB's blkproc automatically zero-pads if needed? Or error?
    # blkproc usually pads with zeros if 'mb' and 'nb' don't divide size?
    # Actually blkproc documentation says:
    # "If the image size is not a multiple of the block size, blkproc pads the image with zeros."

    pad_r = (blockSize - rows % blockSize) % blockSize
    pad_c = (blockSize - cols % blockSize) % blockSize

    if pad_r > 0 or pad_c > 0:
        f_padded = np.pad(
            f, ((0, pad_r), (0, pad_c)), mode="constant", constant_values=0
        )
    else:
        f_padded = f

    padded_rows, padded_cols = f_padded.shape

    # Get Transform Matrix
    # MATLAB: A = tmat4e(type, blockSize);
    A = tmat4e(type_transform, blockSize)
    if A is None:
        raise ValueError("Invalid transform type or block size.")

    # MATLAB: CA = conj(A);
    CA = np.conj(A)

    # ---------------------------------------------------------------------
    # Step 1: Forward Transform
    # ---------------------------------------------------------------------
    # MATLAB: f = blkproc(f, [blockSize blockSize], 'P1 * x * P2', A, A.');
    # P1 = A, P2 = A.' (transpose)
    # Operation: A * block * A.T

    # Let's perform this manually on blocks
    # view_as_blocks returns (n_blocks_row, n_blocks_col, r, c)
    blocks = view_as_blocks(f_padded, block_shape=(blockSize, blockSize))
    n_br, n_bc, r, c = blocks.shape

    # Reshape for efficient multiplication? Or iterate?
    # Vectorized approach:
    # We want A @ block @ A.T for each block.
    # blocks shape: (N, M, B, B)
    # Einstein summation is easiest.
    # A is (B, B).
    # Result[i, j, :, :] = A @ blocks[i, j, :, :] @ A.T

    A_T = A.T

    # Using matmul with broadcasting?
    # blocks can be treated as stack of matrices if we reshape to (N*M, B, B)
    blocks_flat = blocks.reshape(-1, blockSize, blockSize)

    # T = A @ block @ A.T
    # T = A @ (block @ A.T)
    # or (A @ block) @ A.T

    # transformed_blocks = A @ blocks_flat @ A.T
    transformed_blocks = A @ blocks_flat @ A_T

    # ---------------------------------------------------------------------
    # Step 2: Truncation
    # ---------------------------------------------------------------------
    # MATLAB: f = blkproc(f, [blockSize blockSize], @cut, truncate, blockSize);
    # cut function sorts abs(y), sets smallest z*s*s to 0.

    # Iterate over transformed blocks to apply mask
    # Logic: sort absolute values, find threshold, mask.

    # Flatten each block to 1D array of coeffs
    coeffs = transformed_blocks.reshape(-1, blockSize * blockSize)

    # Sort by magnitude to find indices
    abs_coeffs = np.abs(coeffs)
    sorted_indices = np.argsort(abs_coeffs, axis=1)  # Sorts each row

    # Number of coefficients to zero out
    num_cut = int(truncate * blockSize * blockSize)

    if num_cut > 0:
        # We need to set the smallest num_cut values to 0.
        # sorted_indices[:, :num_cut] are the indices of the small values.

        # Advanced indexing to set values.
        # Create row indices: [[0,0...], [1,1...]]
        row_indices = np.arange(coeffs.shape[0])[:, np.newaxis]

        # Select the indices to cut
        cut_indices = sorted_indices[:, :num_cut]

        # Zero them out in the coeffs array
        # Note: we must modify the original complex/float array, not the abs one.
        coeffs[row_indices, cut_indices] = 0

    # Reshape back to blocks
    truncated_blocks = coeffs.reshape(-1, blockSize, blockSize)

    # ---------------------------------------------------------------------
    # Step 3: Inverse Transform
    # ---------------------------------------------------------------------
    # MATLAB: f = blkproc(f, [blockSize blockSize], 'P1 * x * P2', CA.', CA);
    # P1 = CA.' = conj(A).T = A.conj().T = A.H (Hermitian transpose)
    # P2 = CA = conj(A)
    # Inverse: A.H @ block @ A.conj()
    # Note: If A is unitary, A.H = inv(A).
    # The Forward was A * x * A.T.
    # If x was image, T = A x A.T
    # To recover x: A.inv * T * (A.T).inv
    # Since A is unitary/orthogonal, A.inv = A.H (conj transpose).
    # So x = A.H * T * (A.T).H = A.H * T * A.conj()
    # This matches the MATLAB command: CA.' (A.H) * x * CA (A.conj())

    CA_T = CA.T  # This is simple transpose of conj(A)?
    # MATLAB A.' is non-conjugate transpose.
    # CA = conj(A).
    # CA.' = conj(A).T = A.H.
    # Correct.

    # reconstructed_blocks = A.H @ T @ A.conj()
    reconstructed_blocks = CA.T @ truncated_blocks @ CA

    # ---------------------------------------------------------------------
    # Reassemble Image
    # ---------------------------------------------------------------------
    # Put blocks back together
    reconstructed_padded = np.zeros_like(
        f_padded, dtype=complex
    )  # Transforms might be complex

    # Reshape blocks back to 4D
    rec_blocks_4d = reconstructed_blocks.reshape(n_br, n_bc, blockSize, blockSize)

    # Place them into the image
    # Inverse of view_as_blocks isn't direct in skimage < 0.18 maybe?
    # But usually we can simply assign.
    # Actually, view_as_blocks returns a view, but we created a new array "transformed_blocks".
    # We need to reconstruct manually.

    for i in range(n_br):
        for j in range(n_bc):
            r_start = i * blockSize
            c_start = j * blockSize
            reconstructed_padded[
                r_start : r_start + blockSize, c_start : c_start + blockSize
            ] = rec_blocks_4d[i, j]

    # Crop padding
    g_complex = reconstructed_padded[:rows, :cols]

    # MATLAB: g = uint8(abs(f));
    g = np.abs(g_complex)
    g = g.astype(np.uint8)

    return g
