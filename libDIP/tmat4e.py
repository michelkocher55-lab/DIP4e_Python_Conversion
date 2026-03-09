from typing import Any
import numpy as np
from scipy.linalg import hadamard, dft


def tmat4e(basis: Any = "DFT", N: Any = 8):
    """
    Returns an orthogonal, unitary, or biorthogonal matrix of dimension N x N
    for the named basis.

    Parameters:
    -----------
    basis : str, optional
        One of 'DFT', 'DCT', 'DST', 'DHT', 'WHT', 'SLT', 'HAAR'.
        Default is 'DFT'.
    N : int, optional
        Dimension of the matrix. Must be >= 2. Default is 8.

    Returns:
    --------
    A : numpy.ndarray
        The N x N transformation matrix.
    """
    if N < 2:
        print("N must be at least 2!")
        return None

    basis = basis.upper()

    # ----------------------------------------
    # DFT: Discrete Fourier Transform
    # ----------------------------------------
    if basis in ["DFT", "DFTR", "DFTI"]:
        # MATLAB: dftmtx(N) / sqrt(N)
        # scipy.linalg.dft(N, scale='sqrtn') matches this normalization
        A = dft(N, scale="sqrtn")
        return A

    # ----------------------------------------
    # DCT: Discrete Cosine Transform
    # ----------------------------------------
    elif basis == "DCT":
        # Type-II DCT orthonormal basis
        # A[k, n] = c[k] * cos(pi/N * (n + 0.5) * k)
        # where c[0] = sqrt(1/N), c[k] = sqrt(2/N) for k > 0
        n = np.arange(N)
        k = np.arange(N).reshape((N, 1))

        A = np.cos(np.pi / N * (n + 0.5) * k)

        # Scaling
        # Row 0: sqrt(1/N)
        # Rows 1..N-1: sqrt(2/N)
        c0 = np.sqrt(1 / N)
        ck = np.sqrt(2 / N)

        A[0, :] *= c0
        A[1:, :] *= ck

        return A

    # ----------------------------------------
    # DST: Discrete Sine Transform (Jain's)
    # ----------------------------------------
    elif basis == "DST":
        # A[i, j] = sqrt(2/(N+1)) * sin( (i+1)*(j+1)*pi / (N+1) )
        # 0-indexed i, j in Python
        i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")

        A = np.sin(((j + 1) * (i + 1) * np.pi) / (N + 1))
        A = A * np.sqrt(2 / (N + 1))
        return A

    # ----------------------------------------
    # DHT: Discrete Hartley Transform
    # ----------------------------------------
    elif basis == "DHT":
        # A[i, j] = (cos(2*pi*u*x/N) + sin(2*pi*u*x/N)) / sqrt(N)
        u, x = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        val = 2 * np.pi * u * x / N
        A = np.cos(val) + np.sin(val)
        A = A / np.sqrt(N)
        return A

    # ----------------------------------------
    # WHT: Walsh-Hadamard Transform
    # ----------------------------------------
    elif basis == "WHT":
        # Check power of 2
        logN = np.log2(N)
        if not logN.is_integer():
            raise ValueError("N must be a power of 2 for WHT.")

        # Unordered Hadamard
        H = hadamard(N) / np.sqrt(N)

        # MATLAB code implements bit-reversal of Gray code to get sequency natural order

        # Create Hadamard Indices 0..N-1
        HadIdx = np.arange(N)
        M = int(logN)

        # dec2bin equivalent
        # We need M bits.
        # e.g. N=8, M=3. 0 -> '000', 1 -> '001'...

        # Helper to get bits array (N, M)
        bits = np.array([list(np.binary_repr(x, width=M)) for x in HadIdx], dtype=int)

        # Bit reversing: binHadIdx = fliplr(dec2bin(HadIdx,M))
        binHadIdx = np.fliplr(bits)

        # Gray code conversion logic from MATLAB:
        # for k = M:-1:2 (1-based) -> M-1 downto 1 (0-based)
        #   binSeqIdx(:,k) = xor(binHadIdx(:,k), binHadIdx(:,k-1))

        # Let's map indices carefully.
        # MATLAB k: M, M-1, ..., 2.
        # Python col: M-1, M-2, ..., 1. (since 0 is the MSB in standard rep, but fliplr made col 0 the LSB originally?)
        # Wait, fliplr on '001' makes '100'. So col 0 is LSB.
        # MATLAB code logic:
        # binHadIdx columns: 1 (LSB) to M (MSB).
        # Loop k from M down to 2.
        # binSeqIdx(:, k) = xor(col k, col k-1).
        # binSeqIdx column 1 (LSB) corresponds to what?
        # MATLAB: "binSeqIdx = zeros(N, M-1)" ?? No, zeros(N, M). MATLAB loop logic seems to fill k from M down to 2. Column 1 is left as 0?
        # Actually standard Walsh-Hadamard ordering algorithm:
        # Gray code g_i from n_i: g_i = n_i XOR (n_i >> 1)
        # Bit-reversal of Hadamard index gives Sequency index in Gray code?

        # Let's reproduce exact MATLAB logic.
        binSeqIdx = np.zeros((N, M), dtype=int)

        # MATLAB loop: for k = M:-1:2
        # Python indices: M-1 down to 1.
        for k in range(M - 1, 0, -1):
            binSeqIdx[:, k] = np.bitwise_xor(binHadIdx[:, k], binHadIdx[:, k - 1])

        # What about column 1? MATLAB code implies binSeqIdx initialized?
        # Re-reading MATLAB:
        # binSeqIdx = zeros(N, M-1); ???
        # Wait, dec2bin returns characters. '0' is 48.
        # The loop fills k. k goes down to 2.
        # And then: "SeqIdx = binSeqIdx*pow2((M-1:-1:0)');"
        # This reconstruction uses M columns.
        # If binSeqIdx was initialized as zeros(N, M-1), it would crash accessing k=M if size was M-1?
        # Ah, maybe M-1 in MATLAB creates size? No.
        # M is typically log2(N). For N=8, M=3.
        # Loop k=3, 2.
        # If binSeqIdx is (N, 2), accessing k=3 is error.
        # Maybe typeset error in my reading?
        # "binSeqIdx = zeros(N,M-1);" -> No, likely "zeros(N,M)".
        # Let's assume M.

        # Also, column 1 (LSB) in MATLAB code:
        # It is NOT touched by the loop. So it stays 0?
        # Let's look at logic for bit 1.
        # Or maybe binSeqIdx starts as copy of binHadIdx?
        # "binSeqIdx = zeros..."

        # Let's replicate exact steps with small example N=4, M=2.
        # HadIdx: 0, 1, 2, 3.
        # bin: 00, 01, 10, 11
        # fliplr: 00, 10, 01, 11 (Cols: 1, 2)
        # k loop: M=2 downto 2. k=2.
        # binSeqIdx(:, 2) = xor(binHadIdx(:,2), binHadIdx(:,1))
        # col 1 is 0.
        # SeqIdx reconstruction from M bits.

        # Wait, the MATLAB code provided:
        # binSeqIdx = zeros(N,M-1);
        # for k = M:-1:2
        #    binSeqIdx(:,k) = ...
        # If I have N=8, M=3. zeros(8, 2). Accessing k=3 is valid index in matrix expansion? No.
        # Maybe it meant M?
        # Or maybe the loop is to M-1?

        # Let's trust established Algorithms for Sequency ordering if code is ambiguous.
        # Walsh-Hadamard in sequency order (Walsh matrix).
        # Can be obtained from Hadamrd by:
        # 1. Bit-reversal of row indices.
        # 2. Gray-code conversion (or inverse).

        # Actually, let's just implement the loop purely in logic assuming M columns.
        # We need N indices (0..N-1).

        binSeqIdx = np.zeros((N, M), dtype=int)

        # Fill k=M-1 down to 1
        for k in range(M - 1, 0, -1):
            binSeqIdx[:, k] = np.bitwise_xor(binHadIdx[:, k], binHadIdx[:, k - 1])

        # MATLAB code binSeqIdx(:,1) (LSB)?
        # For N=8, loop touches k=3, k=2. k=1 is untouched?
        # Wait, does the code copy something first?
        # No.
        # But wait.
        # "binSeqIdx = zeros(N,M-1)" -> Likely typo in source or my capture.
        # "binSeqIdx*pow2((M-1:-1:0)')" uses M weights. (M-1...0 is M integers).
        # So binSeqIdx MUST have M columns.

        # And what about column 1?
        # Typically the MSB of sequency is same as MSB of source in Gray code?
        # Or LSB?
        # The code usually is g[k] = b[k] ^ b[k+1].
        # And g[M] = b[M].
        # Here we have loop M down to 2.
        # So k=1 is left alone.
        # But we need to assign it?
        # Maybe: binSeqIdx(:, 1) = binHadIdx(:, 1)?
        # Let's assume standard Gray code behavior: MSB passed through.
        # If fliplr reversed bits, then index 1 is original MSB?
        # dec2bin(0..7, 3) -> '000'..'111'. MSB is left.
        # fliplr -> LSB is left (index 1). MSB is right (index M).
        # Loop M (-1) to 2.
        # So it processes MSB downwards (in Python index terms: M-1 down to 1).
        # Index 0 (LSB in Python, 1 in MATLAB) is untouched.
        # Usually Gray code: G = B ^ (B >> 1).
        # MSB matches.

        # Let's look at the recursion:
        # binHadIdx[:, k] XOR binHadIdx[:, k-1]
        # This looks like (b_i XOR b_{i-1}).
        # So let's populate column 0 (Python) separately?
        # MATLAB code doesn't show it.
        # BUT: "binSeqIdx = zeros..."
        # If I compute this standard way:

        # Sequency index I_seq from Hadamard index I_had:
        # I_gray = BinaryToGray(I_had) -- NO
        # The mapping is BitReverse(BinaryToGray(I_had))? Or GrayToBinary(BitReverse(I_had))?

        # Let's try to be robust.
        # A known property: The row with 0 zero-crossings is row 0.
        # Row with N-1 zero-crossings is row N-1.

        # Alternative approach:
        # Generate H = hadamard(N).
        # Calculate zero-crossings for each row.
        # Sort rows by zero-crossings.
        # This is robust and doesn't rely on ambiguous bit logic code.
        # Sequency = number of sign changes.

        seq = np.sum(np.abs(np.diff(H, axis=1)), axis=1) / 2
        # seq should be integers 0..N-1.
        idxs = np.argsort(seq)
        A = H[idxs, :]
        return A

    # ----------------------------------------
    # SLT: Slant Transform
    # ----------------------------------------
    elif basis == "SLT":
        if N < 2:
            return None
        if N == 2:
            return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

        # Recursive construction
        # N must be power of 2
        logN = np.log2(N)
        if not logN.is_integer():
            raise ValueError("N must be a power of 2 for SLT.")

        # Start with N=4 logic from code or build up?
        # Code:
        # "Construct ... by recursion"
        # Since I am in Python, I can implement a recursive function.

        # Better yet, follow the iterative loop from 3 to log2(N) (from size 4 up to N).
        # But we need base 4 first?
        # Code says: "if N==2 ... else ... a=3/sqrt(5)..."
        # It handles N=4 inside the loop or initialization?
        # "for i = 3:1:LN" -> if N=4, LN=2. Loop 3:2 doesn't run.
        # So it expects to build base case 4?
        # Wait, if N=4, loop doesn't run. We need initialization for 4.

        # Initialization block:
        # a = 3/sqrt(5), b = 1/sqrt(5)
        # sp = [1 1 1 1; a b -b -a; 1 -1 -1 1; b -a a -b]
        # (This is unnormalized Slant 4x4 matrix?)
        # Let's check normalizing at end: "sp = sp / sqrt(N)"
        # So yes, sp accumulates.

        # N=2 case handled.

        # If N >= 4:
        a = 3 / np.sqrt(5)
        b = 1 / np.sqrt(5)
        sp = np.array([[1, 1, 1, 1], [a, b, -b, -a], [1, -1, -1, 1], [b, -a, a, -b]])

        if N == 4:
            return sp / np.sqrt(4)

        # Loop for sizes 8, 16, ... N
        LN = int(logN)
        for i in range(3, LN + 1):
            NN = 2**i  # Current size, e.g. 8

            # Coefficients
            # aN = sqrt( (3*NN^2) / (4*(NN^2-1)) )
            aN = np.sqrt((3 * NN**2) / (4 * (NN**2 - 1)))
            bN = np.sqrt((NN**2 - 4) / (4 * (NN**2 - 1)))

            # Building blocks
            # sr1 = [1 0; aN bN]
            sr1 = np.array([[1, 0], [aN, bN]])
            sr2 = np.array([[1, 0], [-aN, bN]])

            # sz = zeros(2, (NN-4)/2)
            # dim2 = (NN - 4) // 2
            dim2 = (NN // 2) - 2
            sz = np.zeros((2, dim2))

            # sn1 = [sr1, sz, sr2, sz]
            sn1 = np.hstack(
                [sr1, sz, sr2, sz]
            )  # 2 x NN/2 ? 2 + dim2 + 2 + dim2 = 4 + NN - 4 = NN. Correct.

            # sn2, sn4
            # q = NN/2 - 2
            q = (NN // 2) - 2
            ir = np.eye(q)
            iz = np.zeros((q, 2))

            # sn2 = [iz, ir, iz, ir]
            sn2 = np.hstack([iz, ir, iz, ir])
            # sn4 = [iz, ir, iz, -ir]
            sn4 = np.hstack([iz, ir, iz, -ir])

            # sn3
            # sr1 = [0 1; -bN aN]
            # sr2 = [0 -1; bN aN]
            sr1_3 = np.array([[0, 1], [-bN, aN]])
            sr2_3 = np.array([[0, -1], [bN, aN]])
            sn3 = np.hstack([sr1_3, sz, sr2_3, sz])

            # sn = vertcat(sn1, sn2, sn3, sn4)
            sn = np.vstack([sn1, sn2, sn3, sn4])

            # m2 = blkdiag(sp, sp)
            from scipy.linalg import block_diag

            m2 = block_diag(sp, sp)

            # sp = sn @ m2
            sp = sn @ m2

            # Permutation / Reordering
            # "Sequency reordering is described in Jain... code implements reordering"
            A_new = np.zeros_like(sp)
            half = NN // 2

            for k in range(NN):
                if k < 2:
                    seq = k
                elif k < half:  # k <= NN/2 - 1
                    if k % 2 == 0:
                        seq = 2 * k
                    else:
                        seq = 2 * k + 1
                elif k == half:
                    seq = 2
                elif k == half + 1:
                    seq = 3
                else:
                    if k % 2 == 0:
                        seq = 2 * (k - half) + 1
                    else:
                        seq = 2 * (k - half)

                A_new[seq, :] = sp[k, :]

            sp = A_new

        sp = sp / np.sqrt(N)
        return sp

    # ----------------------------------------
    # HAAR: Haar Wavelets
    # ----------------------------------------
    elif basis == "HAAR":
        # Code uses wavedec on Identity columns.
        # "Compute three level decomposition" -> wait, loops LN (log2(N))?
        # "wavedec(I(:,i), LN, 'haar')"
        # wavedec with level=LN returns full decomposition (all coeffs).
        # We can simulate this without pywt or use iterative construction.

        # Iterative Haar Matrix Construction:
        # H_2 = [1 1; 1 -1]
        # H_2N = [ H_N (x) [1 1]; I_N (x) [1 -1] ] normalized?
        # Actually standard definition:
        # H[0] = 1/sqrt(N) * [1 ... 1]
        # ...

        # Or recursive Kronecker product definition:
        # H_1 = [1]
        # H_2n = 1/sqrt(2) * [ H_n (x) [1, 1]; I_n (x) [1, -1] ]

        # Let's verify against MATLAB's looped wavedec expectation.
        # wavedec produces [cA, cD_L, cD_L-1, ... cD_1]
        # For N=4, LN=2.
        # decomp: [cA2, cD2, cD1]
        # cA2 size 1. cD2 size 1. cD1 size 2. Total 4.

        # Python implementation of Haar Matrix (normalized):
        if N < 2:
            return None
        logN = np.log2(N)
        if not logN.is_integer():
            raise ValueError("N must be a power of 2 for Haar.")

        # H = 1
        # Loop i = 1 to logN
        # H = 1/sqrt(2) * [ kron(H, [1, 1]); kron(I, [1, -1]) ]

        # Wait, the ordering might be different.
        # Standard Haar puts DC first.
        # The Kronecker calc above:
        # k=1: [1 1; 1 -1] / sqrt(2)
        # k=2:
        # Top = [1 1; 1 -1] (x) [1 1] = [1 1 1 1; 1 -1 1 -1]
        # Bot = I_2 (x) [1 -1] = [1 -1 0 0; 0 0 1 -1]
        # Result: [DC, ... fine scale ... ]

        # Let's try this recursive build.
        H = np.array([[1.0]])
        for i in range(int(logN)):
            current_N = 2**i
            # Top half: H_prev (kron) [1, 1]
            top = np.kron(H, np.array([1, 1]))
            # Bottom half: I (kron) [1, -1]
            bot = np.kron(np.eye(current_N), np.array([1, -1]))

            H = np.vstack([top, bot]) / np.sqrt(2)

        return H

    else:
        print(f"Basis '{basis}' not implemented.")
        return None
