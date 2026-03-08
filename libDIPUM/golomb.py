import numpy as np


def _map(n):
    """Compute forward interleave index (MATLAB map)."""
    if n >= 0:
        return 2 * n
    return -2 * n + 1


def _imap(m):
    """Compute inverse interleave index (MATLAB imap)."""
    odd = m - 2 * (m // 2)
    if odd == 0:
        return m // 2
    return -1 * (m // 2) - 1


def golomb(x, m):
    """
    Compute the Golomb-coded size estimate for input array x.

    If m < 0, uses zero-order exponential-Golomb coding.

    Parameters
    ----------
    x : array_like
        Input values.
    m : int
        Golomb parameter. If m < 0, use exponential-Golomb mode.

    Returns
    -------
    z : ndarray
        Histogram values after optional interleaving.
    zx : ndarray
        Unit-width bins corresponding to z.
    cr : float
        Compression ratio estimate (8 * total_samples / coded_bits).
    """
    x = np.rint(np.asarray(x, dtype=float))
    xmin = int(np.min(x))
    xmax = int(np.max(x))

    # Histogram with unit-width bins from xmin to xmax.
    x = x.ravel()
    hx = np.linspace(xmin, xmax, xmax - xmin + 1)
    edges = np.arange(xmin, xmax + 2)
    h, _ = np.histogram(x, bins=edges)

    # If all nonnegative, no interleaving needed.
    if xmin >= 0:
        zx = hx.astype(float)
        z = h.astype(float)
        zlen = len(z)
    else:
        zlen = _map(xmin) + 1
        if zlen < _map(xmax) + 1:
            zlen = _map(xmax) + 1

        z = np.zeros(zlen, dtype=float)
        zx = np.zeros(zlen, dtype=float)

        for i in range(1, zlen + 1):
            zx[i - 1] = _imap(i - 1)

        # MATLAB uses exact equality on integer-valued support.
        for i in range(1, zlen + 1):
            target = _imap(i - 1)
            w = np.where(hx == target)[0]
            if len(w) == 1:
                z[i - 1] = h[w[0]]

    # Compute number of bits for Golomb / exponential-Golomb code.
    total = np.sum(h)
    bits = 0.0

    if m >= 0:
        if m == 0:
            raise ValueError("m must be > 0 for Golomb coding mode.")

        k = int(np.ceil(np.log(m) / np.log(2)))
        bound = (2 ** k) - m

        for i in range(1, zlen + 1):
            bit = np.floor((i - 1) / m) + 1 + k
            if (i - 1) - m * np.floor((i - 1) / m) < bound:
                bit = bit - 1
            bits = bits + z[i - 1] * bit
    else:
        for i in range(1, zlen + 1):
            bit = 2 * np.floor(np.log(i) / np.log(2)) + 1
            bits = bits + z[i - 1] * bit

    cr = (8 * total / bits) if bits > 0 else np.inf
    return z, zx, cr
