from typing import Any


def N8(p: Any, q: Any, shape: Any):
    """
    Computes the 8 neighbors of linear index p in a clockwise manner.
    The list is rotated such that the first point is q.

    Parameters:
    - p: Linear index of the center pixel.
    - q: Linear index of the starting neighbor.
    - shape: Tuple (rows, cols) of the grid dimensions or int (width).
             Note: MATLAB version took NRows because MATLAB is column-major.
             Python is row-major, so this function relies on the number of columns (width).
             If an int is passed, it is assumed to be the Width.
             If a tuple (H, W) is passed, W is used.

    Returns:
    - N: List of 8 linear neighbor indices.
    """

    if isinstance(shape, tuple):
        w = shape[1]
    else:
        w = shape

    # Python Row-Major Offsets (Clockwise starting from Left)
    # indices: r*w + c
    # Left:  (r, c-1)   -> p - 1
    # TL:    (r-1, c-1) -> p - w - 1
    # Top:   (r-1, c)   -> p - w
    # TR:    (r-1, c+1) -> p - w + 1
    # Right: (r, c+1)   -> p + 1
    # BR:    (r+1, c+1) -> p + w + 1
    # Bottom:(r+1, c)   -> p + w
    # BL:    (r+1, c-1) -> p + w - 1

    neighbors = [
        p - 1,  # L
        p - w - 1,  # TL
        p - w,  # T
        p - w + 1,  # TR
        p + 1,  # R
        p + w + 1,  # BR
        p + w,  # B
        p + w - 1,  # BL
    ]

    # In MATLAB, N8 used: [L, TL, T, TR, R, BR, B, BL] ordering (assuming p-NRows is L).
    # So this order matches.

    # Rotate until q is first
    try:
        idx = neighbors.index(q)
    except ValueError:
        raise ValueError(f"q ({q}) is not an 8-neighbor of p ({p}) with width {w}")

    # Rotate list so idx becomes 0
    # neighbors = neighbors[idx:] + neighbors[:idx]

    # MATLAB: while N(1) ~= q: circshift(N, 1)
    # circshift(N, 1) shifts [1, 2, 3] to [3, 1, 2] (Right shift).
    # It shifts elements "down" the column.
    # So it rotates such that the previous element becomes first.
    # It cycles the list.

    # Actually, let's look at MATLAB code:
    # while (N(1)~=q) N = circshift (N, 1); end;
    # If q is at index 2 (3rd pos), it shifts until it's at index 0.
    # Wait. circshift(N, 1) moves end to start.
    # If N = [a, b, c] and q=b.
    # shift 1: [c, a, b]. N(1)=c != q.
    # shift 2: [b, c, a]. N(1)=b == q. Stop.
    # Result: [b, c, a].
    # This is equivalent to rolling until q is first.
    # In Python: Use deque or slicing.

    N = neighbors[idx:] + neighbors[:idx]

    return N


def test_N8():
    """test_N8."""
    # Test on a 3x3 grid (w=3)
    # 0 1 2
    # 3 4 5
    # 6 7 8
    # Center p=4.
    # Neighbors:
    # L=3, TL=0, T=1, TR=2, R=5, BR=8, B=7, BL=6.
    # List: [3, 0, 1, 2, 5, 8, 7, 6]

    p = 4
    w = 3

    # Case 1: q = 3 (Already first)
    res = N8(p, 3, w)
    assert res[0] == 3
    assert res == [3, 0, 1, 2, 5, 8, 7, 6]

    # Case 2: q = 1 (Top)
    # Expected rotation: circshift moves end to start.
    # MATLAB: moves [3,0,1...] to [6,3,0,1...] -> [7,6,3,0,1...]...
    # Wait.
    # If MATLAB loop shifts right until first element is q.
    # Python list.index(q) gives position.
    # If q is at index 2 (val=1).
    # MATLAB:
    # [3 0 1 ...]. q=1.
    # Shift 1: [6 3 0 1 ...]. 1st=6.
    # Shift 2: [7 6 3 0 1 ...].
    # ...
    # This means the order is REVERSED? Or just cycled differently?
    # If I want [q, next_cw, next_cw...] or [q, prev_cw...]?

    # circshift(N, 1) rotates [0, 1, 2] -> [2, 0, 1].
    # If my list is ordered [L, TL, T...], i.e. CW.
    # And I want result starting at q.
    # If I slice N[idx:] + N[:idx], I preserve the CW order: q, next_cw, etc.
    # Does MATLAB preserve CW order?
    # circshift preserves relative order, just changes start.
    # Yes.
    # BUT, circshift(N, 1) shifts elements to the RIGHT (or down).
    # [a, b, c] -> [c, a, b].
    # If we repeat this, we are effectively scanning the array backwards?
    # No, the array content rotates.
    # The result N will start with q. The sequence following q will be the same relative sequence as before (with wrapping).
    # Example: N=[a, b, c]. Target b.
    # Shift 1: [c, a, b].
    # Shift 2: [b, c, a].
    # Result starts with b. Followed by c, a.
    # Original: a, b, c.
    # Result: b, c, a.
    # This preserves the cyclic order a->b->c->a.
    # So Python slicing `neighbors[idx:] + neighbors[:idx]` produces `[b, c, a]`.
    # It is exactly the same.

    res = N8(p, 1, w)
    assert res[0] == 1
    assert res == [1, 2, 5, 8, 7, 6, 3, 0]

    print("Test N8 Passed.")


if __name__ == "__main__":
    test_N8()
