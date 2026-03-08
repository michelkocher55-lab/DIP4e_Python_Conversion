
import numpy as np

class FreemanChainCodeResult:
    def __init__(self):
        self.x0y0 = None
        self.fcc = None
        self.mm = None
        self.diff = None
        self.diffmm = None

def get_fcc_map():
    # Map (dx, dy) to 8-code
    # dx | dy | 8-code
    # 0    1    0
    # -1   1    1
    # -1   0    2
    # -1  -1    3
    # 0   -1    4
    # 1   -1    5
    # 1    0    6
    # 1    1    7
    
    # Using z = 4*(dx + 2) + (dy + 2)
    # dx, dy in [-1, 0, 1] -> dx+2 in [1, 2, 3] -> 4*(...) in [4, 8, 12]
    # z range: 4(1)+1=5 to 4(3)+3=15
    fcc_lut = {}
    
    mapping = {
        (0, 1): 0,
        (-1, 1): 1,
        (-1, 0): 2,
        (-1, -1): 3,
        (0, -1): 4,
        (1, -1): 5,
        (1, 0): 6,
        (1, 1): 7
    }
    
    return mapping

def minmag(c):
    """
    Finds the sequence in c that would form an integer of minimum magnitude.
    c is a list or 1D array.
    """
    c = np.array(c)
    n = len(c)
    
    # 1. Find all occurrences of min val
    min_val = np.min(c)
    starts = np.where(c == min_val)[0]
    
    # 2. Generate all candidate sequences
    candidates = []
    for s in starts:
        # roll shifts right by k. We want left shift by s (so index s becomes 0).
        # np.roll(c, -s)
        candidates.append(np.roll(c, -s))
        
    A = np.array(candidates)
    
    # 3. Sort lexicographically
    # In MATLAB, unique(A, 'rows') sorts.
    # In Python, we can just sort the rows.
    # But we want the unique minimum. 
    # Actually lexicographical sort of rows gives the minimum at index 0.
    
    # Sort
    # np.lexsort sorts by columns, effectively. But requires keys.
    # Easier to turn into list of tuples and sort?
    
    # Or just use the iterative column min approach like MATLAB for speed?
    # Python generic sort on arrays handles lexicographical comparison.
    # list.sort() on list of numpy arrays compares element by element.
    
    # Convert to list of arrays to sort
    A_list = [row for row in A]
    
    # Remove duplicates?
    # Unique rows
    unique_rows = np.unique(A, axis=0)
    
    # Sort
    # Sorting rows of unique_rows.
    # Simple sort works lexicographically.
    sorted_idx = np.lexsort(np.rot90(unique_rows)) # lexsort uses columns back to front
    # But standard list sort also works.
    
    # Let's trust list sort on tuples for robust comparisons
    rows_as_tuples = [tuple(row) for row in unique_rows]
    rows_as_tuples.sort() # Python tuple comparison is lexicographical
    
    return np.array(rows_as_tuples[0])

def codediff(fcc, conn):
    """
    Computes first difference of chain code.
    Circular difference.
    fcc: array
    conn: 4 or 8
    """
    fcc = np.array(fcc)
    
    # sr = circshift(fcc,[0,-1]); % Shift input left by one location.
    # sr is the next element. fcc[i+1] vs fcc[i].
    # MATLAB: delta = sr - fcc.
    # Python: np.roll(fcc, -1) is next element.
    sr = np.roll(fcc, -1)
    
    delta = sr - fcc
    
    # Modulo arithmetic is cleaner than the IF statement in MATLAB
    # d = (sr - fcc) % conn
    # MATLAB: If delta < 0, add conn.
    # -1 % 8 = 7.
    # So Python modulo handles it naturally.
    
    d = delta % conn
    
    return d

def coderev(fcc):
    """
    Reverses chain code direction.
    """
    # 1. Flip array left to right (reverse order)
    cr = fcc[::-1]
    
    # 2. Opposite direction (add 4 mod 8)
    # 0->4, 1->5, ... 4->0
    cr = (cr + 4) % 8
    
    return cr

def freemanChainCode(b, conn=8, dir='same'):
    """
    Computes the Freeman chain code of a boundary.
    b: Nx2 array of coordinates [x, y] or [row, col]?
       MATLAB doc says "coordinate pairs contained as rows of B".
       "using the book coordinate system with the origin on the top right"? 
       Wait. "origin on the top right (see Gig. 2.1)". 
       Usually image coords are (row, col).
       MATLAB table: dx=0, dy=1 -> 0.
       If B is [x, y] (cols, rows?).
       Usually B from bwboundaries is [row, col].
       Let's check deltas.
       Row 1 to Row 2: dr, dc.
       If B is [row, col].
       MATLAB "freemanChainCode" doc assumes standard Cartesian? Or Image?
       "book coordinate system... origin on the top LEFT" (usually).
       Let's look at the Delta Table:
       dx | dy | 8-code
        0    1    0     -> (0, 1) is Right? Yes.
       -1    1    1     -> (-1, 1) is Top-Right? (Row decreases, Col increases)
                           Wait. If x is row? No, x usually col.
       Let's assume B=[x, y].
       dx = x2 - x1. dy = y2 - y1.
       (0, 1) -> dx=0, dy=1. Right?
       (1, 1) -> Down-Right?
       
       Let's check MATLAB B.
       Standard MATLAB `bwboundaries` returns [row, col].
       If x=row, y=col.
       dx = r2-r1. dy = c2-c1.
       (0, 1) -> Same Row, Next Col -> Right. Code 0. Matches.
       (-1, 1) -> Prev Row, Next Col -> Top-Right. Code 1.
       (-1, 0) -> Prev Row, Same Col -> Top. Code 2.
       ...
       So B is interpreted as [row, col] (x=row, y=col for delta calculation purposes in the comment, but actually image (r,c)).
       
    conn: 4 or 8.
    dir: 'same' or 'reverse'.
    """
    
    b = np.array(b)
    np_points, nc = b.shape
    
    if np_points < nc: 
        raise ValueError("B must be of size np-by-2.")
        
    # Check for closed loop (first == last)
    # If so, eliminate last
    if np.array_equal(b[0], b[-1]) and len(b) > 1:
        b = b[:-1]
        np_points -= 1
        
    # Start point
    c = FreemanChainCodeResult()
    c.x0y0 = b[0]
    
    # Calculate deltas
    # a = circshift(b, [-1, 0]) -> Next points
    a = np.roll(b, -1, axis=0)
    
    DEL = a - b
    
    # Filter out zero deltas (duplicate points)
    non_zero_mask = np.any(DEL != 0, axis=1)
    DEL = DEL[non_zero_mask]
    
    if len(DEL) == 0:
        # Trivial case: single point or all duplicates
        c = FreemanChainCodeResult()
        c.fcc = np.array([])
        c.mm = np.array([])
        c.diff = np.array([])
        c.diffmm = np.array([])
        return c
    
    # Check connectivity
    # If any delta > 1, broken.
    if np.any(np.abs(DEL) > 1):
        raise ValueError("The input curve is broken or points are out of order.")
        
    # Compute Code
    # Use mapping
    # DEL[:, 0] is d(row), DEL[:, 1] is d(col).
    # d_row | d_col | Code
    #   0       1      0 (Right)
    #  -1       1      1 (TR)
    #  -1       0      2 (Top)
    #  ...
    
    # Logic: z = 4*(dx + 2) + (dy + 2)
    # Note: MATLAB code uses this with custom lookup C.
    # C(11)=0, C(7)=1...
    # Let's replicate this exact mapping just to be safe.
    
    # MATLAB Formula: z = 4*(DEL(:,1) + 2) + (DEL(:,2) + 2);
    # Here DEL(:,1) is row_delta?
    # Let's verify Table vs Formula.
    # Table: dx=0, dy=1 -> 0.
    # Formula: 4*(0+2) + (1+2) = 4*2 + 3 = 11. C(11) = 0. Matches.
    # Table: dx=-1, dy=1 -> 1.
    # Formula: 4*(-1+2) + (1+2) = 4*1 + 3 = 7. C(7) = 1. Matches.
    
    # So we can use the formula and LUT.
    
    # Lookup Table (indices 0 to 20 to be safe)
    # Actually Python dict is easier.
    C_map = {
        11: 0,
        7: 1,
        6: 2,
        5: 3,
        9: 4,
        13: 5,
        14: 6,
        15: 7
    }
    
    fcc = []
    dx = DEL[:, 0]
    dy = DEL[:, 1]
    
    z_vals = 4 * (dx + 2) + (dy + 2)
    
    for z in z_vals:
        if z in C_map:
            fcc.append(C_map[z])
        else:
            raise ValueError(f"Invalid delta found: z={z}")
            
    fcc = np.array(fcc)
    
    # Direction
    if dir == 'reverse':
        fcc = coderev(fcc)
        
    # Connectivity 4 check
    if conn == 4:
        # Check if only 0, 2, 4, 6 exist
        valid_4 = [0, 2, 4, 6]
        if not np.all(np.isin(fcc, valid_4)):
            raise ValueError("The code is not of conn = 4.")
        # Divide by 2 to get 0, 1, 2, 3
        fcc = fcc // 2
        
    c.fcc = fcc
    
    # Min Mag
    c.mm = minmag(fcc)
    
    # Diff
    c.diff = codediff(fcc, conn)
    
    # Diff Min Mag
    c.diffmm = minmag(c.diff)
    
    return c

def test_freeman():
    # Test rectangle: (0,0)->(0,1)->(1,1)->(1,0)->(0,0)
    # Row, Col
    B = np.array([
        [0, 0],
        [0, 1], # dx=0, dy=1 -> 0
        [1, 1], # dx=1, dy=0 -> 6 (Down)
        [1, 0], # dx=0, dy=-1 -> 4 (Left)
        [0, 0]  # dx=-1, dy=0 -> 2 (Up)
    ])
    # Expected Code: 0, 6, 4, 2
    
    print("Testing Freeman Chain Code...")
    res = freemanChainCode(B, conn=8)
    print("FCC:", res.fcc)
    assert np.array_equal(res.fcc, [0, 6, 4, 2])
    
    # MinMag: 0246
    print("MinMag:", res.mm)
    assert np.array_equal(res.mm, [0, 6, 4, 2])
    # Is [0 2 4 6] possible? Not by rotation.
    # [0 6 4 2] -> [6 4 2 0] -> [4 2 0 6] -> [2 0 6 4]
    # Smallest is [0 6 4 2].
    
    # Diff: 
    # 0->6: (6-0)%8 = 6
    # 6->4: (4-6)%8 = -2%8 = 6
    # 4->2: (2-4)%8 = 6
    # 2->0: (0-2)%8 = 6
    # Diff: [6, 6, 6, 6]
    print("Diff:", res.diff)
    
    # Reverse test
    res_rev = freemanChainCode(B, conn=8, dir='reverse')
    print("Rev FCC:", res_rev.fcc)
    # Reverse of [0 6 4 2]
    # Flip: [2 4 6 0]
    # Add 4: [6 0 2 4]
    assert np.array_equal(res_rev.fcc, [6, 0, 2, 4]) # Matches reverse path
    
    pass

if __name__ == "__main__":
    test_freeman()
