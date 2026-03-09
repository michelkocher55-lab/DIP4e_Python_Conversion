from typing import Any
import numpy as np


def strsimilarity(a: Any, b: Any):
    """
    Similarity measure between two character vectors/strings.

    Computes R = alpha / (max(La, Lb) - alpha)
    where alpha is the number of matching characters at the same position,
    and La, Lb are lengths (ignoring spaces).

    Parameters
    ----------
    a, b : str
        Input strings.

    Returns
    -------
    R : float
        Similarity measure. infinite if strings are identical.
    a_out, b_out : str
        The processed strings (no spaces, padded).
    """

    # 1. Convert to simple strings (Python strings are immutable char arrays)
    if not isinstance(a, str):
        a = str(a)
    if not isinstance(b, str):
        b = str(b)

    # 2. Remove blanks
    a = a.replace(" ", "")
    b = b.replace(" ", "")

    # 3. Get lengths
    La = len(a)
    Lb = len(b)

    # 4. Pad shorter string
    # MATLAB uses 'blanks(n)' which are spaces.
    # "All blanks... are deleted, so they should not be used as valid characters...".
    # Wait, line 44: `a = a(~isspace(a))` removes blanks.
    # But line 51: `b = [b, blanks(La - Lb)]` ADDS blanks for padding!
    # These new blanks are used for comparison.
    # "All blanks... are deleted" refers to the *input*.
    # Then padding *introduces* blanks to equalize length.
    # Since inputs were stripped of blanks, the padded blanks will strictly NOT match any real character (unless original had NO chars?).
    # So padding with spaces is safe/correct per MATLAB code.

    if La > Lb:
        b = b.ljust(La, " ")
    elif Lb > La:
        a = a.ljust(Lb, " ")

    # Python strings compare directly? No, we need element-wise.
    # Convert to numpy arrays or lists for easy element-wise comparison.
    a_arr = np.array(list(a))
    b_arr = np.array(list(b))

    # 5. Compute Similarity
    # I = find(a == b)
    matches = a_arr == b_arr
    alpha = np.sum(matches)

    den = max(La, Lb) - alpha

    if den == 0:
        R = float("inf")
    else:
        R = float(alpha) / float(den)

    return R, a, b
