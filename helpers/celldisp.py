from typing import Any


def celldisp(c: Any, name: Any = None):
    """
    Display nested list/tuple contents (MATLAB-like celldisp).
    """
    if not isinstance(c, (list, tuple)):
        raise TypeError("celldisp: input must be a list or tuple.")

    if name is None or name == "":
        name = "ans"

    def _subs(idx: Any, shape: Any):
        """_subs."""
        # idx is flat index, shape is tuple
        if len(shape) == 2 and (shape[0] == 1 or shape[1] == 1):
            shape = (max(shape),)
        # Compute subscripts (1-based)
        subs = []
        rem = idx
        for dim in shape:
            subs.append(rem % dim + 1)
            rem //= dim
        return "{" + ",".join(str(s) for s in subs) + "}"

    # Determine shape as 1D list length
    shape = (len(c),)

    def _disp(obj: Any, prefix: Any):
        """_disp."""
        if isinstance(obj, (list, tuple)) and len(obj) > 0:
            for i, item in enumerate(obj):
                _disp(item, f"{prefix}{_subs(i, (len(obj),))}")
        else:
            print(f"{prefix} =")
            if obj is None:
                print("     []")
            elif isinstance(obj, (list, tuple)) and len(obj) == 0:
                print("     []")
            else:
                print(obj)

    for i, item in enumerate(c):
        if isinstance(item, (list, tuple)) and len(item) > 0:
            _disp(item, f"{name}{_subs(i, shape)}")
        else:
            print(f"{name}{_subs(i, shape)} =")
            if item is None:
                print("     []")
            elif isinstance(item, (list, tuple)) and len(item) == 0:
                print("     []")
            else:
                print(item)
