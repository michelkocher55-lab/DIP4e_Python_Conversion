from typing import Any
import heapq
import numpy as np


def huffman1(p: Any, return_tree: Any = False):
    """
    Faster Huffman code builder with deterministic tie handling.
    API-compatible alternative to huffman().
    """
    p = np.asarray(p)
    if p.ndim == 2 and min(p.shape) == 1:
        p = p.reshape(-1)
    elif p.ndim != 1:
        raise ValueError("P must be a real numeric vector.")

    if not np.isrealobj(p) or not np.issubdtype(p.dtype, np.number):
        raise ValueError("P must be a real numeric vector.")

    n = int(p.size)
    if n == 0:
        return ([], []) if return_tree else []

    code = [""] * n

    if n > 1:
        ps = p.astype(float)
        ssum = np.sum(ps)
        if ssum <= 0:
            raise ValueError("Sum of probabilities must be positive.")
        ps = ps / ssum

        tree = _reduce_heap(ps)
        _makecode_iterative(tree, code)
        s = tree
    else:
        code = ["1"]
        s = [1]

    if return_tree:
        return code, s
    return code


def _reduce_heap(p: Any):
    """_reduce_heap."""
    heap = []
    order = 0
    for k, pk in enumerate(np.asarray(p).reshape(-1), start=1):
        heapq.heappush(heap, (float(pk), order, k))
        order += 1

    while len(heap) > 2:
        p1, _, n1 = heapq.heappop(heap)
        p2, _, n2 = heapq.heappop(heap)
        node = [n1, n2]
        heapq.heappush(heap, (p1 + p2, order, node))
        order += 1

    _, _, n1 = heapq.heappop(heap)
    _, _, n2 = heapq.heappop(heap)
    return [n1, n2]


def _makecode_iterative(root: Any, code: Any):
    """_makecode_iterative."""
    stack = [(root, [])]
    while stack:
        node, prefix = stack.pop()
        if isinstance(node, list):
            stack.append((node[1], prefix + [1]))
            stack.append((node[0], prefix + [0]))
        else:
            idx = int(node) - 1
            code[idx] = "".join("1" if b else "0" for b in prefix)
