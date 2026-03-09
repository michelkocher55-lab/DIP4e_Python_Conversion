from typing import Any
import numpy as np

from libDIPUM.huffman1 import huffman1


def _iter_bits(code_field: Any, pad: Any = 0):
    """_iter_bits."""
    if isinstance(code_field, (bytes, bytearray)):
        total = 8 * len(code_field) - int(pad)
        seen = 0
        for bv in code_field:
            for i in range(7, -1, -1):
                if seen >= total:
                    return
                yield (bv >> i) & 1
                seen += 1
        return

    words = np.asarray(code_field).reshape(-1)
    total = 16 * len(words) - int(pad)
    seen = 0
    for w in words:
        w = int(w)
        for i in range(15, -1, -1):
            if seen >= total:
                return
            yield (w >> i) & 1
            seen += 1


def _build_trie(map_codes: Any):
    """_build_trie."""
    root = {}
    for sym, code in enumerate(map_codes, start=1):
        node = root
        for ch in code:
            node = node.setdefault(ch, {})
        node["_sym"] = sym
    return root


def _decode_with_trie(code_field: Any, trie: Any, nvals: Any, pad: Any = 0):
    """_decode_with_trie."""
    out = np.zeros(nvals, dtype=float)
    k = 0
    node = trie

    for b in _iter_bits(code_field, pad=pad):
        node = node.get("1" if b else "0")
        if node is None:
            raise ValueError("Invalid Huffman bitstream.")

        sym = node.get("_sym")
        if sym is not None:
            out[k] = sym
            k += 1
            if k >= nvals:
                break
            node = trie

    if k < nvals:
        raise ValueError("Decoded symbol count does not match target matrix size.")

    return out


def huff2mat1(y: Any):
    """Faster alternative to huff2mat for local use in Figure834 path."""
    if (
        not isinstance(y, dict)
        or "min" not in y
        or "size" not in y
        or "hist" not in y
        or "code" not in y
    ):
        raise ValueError("The input must be a structure as returned by MAT2HUFF.")

    sz = np.asarray(y["size"]).reshape(-1).astype(float)
    m = int(sz[0])
    n = int(sz[1])

    xmin = float(np.asarray(y["min"]).reshape(-1)[0]) - 32768.0
    map_codes = huffman1(np.asarray(y["hist"]).reshape(-1).astype(float))

    trie = _build_trie(map_codes)
    x = _decode_with_trie(y["code"], trie, m * n, pad=int(y.get("pad", 0)))

    x = x + xmin - 1.0
    x = np.asarray(x, dtype=float).reshape((m, n), order="F")
    return x
