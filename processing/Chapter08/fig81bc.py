import os
import matplotlib.pyplot as plt

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DIP4eFigures.dip import Dip
import argparse
import numpy as np


def fig81bc(part, s=256, n=None, p=None):
    """Generate Figure 8.1(b,c) source images."""
    if n is None:
        n = np.array([125, 126, 127, 129, 130, 131], dtype=np.uint8)
        p = np.array([1935, 5123, 9997, 7652, 4755, 1877], dtype=int)
        jend = 6
    else:
        n = np.asarray(n, dtype=np.uint8)
        p = np.asarray(p, dtype=int)
        if n.shape != p.shape:
            raise ValueError("n and p must be the same size!")
        jend = n.shape[0] if n.ndim == 1 else n.shape[1]

    img = np.zeros((s, s), dtype=np.uint8)

    if part == "b":
        gl = np.arange(s, dtype=np.uint8)
        for k in range(s):
            r = int(np.ceil(len(gl) * np.random.rand())) - 1
            img[k, :] = gl[r]
            gl = np.delete(gl, r)
    else:
        img[:, :] = 128
        for j in range(jend):
            for _ in range(int(p[j])):
                img[
                    int(np.ceil(s * np.random.rand())) - 1,
                    int(np.ceil(s * np.random.rand())) - 1,
                ] = n[j]

        for k in range(50, 81):
            img[:, k] = 128

        for j in range(50, 81):
            img[j, :] = 128

    return img


def main(show: bool = True) -> None:
    """Run fig81bc.py with explicit input/output paths."""
    input_data_dir = "AllDataFiles"
    output_dir = os.environ.get("DIP4E_OUTPUT_DIR", str(PROJECT_ROOT / "output"))
    output_figure_base = str(Path(output_dir) / "fig81bc")
    os.makedirs(output_dir, exist_ok=True)

    result = Dip().fig81bc(data_dir=input_data_dir)
    figures = result.get("figures", [])
    if not figures:
        return

    for idx, fig in enumerate(figures, start=1):
        suffix = "" if len(figures) == 1 else f"_{idx}"
        fig.savefig(f"{output_figure_base}{suffix}.png", dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--noshow",
        action="store_true",
        help="Disable interactive figure display.",
    )
    args = parser.parse_args()
    main(show=not args.noshow)
