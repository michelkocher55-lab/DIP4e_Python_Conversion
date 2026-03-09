import os
import matplotlib.pyplot as plt

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libDIP.dip import Dip
import argparse


def main(show: bool = True) -> None:
    """Run Figure81.py with explicit input/output paths."""
    input_data_dir = "AllDataFiles"
    output_dir = str(PROJECT_ROOT / "output")
    output_figure_base = str(Path(output_dir) / "Figure81")
    os.makedirs(output_dir, exist_ok=True)

    result = Dip().figure81(data_dir=input_data_dir)
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
