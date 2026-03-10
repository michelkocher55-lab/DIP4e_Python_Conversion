import os
import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libDIP.dip import Dip
import argparse


def main(show: bool = True) -> None:
    """Run Figure 8.50 caller using Dip processing and local plotting."""
    input_data_dir = "AllDataFiles"
    input_image_filename = "lena.tif"
    output_dir = str(PROJECT_ROOT / "output")
    output_figure_path = str(Path(output_dir) / "Figure850.png")
    os.makedirs(output_dir, exist_ok=True)

    result = Dip().figure850(image_name=input_image_filename, data_dir=input_data_dir)

    fig = plt.figure(1, figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(result["g1"], cmap="gray")
    plt.title(f"Watermarked, r = {result['r1']:.3g}")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(result["diff1"], cmap="gray")
    plt.title(f"Difference (Max) = {np.max(np.abs(result['diff1'])):.6g}")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(result["g2"], cmap="gray")
    plt.title(f"Watermarked, r = {result['r2']:.3g}")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(result["diff2"], cmap="gray")
    plt.title(f"Difference (Max) = {np.max(np.abs(result['diff2'])):.6g}")
    plt.axis("off")

    plt.tight_layout()
    fig.savefig(output_figure_path, dpi=150, bbox_inches="tight")
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
