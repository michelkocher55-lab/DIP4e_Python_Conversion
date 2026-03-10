import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--noshow",
        action="store_true",
        help="Disable interactive figure display.",
    )
    return parser.parse_args()


def run_dip_method(
    method_name: str,
    output_name: str,
    *,
    show: bool = True,
    chapter_attr: str | None = None,
) -> None:
    from libDIP.dip import Dip

    input_data_dir = "AllDataFiles"
    output_dir = PROJECT_ROOT / "output"
    output_figure_base = output_dir / output_name
    os.makedirs(output_dir, exist_ok=True)

    dip = Dip()
    target = getattr(dip, chapter_attr) if chapter_attr else dip
    result = getattr(target, method_name)(data_dir=input_data_dir)
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
