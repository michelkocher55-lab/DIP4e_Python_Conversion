from typing import Any
import argparse
import sys
from pathlib import Path

import numpy as np
from skimage.io import imread


# Add project root so "libDIPUM" imports work when running from Chapter8.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libDIPUM.cv2tifs import cv2tifs
from libDIPUM.tifs2cv import tifs2cv


def _load_frames(path: Any):
    """_load_frames."""
    arr = np.asarray(imread(path))
    if arr.ndim == 2:
        return [arr]
    if arr.ndim == 3:
        # Expected grayscale stack shape: (frames, rows, cols)
        if arr.shape[-1] in (3, 4) and arr.shape[0] != arr.shape[1]:
            raise ValueError("Expected a grayscale TIFF sequence, got color data.")
        return [arr[i, :, :] for i in range(arr.shape[0])]
    raise ValueError("Unsupported TIFF dimensions.")


def _rmse(a: Any, b: Any):
    """_rmse."""
    d = a.astype(float) - b.astype(float)
    return float(np.sqrt(np.mean(d * d)))


def run_roundtrip(input_tif: Any, output_tif: Any, m: Any, d: Any, q: Any):
    """run_roundtrip."""
    if not Path(input_tif).exists():
        raise FileNotFoundError(
            f"Input TIFF not found: {input_tif}\n"
            "Provide --input, or place your file at the default path."
        )

    original = _load_frames(input_tif)

    y = tifs2cv(str(input_tif), m=m, d=d, q=q)
    recon = cv2tifs(y, str(output_tif))

    if len(original) != len(recon):
        raise RuntimeError(
            f"Frame count mismatch: original={len(original)}, reconstructed={len(recon)}"
        )

    print(f"Input           : {input_tif}")
    print(f"Output          : {output_tif}")
    print(f"Frames          : {len(original)}")
    print(f"Block size (m)  : {m}")
    print(f"Search d        : [{d[0]}, {d[1]}]")
    print(f"Quality (q)     : {q}")
    print("-")

    rmses = []
    maxerrs = []
    for i, (fo, fr) in enumerate(zip(original, recon), start=1):
        e = np.abs(fo.astype(float) - fr.astype(float))
        r = _rmse(fo, fr)
        mxe = float(np.max(e))
        rmses.append(r)
        maxerrs.append(mxe)
        print(f"Frame {i:03d}: RMSE={r:.4f}, MaxAbsErr={mxe:.1f}")

    print("-")
    print(f"Mean RMSE       : {np.mean(rmses):.4f}")
    print(f"Max frame RMSE  : {np.max(rmses):.4f}")
    print(f"Global MaxAbsErr: {np.max(maxerrs):.1f}")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "input_sequence.tif"
    default_output = script_dir / "reconstructed_sequence.tif"

    parser = argparse.ArgumentParser(
        description="Round-trip test for tifs2cv/cv2tifs (frame-by-frame RMSE and max error)."
    )
    parser.add_argument(
        "--input",
        default=str(default_input),
        help=f"Input multi-frame TIFF path (default: {default_input})",
    )
    parser.add_argument(
        "--output",
        default=str(default_output),
        help=f"Output reconstructed TIFF path (default: {default_output})",
    )
    parser.add_argument("--m", type=int, default=8, help="Macroblock size (default: 8)")
    parser.add_argument(
        "--dx", type=int, default=16, help="Search displacement x (default: 16)"
    )
    parser.add_argument(
        "--dy", type=int, default=8, help="Search displacement y (default: 8)"
    )
    parser.add_argument(
        "--q", type=float, default=0, help="JPEG quality; 0 means lossless path"
    )
    args = parser.parse_args()

    run_roundtrip(
        input_tif=Path(args.input).expanduser().resolve(),
        output_tif=Path(args.output).expanduser().resolve(),
        m=args.m,
        d=[args.dx, args.dy],
        q=args.q,
    )
