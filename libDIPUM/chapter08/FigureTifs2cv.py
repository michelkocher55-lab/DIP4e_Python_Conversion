from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

from libDIPUM.compare import compare
from libDIPUM.cv2tifs import cv2tifs
from libDIPUM.imratio import imratio
from libDIPUM.tifs2cv import tifs2cv
from libDIPUM.data_path import dip_data


# %% FigureTifs2cv

# %% Parameters (edit these)
input_tif = dip_data("shuttle.tif")
output_tif = dip_data("shuttlereconstructed_sequence.tif")
m = 8
d = [16, 8]
q = 1  # q=0 lossless residual coding, q>0 lossy JPEG residual coding


# %% Helpers


def load_frames(path: Any):
    """load_frames."""
    arr = np.asarray(imread(path))
    if arr.ndim == 2:
        return [arr]
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4) and arr.shape[0] != arr.shape[1]:
            raise ValueError("Expected a grayscale TIFF sequence, got color data.")
        return [arr[i, :, :] for i in range(arr.shape[0])]
    raise ValueError("Unsupported TIFF dimensions.")


# %% Data
original = load_frames(input_tif)


# %% Process
y = tifs2cv(input_tif, m, d, q)
compression_ratio = imratio(input_tif, y)
reconstructed = cv2tifs(y, output_tif)

if len(original) != len(reconstructed):
    raise RuntimeError("Frame count mismatch between original and reconstructed.")

rmse = []
for i in range(len(original)):
    rmse.append(compare(original[i].astype(float), reconstructed[i].astype(float), 0))

print(f"Input file: {input_tif}")
print(f"Output file: {output_tif}")
print(f"Frames: {len(original)}")
print(f"m={m}, d={d}, q={q}")
print(f"Compression ratio: {compression_ratio:.4f}")
print(f"Mean RMSE: {np.mean(rmse):.4f}")
print(f"Max RMSE: {np.max(rmse):.4f}")


# %% Display
show_n = min(3, len(original))
fig = plt.figure(1, figsize=(12, 4 * show_n))
for i in range(show_n):
    plt.subplot(show_n, 3, 1 + 3 * i)
    plt.imshow(original[i], cmap="gray")
    plt.title(f"Original frame {i + 1}")
    plt.axis("off")

    plt.subplot(show_n, 3, 2 + 3 * i)
    plt.imshow(reconstructed[i], cmap="gray")
    plt.title(f"Reconstructed frame {i + 1} (RMSE={rmse[i]:.3f})")
    plt.axis("off")

    plt.subplot(show_n, 3, 3 + 3 * i)
    err = original[i].astype(float) - reconstructed[i].astype(float)
    plt.imshow(err, cmap="gray")
    plt.title(f"Error frame {i + 1}")
    plt.axis("off")

plt.tight_layout()
fig.savefig("FigureTifs2cv.png", dpi=150, bbox_inches="tight")
plt.show()
