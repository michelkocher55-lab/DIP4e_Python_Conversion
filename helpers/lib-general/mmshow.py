from typing import Any
import numpy as np
import matplotlib.pyplot as plt


def mmshow(img: Any, *args: Any):
    """
    Displays an image with fewer than 7 optional binary overlays in different colors.

    Parameters:
    img: The base image (grayscale or RGB).
    *args: Up to 6 binary images to overlay.

    Colors are applied in order:
    1. Red
    2. Green
    3. Blue
    4. Yellow
    5. Cyan
    6. Magenta
    """

    # Define colors (R, G, B)
    colors = [
        [1.0, 0.0, 0.0],  # Red
        [0.0, 1.0, 0.0],  # Green
        [0.0, 0.0, 1.0],  # Blue
        [1.0, 1.0, 0.0],  # Yellow
        [0.0, 1.0, 1.0],  # Cyan
        [1.0, 0.0, 1.0],  # Magenta
    ]

    if len(args) > 6:
        print("Warning: mmshow supports up to 6 overlays. Ignoring extra inputs.")
        overlays = args[:6]
    else:
        overlays = args

    # Prepare Base Image
    img = np.asarray(img)
    if img.dtype.kind != "f":
        # Simple/naive normalization if not float
        # Assuming uint8 or similar if max > 1
        if img.max() > 1:
            img = img.astype(float) / 255.0
        else:
            img = img.astype(float)

    # Convert to RGB if needed
    if img.ndim == 2:
        img_rgb = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 3:
        img_rgb = img.copy()
    else:
        # Unexpected dimensions
        img_rgb = img

    # Apply Overlays
    for i, mask in enumerate(overlays):
        if mask is None:
            continue

        # Ensure mask is boolean
        mask = np.asarray(mask).astype(bool)

        # Check shape compatibility
        if mask.shape != img_rgb.shape[:2]:
            print(
                f"Warning: Overlay {i + 1} shape {mask.shape} does not match image shape {img_rgb.shape[:2]}. Skipping."
            )
            continue

        # Apply color
        # Strategy: Replace pixel with color? Or Blend?
        # MATLAB mmshow behavior usually replaces (sets color).
        color = colors[i]
        img_rgb[mask] = color

    plt.imshow(img_rgb)
    # plt.axis('off') # Optional, preserve axes for now unless requested
