from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from skimage.segmentation import active_contour
import os
import sys
from pathlib import Path

edge_lib_path = str(Path(__file__).resolve().parents[1])
if edge_lib_path not in sys.path:
    sys.path.append(edge_lib_path)

from snakeMap4e import snakeMap4e
from libDIPUM.data_path import dip_data


def Figure113_skimage():
    """
    Re-implementation of Figure 11.3 using skimage.segmentation.active_contour.
    Uses snakeMap4e for standard DIP4e edge map generation logic, but replaces the
    iterative solver with skimage's robust implementation.
    """

    plt.close("all")

    # 1. Load Data
    mat_path = dip_data("Figure112.mat")
    if not os.path.exists(mat_path):
        print(f"Error: {mat_path} not found.")
        return

    print(f"Loading {mat_path}...")
    mat_data = scipy.io.loadmat(mat_path)

    def get_var(name: Any):
        """get_var."""
        if name not in mat_data:
            return None
        val = mat_data[name]
        return val.item() if val.size == 1 else np.squeeze(val)

    g = get_var("g")  # Input image
    T = get_var("T")  # Threshold
    Sig = get_var("Sig")  # Sigma for Gaussian
    NSig = get_var("NSig")  # Size of kernel
    # NIter = get_var('NIter') # Number of iterations (approx 230 in mat file, but we might want more for convergence)
    Alpha = get_var("Alpha")  # Continuity (membrane)
    Beta = get_var("Beta")  # Curvature (thin plate)
    Gamma = get_var("Gamma")  # Step size (Viscosity)

    xi = get_var("xi")  # Initial X (columns)
    yi = get_var("yi")  # Initial Y (rows)
    # Ensure xi, yi are correct shape (N,)
    if xi.ndim > 1:
        xi = xi.flatten()
    if yi.ndim > 1:
        yi = yi.flatten()

    # Construct initial snake (N, 2) -> (row, col) = (y, x)
    init_snake = np.stack([yi, xi], axis=1)

    # Helper to run snake on a pre-computed feature map (energy map)
    def run_snake_on_map(emap: Any, label: Any = ""):
        """run_snake_on_map."""
        print(f"Running snake for {label}...")
        try:
            # We treat the emap as the energy surface.
            # w_line=1: Attract to bright regions (high values in emap).
            # w_edge=0: Do NOT compute gradients of emap, use emap directly.
            # This allows us to feed ANY processed map (e.g. smoothed binary edges) to the snake.
            snake = active_contour(
                emap,
                init_snake,
                alpha=Alpha,
                beta=Beta,
                w_line=1,
                w_edge=0,
                gamma=0.01,
                max_iterations=2500,
                boundary_condition="periodic",
            )
        except Exception as e:
            print(f"Snake failed for {label}: {e}")
            snake = init_snake
        return snake, emap

    # The Logic matches Figure113.m / snakeMap4e options:

    # Case 1: "Both"
    # snakeMap4e(..., 'both') -> Filter Image -> Edge/Thresh -> Filter EdgeMap
    # This produces a smoothed version of the binary edges.
    print("Generating Map: Both...")
    emap1 = snakeMap4e(g, T, Sig, NSig, "both")
    snake1, _ = run_snake_on_map(emap1, label="Both")

    # Case 2: "After"
    # snakeMap4e(..., 'after') -> Edge/Thresh -> Filter EdgeMap
    print("Generating Map: After...")
    emap2 = snakeMap4e(g, T, Sig, NSig, "after")
    snake2, _ = run_snake_on_map(emap2, label="After")

    # Case 3: "Before"
    # snakeMap4e(..., 'before') -> Filter Image -> Edge/Thresh (No Post-Filter)
    # Result is Binary Edges of Smoothed Image.
    # Note: If T='auto', it's binary. Active contour on binary image might be jagged.
    # But this is what the book/MATLAB does.
    print("Generating Map: Before...")
    emap3 = snakeMap4e(g, T, Sig, NSig, "before")
    snake3, _ = run_snake_on_map(emap3, label="Before")

    # Case 4: "None"
    # snakeMap4e(..., 'none') -> Edge/Thresh (No Filter)
    # Binary Edges of Raw Image. (Very Noisy)
    print("Generating Map: None...")
    emap4 = snakeMap4e(g, T, Sig, NSig, "none")
    snake4, _ = run_snake_on_map(emap4, label="None")

    # 3. Visualization
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    def plot_snake(ax: Any, img_bg: Any, snake: Any, emap_disp: Any, title: Any):
        """plot_snake."""
        # We display the EDGE MAP in the background to show what the snake sees/follows,
        # or the original image? Figure 11.3 shows EDGE MAPS in top row, Snake on Original in bottom.
        # But here we have 4 subplots. Let's show Snake on Original, but maybe an overlay?
        # Let's stick to Snake on Original Data (g), but title explains the process.

        ax.imshow(g, cmap="gray")
        ax.plot(init_snake[:, 1], init_snake[:, 0], "--r", lw=1, label="Init")
        ax.plot(snake[:, 1], snake[:, 0], "-g", lw=2, label="Final")
        ax.set_title(title)
        ax.axis("off")

    # To strictly match Figure 11.3 layout (4 quadrants, showing results of 4 configs)
    plot_snake(axs[0, 0], g, snake1, emap1, "Both (Filter Img & Map)")
    plot_snake(axs[0, 1], g, snake2, emap2, "After (Filter Map Only)")
    plot_snake(axs[1, 0], g, snake3, emap3, "Before (Filter Img Only)")
    plot_snake(axs[1, 1], g, snake4, emap4, "None (No Filtering)")

    # Optional: Save edge maps for debugging?
    # fig_maps, axs_maps = plt.subplots(2, 2)
    # axs_maps[0,0].imshow(emap1, cmap='gray'); axs_maps[0,0].set_title("Map Both")
    # axs_maps[0,1].imshow(emap2, cmap='gray'); axs_maps[0,1].set_title("Map After")
    # ...

    plt.tight_layout()
    plt.savefig("Figure113_skimage.png")
    print("Saved Figure113_skimage.png")

    # Also save Figure113_maps.png to allow user to see the intermediate potentials
    fig_m, ax_m = plt.subplots(2, 2, figsize=(10, 10))
    ax_m[0, 0].imshow(emap1, cmap="gray")
    ax_m[0, 0].set_title("Energy Map: Both")
    ax_m[0, 1].imshow(emap2, cmap="gray")
    ax_m[0, 1].set_title("Energy Map: After")
    ax_m[1, 0].imshow(emap3, cmap="gray")
    ax_m[1, 0].set_title("Energy Map: Before")
    ax_m[1, 1].imshow(emap4, cmap="gray")
    ax_m[1, 1].set_title("Energy Map: None")
    plt.tight_layout()
    plt.savefig("Figure113_skimage_maps.png")
    print("Saved Figure113_skimage_maps.png")

    plt.show()


if __name__ == "__main__":
    Figure113_skimage()
