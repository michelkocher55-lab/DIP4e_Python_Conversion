import sys
import os
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage import uniform_filter

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from nCutSegmentation import mat2gray
    from ncut_impl import computeEdges, NcutImage
except ImportError:
    try:
        from .nCutSegmentation import mat2gray
        from .ncut_impl import computeEdges, NcutImage
    except ImportError:
        from nCutSegmentation import mat2gray
        from ncut_impl import computeEdges, NcutImage


def debug_figure1059():
    """debug_figure1059."""
    # Data
    image_path = "/Users/michelkocher/michel/Data/DIPUM3e/DIPUM3E_Images/iceberg.tif"
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    f_raw = imread(image_path)
    if f_raw.ndim == 3:
        f = f_raw.mean(axis=2)
    else:
        f = f_raw.astype(float)

    f_norm = mat2gray(f)

    # Smoothing
    I_smooth = uniform_filter(f_norm, size=25, mode="nearest")

    # Analyze Edges
    print("Computing edges...")
    # Default parameters from ncut_impl/quadedgep: [4, 3, 21, 3] or similar
    edge_res = computeEdges(I_smooth)
    emag = edge_res["emag"]
    imageEdges = edge_res["imageEdges"]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(emag, cmap="jet")
    plt.title("Edge Magnitude (emag)")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(imageEdges, cmap="gray")
    plt.title("Thresholded Edges")
    plt.savefig("debug_edges_1059.png")
    print("Saved debug_edges_1059.png")

    # Try Segmentation with higher variance
    print("Running NcutImage(3) with edgeVariance=0.5...")
    SegLabel, _, _, _, _, _ = NcutImage(I_smooth, 3, edgeVariance=0.5, sample_rate=0.2)

    plt.figure()
    plt.imshow(mat2gray(SegLabel), cmap="gray")
    plt.title("Seg (3) Var=0.5")
    plt.savefig("debug_seg_0.5.png")
    print("Saved debug_seg_0.5.png")


if __name__ == "__main__":
    debug_figure1059()
