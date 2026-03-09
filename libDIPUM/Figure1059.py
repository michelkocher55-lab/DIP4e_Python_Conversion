import sys
import os
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage import uniform_filter
from libDIPUM.data_path import dip_data

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from nCutSegmentation import nCutSegmentation, mat2gray
except ImportError:
    try:
        from .nCutSegmentation import nCutSegmentation, mat2gray
    except ImportError:
        import nCutSegmentation
        from nCutSegmentation import mat2gray


def Figure1059():
    """Figure1059."""
    # Data
    image_path = dip_data("iceberg.tif")
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    f_raw = imread(image_path)
    # Ensure grayscale
    if f_raw.ndim == 3:
        f = f_raw.mean(axis=2)
    else:
        f = f_raw.astype(float)

    f_norm = mat2gray(f)

    # Smooth the image.
    # w = ones(25)/numel(ones(25));
    # I = imfilter(f, w, 'replicate');
    print("Applying smoothing (25x25)...")
    I_smooth = uniform_filter(f_norm, size=25, mode="nearest")

    # Specify 2 regions
    # Tuning: sample_rate=0.2, edgeVariance=0.9 (strong connectivity), sampleRadius=20 (larger context)
    print("Running nCutSegmentation(2)... with edgeVariance=0.9, r=20")
    S2 = nCutSegmentation(
        I_smooth, 2, edgeVariance=0.9, sample_rate=0.2, sampleRadius=20
    )

    # Note: MATLAB code manually resets 0/1 labels.
    # nCutSegmentation returns labels 1, 2.
    # We will just normalize to 0-1 range.
    S2_norm = mat2gray(S2)

    # Specify 3 regions
    print("Running nCutSegmentation(3)... with edgeVariance=0.9, r=20")
    S3 = nCutSegmentation(
        I_smooth, 3, edgeVariance=0.9, sample_rate=0.2, sampleRadius=20
    )
    S3_norm = mat2gray(S3)

    # Display
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(f_norm, cmap="gray")
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(I_smooth, cmap="gray")
    axes[0, 1].set_title("Smoothed Image")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(S2_norm, cmap="gray")
    axes[1, 0].set_title("N-Cut (2 regions)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(S3_norm, cmap="gray")
    axes[1, 1].set_title("N-Cut (3 regions)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig("Figure1059.png")
    print("Saved Figure1059.png")
    plt.show()


if __name__ == "__main__":
    Figure1059()
