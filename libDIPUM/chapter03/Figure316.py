import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from libDIPUM.data_path import dip_data

# Filenames key map
# MATLAB:
# f.dark = imread ('Pollen-dark.tif');
# f.light = imread ('Pollen-light.tif');
# f.lowcontrast = imread ('pollen-lowcontrast.tif');
# f.highcontrast = imread ('Pollen-high-contrast.tif');

files_map = [
    ("dark", "Pollen-dark.tif", "Dark"),
    ("light", "Pollen-light.tif", "Light"),
    ("lowcontrast", "pollen-lowcontrast.tif", "Low Contrast"),
    ("highcontrast", "Pollen-high-contrast.tif", "High Contrast"),
]

images = []

for key, fname, title in files_map:
    img = imread(dip_data(fname))
    if img.ndim == 3:
        img = img[:, :, 0]
    images.append((img, title))

# Plotting
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i, (img, title) in enumerate(images):
    # Top row: Images
    ax_img = axes[0, i]
    # Use vmin=0, vmax=255 to show absolute intensity/contrast
    ax_img.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax_img.set_title(title)
    ax_img.axis("off")

    # Bottom row: Histograms
    ax_hist = axes[1, i]
    counts, bins = np.histogram(img.ravel(), bins=256, range=(0, 255))
    # Plot bar. bins has 257 edges.
    # Use bin centers or left edges.
    # MATLAB bar usually centers on value if x is provided.
    # Here we just plot valid distribution.

    ax_hist.bar(bins[:-1], counts, width=1, color="black", align="edge")
    ax_hist.set_xlim([0, 255])
    # MATLAB has axis([0 255 0 max(N)]). matplotlib autolimits y usually fine.
    ax_hist.set_ylim([0, counts.max() * 1.05])

    # Make subplot square-ish aspect ratio like MATLAB 'axis square'?
    # In matplotlib, aspect='equal' forces data units to be equal-> not suitable for hist (x=255, y=3000).
    # We can set box aspect.
    ax_hist.set_box_aspect(1)

plt.tight_layout()
plt.savefig("Figure316.png")
print("Saved Figure316.png")
plt.show()
