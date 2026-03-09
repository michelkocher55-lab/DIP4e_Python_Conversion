import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from libDIPUM.compare import compare
from libDIPUM.waveback import waveback
from libDIPUM.wavecopy import wavecopy
from libDIPUM.wavefast import wavefast
from libDIPUM.wavepaste import wavepaste
from libDIPUM.data_path import dip_data

# Figure 8.44 (dead-zone threshold sweep)

# Parameters
n_level = 3  # 3-scale biorthogonal wavelet
thresholds = np.arange(0, 19)  # dead-zone threshold (width) from 0 to 18

# Data
f = imread(dip_data("lena.tif"))
if f.ndim == 3:
    f = f[..., 0]
if f.dtype != np.uint8:
    f = np.clip(np.round(f), 0, 255).astype(np.uint8)

# Wavelet decomposition once (biorthogonal JPEG 9/7).
c, s = wavefast(f.astype(float) - 128.0, n_level, "jpeg9.7")

# Count total detail coefficients (h, v, d at all levels) for percentage.
detail_total = 0
for k in range(1, n_level + 1):
    detail_total += wavecopy("h", c, s, k).size
    detail_total += wavecopy("v", c, s, k).size
    detail_total += wavecopy("d", c, s, k).size

# Process
rms = np.zeros_like(thresholds, dtype=float)
truncated_pct = np.zeros_like(thresholds, dtype=float)

for i, T in enumerate(thresholds):
    # Interpret threshold as dead-zone width -> half-width criterion.
    half_width = float(T) / 2.0

    cq = c.copy()
    truncated_count = 0

    # Apply dead-zone truncation to detail subbands only.
    for k in range(1, n_level + 1):
        for band in ("h", "v", "d"):
            w = wavecopy(band, cq, s, k)
            mask = (np.abs(w) <= half_width) & (w != 0)
            truncated_count += np.count_nonzero(mask)
            w[mask] = 0.0
            cq = wavepaste(band, cq, s, k, w)

    truncated_pct[i] = 100.0 * truncated_count / detail_total

    rec = waveback(cq, s, "jpeg9.7")
    rec = np.clip(np.round(rec + 128.0), 0, 255)
    rms[i] = compare(f.astype(float), rec.astype(float), 0)

# Display
fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(thresholds, rms, "k-s", linewidth=1.2, markersize=4)
ax1.set_xlabel("Dead-zone threshold")
ax1.set_ylabel("RMS error")
ax1.set_xticks(np.arange(0, 19, 1))
ax1.tick_params(axis="y")

ax2 = ax1.twinx()
ax2.plot(thresholds, truncated_pct, "k--o", linewidth=1.2, markersize=3)
ax2.set_ylabel("Truncated detail coefficients (%)")
ax2.tick_params(axis="y")
ax2.set_ylim(0, 100)

ax1.set_title(
    "3-Scale Biorthogonal Wavelet: RMS and Truncated Coefficients vs Dead-zone Threshold"
)
fig.tight_layout()
fig.savefig("Figure844.png", dpi=300, bbox_inches="tight")
plt.show()
