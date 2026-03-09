from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import random_noise
from libDIP.intScaling4e import intScaling4e
from libDIPUM.sobel import sobel
from libDIPUM.lap import lap
from libDIPUM.data_path import dip_data

# Figure 10.11


def imcrop_matlab(img: Any, rect: Any):
    """imcrop_matlab."""
    # MATLAB rect = [x, y, w, h], inclusive for integer coordinates.
    x, y, w, h = [int(v) for v in rect]
    r0 = max(y - 1, 0)
    c0 = max(x - 1, 0)
    r1 = min(r0 + h + 1, img.shape[0])
    c1 = min(c0 + w + 1, img.shape[1])
    return img[r0:r1, c0:c1]


def improfile_top_row(img: Any, x_start: Any = 1, x_end: Any = 596):
    """improfile_top_row."""
    # MATLAB improfile(I, [1,596], [1,1]) -> top row profile.
    c0 = max(x_start - 1, 0)
    c1 = min(x_end, img.shape[1])
    return img[0, c0:c1]


# Parameters
sig = np.array([0.1, 1, 10], dtype=float)
var = (sig**2) / (255.0**2)

# Data
a = imread(dip_data("graywedge.png"))
if a.ndim == 3:
    a = a[..., 0]

# MATLAB im2double behavior
if np.issubdtype(a.dtype, np.integer):
    ad = a.astype(np.float64) / float(np.iinfo(a.dtype).max)
else:
    ad = a.astype(np.float64)

rect = [2, 2, 596, 248]

# No noise
ac = imcrop_matlab(ad, rect)
as_ = intScaling4e(ac, "full")
ap = improfile_top_row(ac, 1, 596)

# First derivative
s, _ = sobel(ad)
sc = imcrop_matlab(s, rect)
sp = improfile_top_row(sc, 1, 596)
ss = intScaling4e(sc, "full")

# Second derivative
l = lap(ad)
lc = imcrop_matlab(l, rect)
lp = improfile_top_row(lc, 1, 596)
ls = intScaling4e(lc)

# Noise + derivatives
anc, anp, ans = [], [], []
snc, snp, sns = [], [], []
lnc, lnp, lns = [], [], []

for v in var:
    an = random_noise(ad, mode="gaussian", mean=0.0, var=float(v))

    anc_i = imcrop_matlab(an, rect)
    anp_i = improfile_top_row(anc_i, 1, 596)
    ans_i = intScaling4e(anc_i, "full")

    sn_i, _ = sobel(an)
    snc_i = imcrop_matlab(sn_i, rect)
    snp_i = improfile_top_row(snc_i, 1, 596)
    sns_i = intScaling4e(snc_i, "full")

    ln_i = lap(an)
    lnc_i = imcrop_matlab(ln_i, rect)
    lnp_i = improfile_top_row(lnc_i, 1, 596)
    lns_i = intScaling4e(lnc_i, "full")

    anc.append(anc_i)
    anp.append(anp_i)
    ans.append(ans_i)

    snc.append(snc_i)
    snp.append(snp_i)
    sns.append(sns_i)

    lnc.append(lnc_i)
    lnp.append(lnp_i)
    lns.append(lns_i)

# Display
fig = plt.figure(figsize=(12, 20))

plt.subplot(8, 3, 1)
plt.imshow(as_, cmap="gray")
plt.axis("off")

plt.subplot(8, 3, 2)
plt.imshow(ss, cmap="gray")
plt.axis("off")

plt.subplot(8, 3, 3)
plt.imshow(ls, cmap="gray")
plt.axis("off")

plt.subplot(8, 3, 4)
plt.plot(ap, "k-")

plt.subplot(8, 3, 5)
plt.plot(sp, "k-")

plt.subplot(8, 3, 6)
plt.plot(lp, "k-")

subplot_idx = 7
for i in range(len(var)):
    plt.subplot(8, 3, subplot_idx)
    subplot_idx += 1
    plt.imshow(ans[i], cmap="gray")
    plt.axis("off")

    plt.subplot(8, 3, subplot_idx)
    subplot_idx += 1
    plt.imshow(sns[i], cmap="gray")
    plt.axis("off")

    plt.subplot(8, 3, subplot_idx)
    subplot_idx += 1
    plt.imshow(lns[i], cmap="gray")
    plt.axis("off")

    plt.subplot(8, 3, subplot_idx)
    subplot_idx += 1
    plt.plot(anp[i], "k-")

    plt.subplot(8, 3, subplot_idx)
    subplot_idx += 1
    plt.plot(snp[i], "k-")

    plt.subplot(8, 3, subplot_idx)
    subplot_idx += 1
    plt.plot(lnp[i], "k-")

plt.tight_layout()
plt.savefig("Figure1011.png", dpi=300, bbox_inches="tight")
plt.show()
