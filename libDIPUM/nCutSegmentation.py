from typing import Any
import numpy as np
from skimage.transform import resize
from skimage.segmentation import slic

try:
    from skimage.future import graph
except Exception as e:
    raise ImportError("skimage.future.graph is required for nCutSegmentation") from e


def mat2gray(img: Any):
    """Scale array to [0, 1] like MATLAB mat2gray."""
    arr = np.asarray(img, dtype=np.float64)
    min_v = arr.min()
    max_v = arr.max()
    if max_v - min_v < 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - min_v) / (max_v - min_v)


def _slic_compat(image: Any, n_segments: Any, compactness: Any):
    """Call slic with compatibility across scikit-image versions."""
    try:
        return slic(
            image,
            n_segments=int(n_segments),
            compactness=float(compactness),
            start_label=1,
            channel_axis=None,
        )
    except TypeError:
        return slic(
            image,
            n_segments=int(n_segments),
            compactness=float(compactness),
            start_label=1,
            multichannel=False,
        )


def _force_k_labels_from_region_means(labels: Any, image: Any, k: Any):
    """Reduce any label map to exactly k labels using weighted 1D k-means on region means."""
    if k <= 1:
        return np.ones_like(labels, dtype=np.int32)

    labels = labels.astype(np.int64)
    uniq = np.unique(labels)
    n_regions = len(uniq)
    if n_regions <= k:
        out = np.zeros_like(labels, dtype=np.int32)
        for i, lab in enumerate(uniq, start=1):
            out[labels == lab] = i
        return out

    means = np.zeros(n_regions, dtype=np.float64)
    counts = np.zeros(n_regions, dtype=np.float64)

    for i, lab in enumerate(uniq):
        m = labels == lab
        counts[i] = float(m.sum())
        means[i] = float(image[m].mean())

    # Weighted 1D k-means (stable and dependency-light)
    centers = np.linspace(means.min(), means.max(), k)
    assign = np.zeros(n_regions, dtype=np.int32)

    for _ in range(30):
        d = np.abs(means[:, None] - centers[None, :])
        new_assign = np.argmin(d, axis=1)
        if np.array_equal(new_assign, assign):
            break
        assign = new_assign
        for j in range(k):
            idx = np.where(assign == j)[0]
            if idx.size > 0:
                w = counts[idx]
                centers[j] = np.sum(means[idx] * w) / np.sum(w)

    out = np.zeros_like(labels, dtype=np.int32)
    for i, lab in enumerate(uniq):
        out[labels == lab] = int(assign[i]) + 1

    return out


def _kmeans1d_pixels(image: Any, k: Any, max_iter: Any = 40):
    """Cluster pixel intensities into exactly k classes, labels in {1..k}."""
    x = np.asarray(image, dtype=np.float64).ravel()
    if k <= 1:
        return np.ones_like(image, dtype=np.int32)

    c = np.linspace(x.min(), x.max(), k)
    a = np.zeros_like(x, dtype=np.int32)
    for _ in range(max_iter):
        d = np.abs(x[:, None] - c[None, :])
        na = np.argmin(d, axis=1)
        if np.array_equal(na, a):
            break
        a = na
        for j in range(k):
            idx = a == j
            if np.any(idx):
                c[j] = x[idx].mean()

    out = (a.reshape(image.shape) + 1).astype(np.int32)
    return out


def nCutSegmentation(I: Any, NR: Any, sf: Any = 1.0, **kwargs: Any):
    """Segment grayscale image using a normalized-cut equivalent in Python.

    Parameters
    ----------
    I : ndarray
        Grayscale input image.
    NR : int
        Number of output regions.
    sf : float, optional
        Scale factor for processing (0 < sf <= 1 typically).

    Optional kwargs
    ---------------
    n_segments : int
        Number of SLIC superpixels used to build the graph.
    compactness : float
        SLIC compactness.
    ncut_thresh : float
        Threshold for normalized-cut recursion.
    ncut_num_cuts : int
        Maximum recursive cuts attempted by normalized cuts.

    Returns
    -------
    S : ndarray
        Label image with values in {1, ..., NR}, resized back to input size.
    """
    I = np.asarray(I)
    if I.ndim != 2:
        raise ValueError("Input image must be grayscale (2D).")

    NR = int(NR)
    if NR < 1:
        raise ValueError("NR must be >= 1.")

    sf = float(sf)
    if sf <= 0:
        raise ValueError("sf must be > 0.")

    # Process scale
    if sf != 1.0:
        new_shape = (
            max(2, int(round(I.shape[0] * sf))),
            max(2, int(round(I.shape[1] * sf))),
        )
        I_proc = resize(
            I.astype(np.float64), new_shape, anti_aliasing=True, preserve_range=True
        )
    else:
        I_proc = I.astype(np.float64)

    I_proc = mat2gray(I_proc)

    # Superpixel graph size heuristic
    n_segments = int(kwargs.get("n_segments", max(800, min(6000, I_proc.size // 60))))
    compactness = float(kwargs.get("compactness", 10.0))

    # 1) Build superpixels
    sp_labels = _slic_compat(I_proc, n_segments=n_segments, compactness=compactness)

    # 2) Build RAG and run normalized cuts
    # rag_mean_color expects multichannel image; use gray replicated to 3 channels.
    I_rgb = np.dstack([I_proc, I_proc, I_proc])
    rag = graph.rag_mean_color(I_rgb, sp_labels, mode="distance")

    ncut_thresh = float(kwargs.get("ncut_thresh", 1e-3))
    ncut_num_cuts = int(kwargs.get("ncut_num_cuts", max(10, NR * 6)))

    sp_ncut = graph.cut_normalized(
        sp_labels,
        rag,
        thresh=ncut_thresh,
        num_cuts=ncut_num_cuts,
        in_place=False,
    )

    # 3) Force exactly NR labels (MATLAB wrapper behavior expected by figures).
    # If normalized cuts collapses to one region, fallback to clustering the
    # original superpixels by mean intensity to still get NR regions.
    if np.unique(sp_ncut).size < NR:
        S_proc = _force_k_labels_from_region_means(sp_labels, I_proc, NR)
    else:
        S_proc = _force_k_labels_from_region_means(sp_ncut, I_proc, NR)

    # 4) Resize labels back to original size if needed
    if sf != 1.0:
        S = resize(
            S_proc,
            I.shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.int32)
    else:
        S = S_proc.astype(np.int32)

    # Guarantee exactly NR labels in the final output.
    if np.unique(S).size < NR:
        S = _kmeans1d_pixels(
            I_proc if sf == 1.0 else mat2gray(I.astype(np.float64)), NR
        )

    return S
