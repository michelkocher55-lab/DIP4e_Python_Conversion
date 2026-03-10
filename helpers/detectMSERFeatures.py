"""MATLAB-like detectMSERFeatures using OpenCV MSER.

Example (MATLAB-like positional name/value style):
    R, RCC = detectMSERFeatures(
        Is,
        'ThresholdDelta', 3.92,
        'RegionAreaRange', [10260, 34200],
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover
    raise ImportError("OpenCV (cv2) is required for detectMSERFeatures") from exc


@dataclass
class MSERRegions:
    """Lightweight region container similar to MATLAB MSERRegions."""

    PixelList: List[np.ndarray]  # each region is (Ni,2) with [x,y], 1-based

    @property
    def Count(self) -> int:
        """Count."""
        return len(self.PixelList)

    @property
    def Lengths(self) -> np.ndarray:
        """Lengths."""
        if not self.PixelList:
            return np.zeros((0,), dtype=np.int32)
        return np.asarray([p.shape[0] for p in self.PixelList], dtype=np.int32)


def _im2uint8(I: np.ndarray) -> np.ndarray:
    """_im2uint8."""
    a = np.asarray(I)
    if a.ndim != 2:
        raise ValueError("I must be a 2-D grayscale image")
    if not np.isrealobj(a):
        raise ValueError("I must be real")

    if a.dtype == np.uint8:
        return a.copy()

    if a.dtype == np.bool_:
        return a.astype(np.uint8) * 255

    if np.issubdtype(a.dtype, np.floating):
        # MATLAB im2uint8 semantics: [0,1] expected for float.
        out = np.round(np.clip(a, 0.0, 1.0) * 255.0)
        return out.astype(np.uint8)

    if np.issubdtype(a.dtype, np.integer):
        info = np.iinfo(a.dtype)
        arr = a.astype(np.float64)
        if info.min < 0:
            arr = (arr - info.min) / (info.max - info.min)
        else:
            arr = arr / info.max
        out = np.round(np.clip(arr, 0.0, 1.0) * 255.0)
        return out.astype(np.uint8)

    return np.round(np.clip(a.astype(np.float64), 0.0, 1.0) * 255.0).astype(np.uint8)


def _default_params(image_shape: Tuple[int, int]) -> Dict[str, Any]:
    """_default_params."""
    h, w = image_shape
    return {
        "ThresholdDelta": 5 * 100.0 / 255.0,
        "RegionAreaRange": [30, 14000],
        "MaxAreaVariation": 0.25,
        "ROI": [1, 1, w, h],
        "usingROI": False,
    }


def _to_param_dict(args: Sequence[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """_to_param_dict."""
    if len(args) % 2 != 0:
        raise ValueError("Name/value arguments must come in pairs")

    out: Dict[str, Any] = {}

    # Parse positional name/value pairs like MATLAB call style.
    for i in range(0, len(args), 2):
        k = args[i]
        v = args[i + 1]
        if not isinstance(k, str):
            raise ValueError("Parameter names must be strings")
        out[k.lower()] = v

    # Merge kwargs (also case-insensitive).
    for k, v in kwargs.items():
        out[str(k).lower()] = v

    return out


def _check_roi(roi: Sequence[int], image_shape: Tuple[int, int]) -> None:
    """_check_roi."""
    if len(roi) != 4:
        raise ValueError("ROI must be [x y width height]")

    x, y, w, h = [int(v) for v in roi]
    if w <= 0 or h <= 0:
        raise ValueError("ROI width and height must be positive")

    H, W = image_shape
    if x < 1 or y < 1 or (x + w - 1) > W or (y + h - 1) > H:
        raise ValueError("ROI is outside image bounds")


def _apply_roi(I: np.ndarray, roi: Sequence[int], using_roi: bool) -> np.ndarray:
    """_apply_roi."""
    if not using_roi:
        return I
    x, y, w, h = [int(v) for v in roi]
    x0 = x - 1
    y0 = y - 1
    return I[y0 : y0 + h, x0 : x0 + w]


def _offset_region_points(
    points_xy_1based: np.ndarray, roi: Sequence[int], using_roi: bool
) -> np.ndarray:
    """_offset_region_points."""
    if not using_roi:
        return points_xy_1based
    x, y, _, _ = [int(v) for v in roi]
    out = points_xy_1based.copy()
    out[:, 0] += x - 1
    out[:, 1] += y - 1
    return out


def _region2cc(regions: MSERRegions, image_shape: Tuple[int, int]) -> Dict[str, Any]:
    """_region2cc."""
    # MATLAB-like connected component struct fields.
    H, W = image_shape
    pixel_idx_list: List[np.ndarray] = []

    for loc in regions.PixelList:
        # loc is [x,y] 1-based. Convert to 1-based linear indices with column-major convention.
        # sub2ind([H,W], row, col) => row + (col-1)*H
        row = loc[:, 1].astype(np.int64)
        col = loc[:, 0].astype(np.int64)
        idx = row + (col - 1) * H
        pixel_idx_list.append(idx)

    return {
        "Connectivity": 8,
        "ImageSize": tuple(image_shape),
        "NumObjects": regions.Count,
        "PixelIdxList": pixel_idx_list,
    }


def detectMSERFeatures(I: np.ndarray, *args: Any, **kwargs: Any):
    """Detect MSER regions with MATLAB-like parameters.

    Parameters (case-insensitive, as positional name/value pairs or kwargs):
    - ThresholdDelta: scalar > 0, mapped directly to OpenCV delta (uint8 gray-level step).
    - RegionAreaRange: [minArea, maxArea]
    - MaxAreaVariation: scalar >= 0
    - ROI: [x y width height] (1-based image coordinates)

    Returns
    -------
    regions : MSERRegions
    cc : dict (only if requested by caller in unpacking style)
        Contains Connectivity, ImageSize, NumObjects, PixelIdxList.
    """
    Iu8 = _im2uint8(I)
    image_shape = tuple(Iu8.shape)

    if min(image_shape) < 3:
        raise ValueError("Image dimensions must be at least 3x3")

    params = _default_params(image_shape)
    user = _to_param_dict(args, kwargs)

    # Accepted aliases/case-insensitive names.
    if "thresholddelta" in user:
        params["ThresholdDelta"] = float(user["thresholddelta"])
    if "regionarearange" in user:
        rar = user["regionarearange"]
        if len(rar) != 2:
            raise ValueError("RegionAreaRange must be [min max]")
        params["RegionAreaRange"] = [int(rar[0]), int(rar[1])]
    if "maxareavariation" in user:
        params["MaxAreaVariation"] = float(user["maxareavariation"])
    if "roi" in user:
        params["ROI"] = [int(v) for v in user["roi"]]
        params["usingROI"] = True

    # Validation
    td = params["ThresholdDelta"]
    if not np.isfinite(td) or td <= 0:
        raise ValueError("ThresholdDelta must be > 0")

    min_area, max_area = params["RegionAreaRange"]
    if min_area <= 0 or max_area <= 0:
        raise ValueError("RegionAreaRange values must be positive integers")
    if max_area < min_area:
        raise ValueError("RegionAreaRange max must be >= min")

    mav = params["MaxAreaVariation"]
    if not np.isfinite(mav) or mav < 0:
        raise ValueError("MaxAreaVariation must be >= 0")

    if params["usingROI"]:
        _check_roi(params["ROI"], image_shape)

    img = _apply_roi(Iu8, params["ROI"], params["usingROI"])

    # MATLAB/OpenCV parameter mapping:
    # use ThresholdDelta directly as gray-level step for uint8 data.
    delta = int(max(1, round(td)))

    # OpenCV MSER with defaults aligned to MATLAB wrapper.
    # Different OpenCV builds expose different Python signatures:
    # - some accept underscore kwargs, some positional-only, some no args.
    try:
        mser = cv2.MSER_create(
            _delta=delta,
            _min_area=int(min_area),
            _max_area=int(max_area),
            _max_variation=float(mav),
            _min_diversity=float(0.2),
            _max_evolution=int(200),
            _area_threshold=float(1.0),
            _min_margin=float(0.003),
            _edge_blur_size=int(5),
        )
    except TypeError:
        try:
            # Positional signature fallback used by many OpenCV wheels.
            mser = cv2.MSER_create(
                int(delta),
                int(min_area),
                int(max_area),
                float(mav),
                float(0.2),
                int(200),
                float(1.0),
                float(0.003),
                int(5),
            )
        except TypeError:
            # Last resort: create with defaults and set what is available.
            mser = cv2.MSER_create()
            if hasattr(mser, "setDelta"):
                mser.setDelta(int(delta))
            if hasattr(mser, "setMinArea"):
                mser.setMinArea(int(min_area))
            if hasattr(mser, "setMaxArea"):
                mser.setMaxArea(int(max_area))
            if hasattr(mser, "setMaxVariation"):
                mser.setMaxVariation(float(mav))
            if hasattr(mser, "setMinDiversity"):
                mser.setMinDiversity(float(0.2))
            if hasattr(mser, "setMaxEvolution"):
                mser.setMaxEvolution(int(200))
            if hasattr(mser, "setAreaThreshold"):
                mser.setAreaThreshold(float(1.0))
            if hasattr(mser, "setMinMargin"):
                mser.setMinMargin(float(0.003))
            if hasattr(mser, "setEdgeBlurSize"):
                mser.setEdgeBlurSize(int(5))

    # OpenCV returns list of regions with points [x,y], 0-based.
    reg_pts, _ = mser.detectRegions(img)

    pixel_list: List[np.ndarray] = []
    seen_signatures = set()
    h_img, w_img = img.shape[:2]
    for pts in reg_pts:
        if pts.size == 0:
            continue

        # Ensure shape (N,2), [x,y], 0-based integer.
        pts = np.asarray(pts, dtype=np.int32).reshape(-1, 2)
        if pts.shape[0] == 0:
            continue

        # Deduplicate points and compute pixel-area explicitly.
        # OpenCV may return repeated points/regions depending on build/version.
        lin = pts[:, 1].astype(np.int64) * int(w_img) + pts[:, 0].astype(np.int64)
        lin = np.unique(lin)
        area = int(lin.size)
        if area < int(min_area) or area > int(max_area):
            continue

        # Remove exact duplicate regions by their sorted linear indices.
        sig = lin.tobytes()
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)

        y = lin // int(w_img)
        x = lin % int(w_img)
        loc = np.column_stack((x, y)).astype(np.int64) + 1

        # Offset back if ROI was used.
        loc = _offset_region_points(loc, params["ROI"], params["usingROI"])

        pixel_list.append(loc)

    regions = MSERRegions(pixel_list)
    cc = _region2cc(regions, image_shape)
    return regions, cc


__all__ = ["detectMSERFeatures", "MSERRegions"]
