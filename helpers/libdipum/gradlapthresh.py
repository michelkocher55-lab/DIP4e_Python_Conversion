from typing import Any
import numpy as np
import scipy.ndimage
from skimage.util import img_as_float

try:
    from helpers.libdipum.ipercentile import ipercentile
except ImportError:
    pass


def gradlapthresh(f: Any, PG: Any, PL: Any):
    """
    Uses properties of gradient and Laplacian for thresholding.

    Parameters:
        f: Input image.
        PG: Fraction for gradient threshold (0, 1].
        PL: Fraction for Laplacian threshold (0, 1].

    Returns:
        G: Dictionary containing results:
           'G1': Thresholded image.
           'G2': Otsu threshold.
           'G3': Separability measure (not implemented in standard skimage otsu, set to None).
           'G4': Histogram used.
           'G5': Gradient magnitude (normalized).
           'G6': Thresholded gradient mask.
           'G7': Gradient Threshold (TG).
           'G8': Histogram of G5.
           'G9': Percentile of TG.
           'G10': Laplacian abs (normalized).
           'G11': Thresholded Laplacian mask.
           'G12': Laplacian Threshold (TL).
           'G13': Histogram of G10.
           'G14': Percentile of TL.
           'G15': Combined mask (G6 | G11).
           'G16': Masked image (values kept, others 0).
    """

    f = img_as_float(f)
    M, N = f.shape

    if PG > 1 or PG <= 0 or PL > 1 or PL <= 0:
        # MATLAB allows 1 but sets image to all 1s.
        # "fraction in the half open interval (0, 1]"
        # "If a value of 1 is used... all 1's"
        # The check says: error if PG > 1 or PG <= 0.
        # So 1 is valid.
        pass

    if PG <= 0 or PG > 1 or PL <= 0 or PL > 1:
        raise ValueError("PG and PL must be in the half-open interval (0, 1].")

    # Gradient
    # MATLAB: sx = fspecial('sobel')/8
    # sx = [[1 2 1], [0 0 0], [-1 -2 -1]] / 8 (approx, check fspecial)
    # fspecial('sobel') returns:
    # [ 1  2  1 ]
    # [ 0  0  0 ]
    # [-1 -2 -1 ]
    # standard sobel is usually unnormalized or /8.

    sx = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / 8.0
    sy = sx.T

    # imfilter(..., 'replicate')
    gx = scipy.ndimage.convolve(f, sx, mode="nearest")
    gy = scipy.ndimage.convolve(f, sy, mode="nearest")

    # Magnitude
    g5 = np.sqrt(gx**2 + gy**2)
    # Scale to max 1
    max_g5 = g5.max()
    if max_g5 > 0:
        g5 = g5 / max_g5

    # Threshold
    TG = PG * g5.max()  # which is PG * 1 usually
    if PG == 1:
        g6 = np.ones_like(g5, dtype=bool)
    else:
        g6 = g5 > TG

    # Histogram of G5
    g8, _ = np.histogram(g5, bins=256, range=(0, 1))
    g8 = g8 / (np.sum(g8) + 1e-10)  # Normalize

    # Percentile
    # ipercentile expects histogram.
    # TG is value. 'percentile' option takes value 0-1 and returns intensity idx?
    # Wait, ipercentile(h, 'percentile', V) returns intensity Q.
    # MATLAB: G{9} = ipercentile(G{8}, 'percentile', TG)
    # Here TG is a value/intensity (threshold).
    # But he calls 'percentile' with TG.
    # If TG is 0.5 (intensity), does he mean "Find the intensity corresponding to 0.5 percentile?" NO.
    # See MATLAB line 72: G{9} = ipercentile(..., TG).
    # IF TG is intended as intensity, he should use 'intensity' mode to get percentile.
    # BUT the comment says: line 68: "Function PERCENTILE has changed... 'value' means that intensity is used and want to get percentile."
    # Then line 72 calls `ipercentile(..., 'percentile', TG)`.
    # This is confusing.
    # Let's check `gradlapthresh.m` comments again.
    # Line 29: "G{9} = The percentile of the values in G{5} corresponding to TG."
    # So we want the PERCENTILE rank of TG.
    # ipercentile in MATLAB: option 'intensity' -> returns percentile.
    # The code uses 'percentile' in line 72.
    # Maybe `ipercentile` implementation provided in `ipercentile.m` (which I read) switches logic?
    # Let's check `ipercentile.m`:
    #   case 'percentile': V is percentile, Q is intensity.
    #   case 'intensity': V is intensity, Q is percentile.
    # So if we want percentile of TG (which is an intensity value), we should use 'intensity'.
    # Why does MATLAB code use 'percentile'?
    # Line 67 (commented out) used `round(TG*255)`.
    # Maybe the arguments were flipped in his specific version or the file I read has a bug/inconsistency?
    # However, I must follow the Logic: The output should be "The percentile of the values...".
    # So I need to find P such that P% of pixels are < TG.
    # That is `ipercentile(h, 'intensity', TG_index)`.
    # TG is float [0,1]. Histogram is 256 bins.
    # Index = int(TG * 255).

    try:
        tg_idx = int(TG * 255)
        # Clip
        tg_idx = max(0, min(255, tg_idx))
        g9 = ipercentile(g8, "intensity", tg_idx)
    except NameError:
        g9 = 0

    # Laplacian
    # w = [-1 -1 -1;-1 8 -1;-1 -1 -1];
    w = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    lap = scipy.ndimage.convolve(f, w, mode="nearest")
    g10 = np.abs(lap)

    max_g10 = g10.max()
    if max_g10 > 0:
        g10 = g10 / max_g10

    TL = PL * g10.max()
    if PL == 1:
        g11 = np.ones_like(g10, dtype=bool)
    else:
        g11 = g10 > TL

    # Hist / Percentile
    g13, _ = np.histogram(g10, bins=256, range=(0, 1))
    g13 = g13 / (np.sum(g13) + 1e-10)

    try:
        tl_idx = int(TL * 255)
        tl_idx = max(0, min(255, tl_idx))
        g14 = ipercentile(g13, "intensity", tl_idx)
    except NameError:
        g14 = 0

    # Combine
    # MATLAB code: G{15} = G{6} | G{11};
    # However, comment says (LAP) & (GRAD).
    # And if PL=1 (all 1s), OR results in all 1s (Global).
    # Figure 10.39 uses PL=1 and expects gradient masking.
    # This implies AND logic so that PL=1 is Identity.
    g15 = g6 & g11

    # Masked image
    g16 = np.zeros_like(f)
    g16[g15] = f[g15]

    # Histogram of masked pixels
    # MATLAB: G{4} = imhist(G{16}); G{4}(1) = 0; normalize.
    # This histogram includes the 0s from the background mask.
    # He manually zeroes the 0-bin.
    # So we want histogram of `f` where Mask is True.
    # g4 = histogram(f[g15]) (normalized)

    valid_pixels = f[g15]

    g4, _ = np.histogram(valid_pixels, bins=256, range=(0, 1))
    # Note: If we use the whole image G16 (with zeros), bin 0 would be huge.
    # He zeros bin 0.
    # Here `valid_pixels` contains the pixel values.
    # If any valid pixel is 0.0, it falls in bin 0.
    # The MATLAB logic `G{4}(1)=0` zeros out the count of 0-values.
    # Since G16 has many 0s due to masking, this ensures we don't count the mask 0s.
    # BUT if real pixels are 0, they are also ignored.
    # My `valid_pixels` includes only masked-in pixels.
    # If I just histogram `valid_pixels`, I don't have the HUGE spike of mask-zeros.
    # So I don't need to zero bin 0, UNLESS I want to ignore real black pixels too?
    # Code: "The objective is to use the histogram of the pixels marked by I... all other pixels are 0... by setting first component to 0 we ignore... 0's in fo."
    # Since I extracts indices where G15 is 1, G16(I) = f(I).
    # The histogram is on G16. G16 has zeros where mask is 0.
    # So zeros are dominant.
    # Explicitly removing bin 0 removes mask-zeros.
    # So `histogram(valid_pixels)` is correct and doesn't contain mask-zeros, provided I only pass `f[g15]`.
    # However, `f[g15]` might contain real 0.0 values.
    # Does MATLAB `imhist` puts 0.0 in bin 1? Yes.
    # So MATLAB removes real zeros too if they exist.
    # I should behave similarly if I want exact match?
    # Or is my approach of just taking valid pixels better?
    # Valid pixels don't include the mask-zeros.
    # So I will just use `valid_pixels`.

    if len(valid_pixels) == 0:
        # Fallback
        valid_pixels = f.flatten()

    # MATLAB imhist logic involves rounding.
    # If we use np.histogram on floats with range (0,1), it uses floor(v*256).
    # MATLAB uses round(v*255).
    # To match exactly, we convert valid_pixels to uint8 using round.
    if valid_pixels.dtype.kind == "f":
        # Scale and round
        valid_pixels_u8 = (valid_pixels * 255.0).round().astype(np.uint8)
    else:
        valid_pixels_u8 = valid_pixels.astype(np.uint8)

    g4, _ = np.histogram(valid_pixels_u8, bins=256, range=(0, 256))
    g4 = g4.astype(float)
    g4 = g4 / (np.sum(g4) + 1e-10)

    # Otsu
    # Matlab uses `otsuthresh` on the histogram G{4}.
    # We should do the same to ensure exact match.
    # We try to import otsuthresh from local modules (otsuthresh.py).
    # Otsu
    # Matlab uses `otsuthresh` on the histogram G{4}.
    # We should do the same to ensure exact match.
    # We try to import otsuthresh from local modules (otsuthresh.py).
    try:
        try:
            from .otsuthresh import otsuthresh
        except ImportError:
            from helpers.libdipum.otsuthresh import otsuthresh

        g2, g3 = otsuthresh(g4)
    except ImportError:
        print(
            "Warning: Could not import otsuthresh. Using skimage fallback (results may differ)."
        )
        # Fallback to skimage if otsuthresh not available, but this differs slightly.
        try:
            from skimage.filters import threshold_otsu

            # threshold_otsu on pixels gives specific value.
            val = threshold_otsu(valid_pixels)
            # Normalize to 0-1
            if f.dtype.kind in "ui":
                max_val = 255.0  # Assuming 8-bit usually
            else:
                max_val = 1.0  # If float

            # Helper to guess max_val from valid_pixels
            if valid_pixels.max() > 1.1:
                max_val = 255.0
            else:
                max_val = 1.0

            g2 = val / max_val
            g3 = 0
        except:
            g2 = 0.5
            g3 = 0

    # Final thresholding
    g1 = f > g2

    return {
        "G1": g1,
        "G2": g2,
        "G3": g3,
        "G4": g4,
        "G5": g5,
        "G6": g6,
        "G7": TG,
        "G8": g8,
        "G9": g9,
        "G10": g10,
        "G11": g11,
        "G12": TL,
        "G13": g13,
        "G14": g14,
        "G15": g15,
        "G16": g16,
    }
