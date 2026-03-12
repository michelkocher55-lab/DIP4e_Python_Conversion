from pathlib import Path as _Path
import os as _os

from typing import Any


class Chapter07Mixin:
    def figure725(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter07 script `Figure725.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.libdipum.data_path import dip_data

            # %% Data
            fIR = imread(dip_data("WashingtonDC-Band4-NearInfrared-512.tif"))
            fR = imread(dip_data("WashingtonDC-Band3-Red-512.tif"))
            fG = imread(dip_data("WashingtonDC-Band2-Green-512.tif"))
            fB = imread(dip_data("WashingtonDC-Band1-Blue-512.tif"))

            # %% Substitute Red by Infrared
            fRtoIR = np.stack((fIR, fG, fB), axis=2)

            # %% Substitute Green by Infrared
            fGtoIR = np.stack((fR, fIR, fB), axis=2)

            # %% Display
            plt.figure(figsize=(10, 7))

            plt.subplot(2, 3, 1)
            plt.imshow(fR, cmap="gray")
            plt.axis("off")
            plt.title("Red channel")

            plt.subplot(2, 3, 2)
            plt.imshow(fG, cmap="gray")
            plt.axis("off")
            plt.title("Green channel")

            plt.subplot(2, 3, 3)
            plt.imshow(fB, cmap="gray")
            plt.axis("off")
            plt.title("Blue channel")

            plt.subplot(2, 3, 4)
            plt.imshow(fIR, cmap="gray")
            plt.axis("off")
            plt.title("Infra red channel")

            plt.subplot(2, 3, 5)
            plt.imshow(fRtoIR)
            plt.axis("off")
            plt.title("Infra Red -> Red")

            plt.subplot(2, 3, 6)
            plt.imshow(fGtoIR)
            plt.axis("off")
            plt.title("Infra red -> Green")

            plt.tight_layout()
            plt.savefig("Figure725.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure736(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter07 script `Figure736.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from helpers.libdipum.data_path import dip_data

            # %% Data
            img_path = dip_data("lenna-RGB.tif")
            f = img_as_float(imread(img_path))

            # %% Convert to RGB
            R = f[:, :, 0]
            G = f[:, :, 1]
            B = f[:, :, 2]

            # %% Display
            plt.figure(figsize=(8, 8))

            plt.subplot(2, 2, 1)
            plt.imshow(f)
            plt.axis("off")
            plt.title("RGB")

            plt.subplot(2, 2, 2)
            plt.imshow(R, cmap="gray")
            plt.axis("off")
            plt.title("R")

            plt.subplot(2, 2, 3)
            plt.imshow(G, cmap="gray")
            plt.axis("off")
            plt.title("G")

            plt.subplot(2, 2, 4)
            plt.imshow(B, cmap="gray")
            plt.axis("off")
            plt.title("B")

            plt.tight_layout()
            plt.savefig("Figure736.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure737(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter07 script `Figure737.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float

            from helpers.libdip.rgb2hsi4e import rgb2hsi4e
            from helpers.libdipum.data_path import dip_data

            # %% Data
            img_path = dip_data("lenna-RGB.tif")
            f = img_as_float(imread(img_path))

            # %% Extract individual RGB and HSI components
            r = f[:, :, 0]
            g = f[:, :, 1]
            b = f[:, :, 2]
            _ = (r, g, b)

            # %% Transform to HSI
            H = rgb2hsi4e(f)
            h = H[:, :, 0]
            s = H[:, :, 1]
            i = H[:, :, 2]

            # %% Display
            plt.figure(figsize=(10, 3.5))

            plt.subplot(1, 3, 1)
            plt.imshow(h, cmap="gray", vmin=np.min(h), vmax=np.max(h))
            plt.axis("off")
            plt.title("Hue")

            plt.subplot(1, 3, 2)
            plt.imshow(s, cmap="gray", vmin=np.min(s), vmax=np.max(s))
            plt.axis("off")
            plt.title("Saturation")

            plt.subplot(1, 3, 3)
            plt.imshow(i, cmap="gray", vmin=np.min(i), vmax=np.max(i))
            plt.axis("off")
            plt.title("Intensity")

            plt.tight_layout()
            plt.savefig("Figure737.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure738(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter07 script `Figure738.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy.ndimage import convolve
            from skimage.io import imread
            from skimage.util import img_as_float

            from helpers.libdip.rgb2hsi4e import rgb2hsi4e
            from helpers.libdip.hsi2rgb4e import hsi2rgb4e
            from helpers.libdipum.data_path import dip_data

            # %% Data
            img_path = dip_data("lenna-RGB.tif")
            f = img_as_float(imread(img_path))

            # %% Extract individual RGB and HSI components
            r = f[:, :, 0]
            g = f[:, :, 1]
            b = f[:, :, 2]

            # %% Transform to HSI
            H = rgb2hsi4e(f)
            h = H[:, :, 0]
            s = H[:, :, 1]
            i = H[:, :, 2]

            # %% Filter individual RGB components
            w = np.ones((5, 5), dtype=float) / 25.0
            rf = convolve(r, w, mode="nearest")
            gf = convolve(g, w, mode="nearest")
            bf = convolve(b, w, mode="nearest")

            # %% Convert back to RGB format
            fRGB_filtered = np.stack((rf, gf, bf), axis=2)

            # %% Filter Intensity component of HSI image
            If = convolve(i, w, mode="nearest")

            # %% Convert back to HSI
            fHSI_filtered = np.stack((h, s, If), axis=2)

            # %% Convert to RGB for comparisons
            fHSI_filtered = hsi2rgb4e(fHSI_filtered)

            # %% Convert to gray so that differences show clearly
            f1 = (
                0.2989 * fRGB_filtered[:, :, 0]
                + 0.5870 * fRGB_filtered[:, :, 1]
                + 0.1140 * fRGB_filtered[:, :, 2]
            )
            f2 = (
                0.2989 * fHSI_filtered[:, :, 0]
                + 0.5870 * fHSI_filtered[:, :, 1]
                + 0.1140 * fHSI_filtered[:, :, 2]
            )
            d = f1 - f2

            # %% Display
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(fRGB_filtered)
            plt.axis("off")
            plt.title("each component R, G and B are filtered")

            plt.subplot(1, 3, 2)
            plt.imshow(fHSI_filtered)
            plt.axis("off")
            plt.title("Only the intensity is filtered")

            plt.subplot(1, 3, 3)
            plt.imshow(d, cmap="gray", vmin=np.min(d), vmax=np.max(d))
            plt.axis("off")
            plt.title("Difference between the 2")

            plt.tight_layout()
            plt.savefig("Figure738.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure739(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter07 script `Figure739.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy.ndimage import convolve
            from skimage.io import imread
            from skimage.util import img_as_float

            from helpers.libdip.rgb2hsi4e import rgb2hsi4e
            from helpers.libdip.hsi2rgb4e import hsi2rgb4e
            from helpers.libdipum.data_path import dip_data

            # %% Data
            img_path = dip_data("lenna-RGB.tif")
            f = img_as_float(imread(img_path))

            # %% Convert to RGB
            R = f[:, :, 0]
            G = f[:, :, 1]
            B = f[:, :, 2]

            # %% Transform to HSI
            H = rgb2hsi4e(f)
            h = H[:, :, 0]
            s = H[:, :, 1]
            i = H[:, :, 2]

            # %% Laplacian kernel
            w = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float)

            # %% Filtering in RGB domain
            LR = convolve(R, w, mode="nearest")
            LG = convolve(G, w, mode="nearest")
            LB = convolve(B, w, mode="nearest")
            fRGB_filtered = np.stack((LR, LG, LB), axis=2)

            # %% Filtering in HSI domain
            If = convolve(i, w, mode="nearest")
            Hd = img_as_float(h)
            Sd = img_as_float(s)
            fHSI_filtered = np.stack((Hd, Sd, If), axis=2)

            # %% Back to RGB domain
            RGB1 = hsi2rgb4e(fHSI_filtered)

            # %% Convert to gray so that differences will show up clearly
            f1 = (
                0.2989 * fRGB_filtered[:, :, 0]
                + 0.5870 * fRGB_filtered[:, :, 1]
                + 0.1140 * fRGB_filtered[:, :, 2]
            )
            f2 = (
                0.2989 * RGB1[:, :, 0] + 0.5870 * RGB1[:, :, 1] + 0.1140 * RGB1[:, :, 2]
            )
            d = f1 - f2

            # %% Display
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(fRGB_filtered)
            plt.axis("off")
            plt.title("each component R, G and B are filtered")

            plt.subplot(1, 3, 2)
            plt.imshow(RGB1)
            plt.axis("off")
            plt.title("Only the intensity is filtered")

            plt.subplot(1, 3, 3)
            plt.imshow(d, cmap="gray", vmin=np.min(d), vmax=np.max(d))
            plt.axis("off")
            plt.title("Difference between the 2")

            plt.tight_layout()
            plt.savefig("Figure739.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure740(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter07 script `Figure740.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float

            from helpers.libdip.rgb2hsi4e import rgb2hsi4e
            from helpers.libdipum.data_path import dip_data

            # %% Data
            RGB_u8 = imread(dip_data("jupiter-moon-closeup.tif"))
            RGB = img_as_float(RGB_u8)

            # %% Transform to HSI
            HSI = rgb2hsi4e(RGB)
            H = HSI[:, :, 0]
            S = HSI[:, :, 1]
            I = HSI[:, :, 2]

            # %% Mask
            T = 0.1 * np.max(S)
            Mask = S > T

            # %% Masked Hue
            MaskedHue = Mask * H
            Hist, edges = np.histogram(MaskedHue.ravel(), bins=256)
            Bin = 0.5 * (edges[:-1] + edges[1:])

            # %% Segmentation
            X = MaskedHue > 0.9

            # %% Display
            plt.figure(figsize=(10, 14))

            plt.subplot(4, 2, 1)
            plt.imshow(RGB)
            plt.axis("off")
            plt.title("RGB")

            plt.subplot(4, 2, 2)
            plt.imshow(H, cmap="gray", vmin=np.min(H), vmax=np.max(H))
            plt.axis("off")
            plt.title("H")

            plt.subplot(4, 2, 3)
            plt.imshow(S, cmap="gray", vmin=np.min(S), vmax=np.max(S))
            plt.axis("off")
            plt.title("S")

            plt.subplot(4, 2, 4)
            plt.imshow(I, cmap="gray", vmin=np.min(I), vmax=np.max(I))
            plt.axis("off")
            plt.title("I")

            plt.subplot(4, 2, 5)
            plt.imshow(Mask, cmap="gray", vmin=0, vmax=1)
            plt.axis("off")
            plt.title(f"Mask = S > {T:.4f}")

            plt.subplot(4, 2, 6)
            plt.imshow(
                MaskedHue, cmap="gray", vmin=np.min(MaskedHue), vmax=np.max(MaskedHue)
            )
            plt.axis("off")
            plt.title("Masked Hue")

            plt.subplot(4, 2, 7)
            plt.bar(Bin, Hist, width=(Bin[1] - Bin[0]) if len(Bin) > 1 else 1.0)
            plt.title("Hist (Masked Hue)")
            plt.gca().set_box_aspect(1)
            plt.axis("tight")

            plt.subplot(4, 2, 8)
            plt.imshow(X, cmap="gray", vmin=0, vmax=1)
            plt.axis("off")
            plt.title("Segmentation")

            plt.tight_layout()
            plt.savefig("Figure740.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure742(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter07 script `Figure742.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from helpers.libdipum.data_path import dip_data

            # %% Data
            RGB = img_as_float(imread(dip_data("jupiter-moon-closeup.tif")))
            R = RGB[:, :, 0]
            G = RGB[:, :, 1]
            B = RGB[:, :, 2]

            # %% Crop (MATLAB rect = [x, y, width, height], inclusive span)
            x, y, w, h = 60, 240, 37, 80
            RGBCrop = RGB[y : y + h + 1, x : x + w + 1, :]
            RCrop = RGBCrop[:, :, 0]
            GCrop = RGBCrop[:, :, 1]
            BCrop = RGBCrop[:, :, 2]

            # %% Segmentation
            mean_r = np.mean(RCrop)
            std_r = np.std(RCrop, ddof=1)
            mean_g = np.mean(GCrop)
            std_g = np.std(GCrop, ddof=1)
            mean_b = np.mean(BCrop)
            std_b = np.std(BCrop, ddof=1)

            k = 1.25
            X = (
                (R > mean_r - k * std_r)
                & (R < mean_r + k * std_r)
                & (G > mean_g - k * std_g)
                & (G < mean_g + k * std_g)
                & (B > mean_b - k * std_b)
                & (B < mean_b + k * std_b)
            )

            # %% Display
            plt.figure(figsize=(9, 7))

            plt.subplot(2, 2, 1)
            plt.imshow(RGB)
            plt.axis("off")
            plt.title("RGB")

            plt.subplot(2, 2, 2)
            plt.imshow(RGBCrop)
            plt.axis("off")
            plt.title("RGB cropped")

            plt.subplot(2, 2, 3)
            plt.imshow(X, cmap="gray", vmin=0, vmax=1)
            plt.axis("off")
            plt.title("Segmentation")

            plt.tight_layout()
            plt.savefig("Figure742.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure744(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter07 script `Figure744.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float

            from helpers.libdipum.colorgrad import colorgrad
            from helpers.libdipum.data_path import dip_data

            # %% Data
            RGB = img_as_float(imread(dip_data("lenna-RGB.tif")))
            R = RGB[:, :, 0]
            G = RGB[:, :, 1]
            B = RGB[:, :, 2]
            _ = (R, G, B)

            # %% Gradient
            VectorGradient, Angle, PerPlaneGradient = colorgrad(RGB)
            Diff = VectorGradient - PerPlaneGradient
            _ = Angle

            # %% Display
            plt.figure(figsize=(9, 7))

            plt.subplot(2, 2, 1)
            plt.imshow(RGB)
            plt.axis("off")
            plt.title("RGB")

            plt.subplot(2, 2, 2)
            plt.imshow(
                VectorGradient,
                cmap="gray",
                vmin=np.min(VectorGradient),
                vmax=np.max(VectorGradient),
            )
            plt.axis("off")
            plt.title("Vector gradient")

            plt.subplot(2, 2, 3)
            plt.imshow(
                PerPlaneGradient,
                cmap="gray",
                vmin=np.min(PerPlaneGradient),
                vmax=np.max(PerPlaneGradient),
            )
            plt.axis("off")
            plt.title("Per plane gradient")

            plt.subplot(2, 2, 4)
            plt.imshow(Diff, cmap="gray", vmin=np.min(Diff), vmax=np.max(Diff))
            plt.axis("off")
            plt.title("Difference")

            plt.tight_layout()
            plt.savefig("Figure744.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure746(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter07 script `Figure746.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float

            from helpers.libdipum.imnoise2 import imnoise2
            from helpers.libdip.rgb2hsi4e import rgb2hsi4e
            from helpers.libdipum.data_path import dip_data

            # %% Parameters
            Mu = 0
            Std = 28 / 255.0
            Std = 20 / 255.0

            # %% Data
            RGB = img_as_float(imread(dip_data("lenna-RGB.tif")))
            R = RGB[:, :, 0]
            G = RGB[:, :, 1]
            B = RGB[:, :, 2]

            # %% Noise adding
            Rn, n = imnoise2(R, "gaussian", Mu, Std)
            Gn, n = imnoise2(G, "gaussian", Mu, Std)
            Bn, n = imnoise2(B, "gaussian", Mu, Std)
            _ = n

            RGBn = np.stack((Rn, Gn, Bn), axis=2)

            # %% Convert to HSI
            HSI = rgb2hsi4e(RGBn)
            Hn = HSI[:, :, 0]
            Sn = HSI[:, :, 1]
            In = HSI[:, :, 2]

            # %% Display (Figure746)
            fig1 = plt.figure(figsize=(8, 8))

            plt.subplot(2, 2, 1)
            plt.imshow(Rn, cmap="gray")
            plt.axis("off")
            plt.title("Noisy R")

            plt.subplot(2, 2, 2)
            plt.imshow(Gn, cmap="gray", vmin=np.min(Gn), vmax=np.max(Gn))
            plt.axis("off")
            plt.title("Noisy G")

            plt.subplot(2, 2, 3)
            plt.imshow(Bn, cmap="gray", vmin=np.min(Bn), vmax=np.max(Bn))
            plt.axis("off")
            plt.title("Noisy B")

            plt.subplot(2, 2, 4)
            plt.imshow(RGBn)
            plt.axis("off")
            plt.title("Noisy RGB")

            plt.tight_layout()
            fig1.savefig("Figure746.png")

            # %% Display (Figure747)
            fig2 = plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(Hn, cmap="gray", vmin=np.min(Hn), vmax=np.max(Hn))
            plt.axis("off")
            plt.title("Hue")

            plt.subplot(1, 3, 2)
            plt.imshow(Sn, cmap="gray", vmin=np.min(Sn), vmax=np.max(Sn))
            plt.axis("off")
            plt.title("Saturation")

            plt.subplot(1, 3, 3)
            plt.imshow(In, cmap="gray", vmin=np.min(In), vmax=np.max(In))
            plt.axis("off")
            plt.title("Intensity")

            plt.tight_layout()
            fig2.savefig("Figure747.png")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure748(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter07 script `Figure748.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float

            from helpers.libdipum.imnoise2 import imnoise2
            from helpers.libdip.rgb2hsi4e import rgb2hsi4e
            from helpers.libdipum.data_path import dip_data

            # %% Parameters
            Ps = 0.05
            Pp = 0.05

            # %% Data
            RGB = img_as_float(imread(dip_data("lenna-RGB.tif")))
            R = RGB[:, :, 0]
            G = RGB[:, :, 1]
            B = RGB[:, :, 2]

            # %% Noise adding
            Gn, n = imnoise2(G, "salt & pepper", Ps, Pp)
            _ = n
            RGBn = np.stack((R, Gn, B), axis=2)

            # %% Convert to HSI
            HSI = rgb2hsi4e(RGBn)
            Hn = HSI[:, :, 0]
            Sn = HSI[:, :, 1]
            In = HSI[:, :, 2]

            # %% Display
            plt.figure(figsize=(8, 8))

            plt.subplot(2, 2, 1)
            plt.imshow(RGBn)
            plt.axis("off")
            plt.title("Noisy RGB")

            plt.subplot(2, 2, 2)
            plt.imshow(Hn, cmap="gray", vmin=np.min(Hn), vmax=np.max(Hn))
            plt.axis("off")
            plt.title("Hue")

            plt.subplot(2, 2, 3)
            plt.imshow(Sn, cmap="gray", vmin=np.min(Sn), vmax=np.max(Sn))
            plt.axis("off")
            plt.title("Saturation")

            plt.subplot(2, 2, 4)
            plt.imshow(In, cmap="gray", vmin=np.min(In), vmax=np.max(In))
            plt.axis("off")
            plt.title("Intensity")

            plt.tight_layout()
            plt.savefig("Figure748.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure749(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter07 script `Figure749.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            import numpy as np

            from helpers.libdip.im2jpeg4e import im2jpeg4e
            from helpers.libdip.jpeg2im4e import jpeg2im4e
            from helpers.libdipum.imratio import imratio
            from helpers.libdipum.compare import compare
            from helpers.libdipum.data_path import dip_data

            # %% Parameters
            Quality = 20

            # %% Data
            RGB = imread(dip_data("Fig0604(a)(iris).tif"))
            R = RGB[:, :, 0]
            G = RGB[:, :, 1]
            B = RGB[:, :, 2]

            # %% Compression JPEG
            y = im2jpeg4e(R, Quality)
            RHat = jpeg2im4e(y)
            CR_R = imratio(R, y)
            RMSE_R = compare(R, RHat, 0)

            y = im2jpeg4e(G, Quality)
            GHat = jpeg2im4e(y)
            CR_G = imratio(G, y)
            RMSE_G = compare(G, GHat, 0)

            y = im2jpeg4e(B, Quality)
            BHat = jpeg2im4e(y)
            CR_B = imratio(B, y)
            RMSE_B = compare(B, BHat, 0)

            RGBCompressed = np.stack((RHat, GHat, BHat), axis=2)
            _ = (RMSE_R, RMSE_G, RMSE_B)

            # %% Display
            plt.figure(figsize=(10, 4))

            plt.subplot(1, 2, 1)
            plt.imshow(RGB)
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(RGBCompressed)
            plt.axis("off")
            plt.title(f"CR = {CR_R:.3g}, {CR_G:.3g}, {CR_B:.3g}")

            plt.tight_layout()
            plt.savefig("Figure749.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

