from pathlib import Path as _Path
import os as _os

from typing import Any


class Chapter03Mixin:
    def figure310(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure310.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage import filters
            from skimage.io import imread
            from helpers.libdipum.intensityTransformations import intensityTransformations
            from helpers.libdipum.data_path import dip_data

            # Image loading
            img_name = dip_data("pollen-lowcontrast.tif")
            f = imread(img_name)
            if f.ndim == 3:
                f = f[:, :, 0]

            # TXFun = [0, 1/8, 7/8, 1]; % For input = 0, 1/3, 2/3, 1
            TXFun = np.array([0, 1 / 8, 7 / 8, 1])

            # Contrast stretching
            # f needs to be float [0,1] or intensityTransformations handles it?
            # intensityTransformations handles conversion to float then applies specified.
            g = intensityTransformations(f, "specified", TXFun)

            # Otsu threshold
            # skimage.filters.threshold_otsu needs input in certain range?
            # g is float [0,1] from specified transform (if inputs were standard).
            # intensityTransformations returns float for 'specified'.

            thresh = filters.threshold_otsu(g)
            X = g > thresh

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            # 1. Plot Law
            # axis inputs: 0, 1/3, 2/3, 1
            x_law = np.linspace(0, 1, 4)
            axes[0, 0].plot(x_law, TXFun, "b-o")
            axes[0, 0].set_title("Contrast stretching law")
            axes[0, 0].set_xlim([0, 1])
            axes[0, 0].set_ylim([0, 1])
            axes[0, 0].grid(True)

            # 2. Input
            axes[0, 1].imshow(f, cmap="gray", vmin=0, vmax=255)
            axes[0, 1].set_title("Input image")
            axes[0, 1].axis("off")

            # 3. Output
            axes[1, 0].imshow(g, cmap="gray", vmin=0, vmax=255)
            axes[1, 0].set_title("Output image")
            axes[1, 0].axis("off")

            # 4. Binary
            axes[1, 1].imshow(X, cmap="gray")
            axes[1, 1].set_title("Otsu applied to output image")
            axes[1, 1].axis("off")

            plt.tight_layout()
            plt.savefig("Figure310.png")
            print("Saved Figure310.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure314(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure314.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.libdipum.data_path import dip_data

            # Image loading (Exact path)
            img_name = dip_data("trophozoite.tif")
            f = imread(img_name)
            if f.ndim == 3:
                f = f[:, :, 0]

            # Bit slicing
            # MATLAB loop implementation:
            # Res = f
            # for i = 1:8
            #    Bit(:,:,i) = Res >= 2^(NBits-i)
            #    Res = Res - Bit * 2^(NBits-i)

            # This extracts bits from MSB (128) down to LSB (1).
            # i=1: 2^7=128. Bit plane 7 (MSB).
            # i=8: 2^0=1.   Bit plane 0 (LSB).

            NBits = 8

            # We can do this efficiently with bitwise AND
            # But replicating loop/order:

            Bits = []
            # Loop 1 to 8
            # Using numpy bitwise is safer than manual subtraction logic for types,
            # but manual subtraction works if types are uint8/int.

            # Let's use standard bitwise extraction which is cleaner in Python
            # But adhere to the order: MSB first (i=1 matches MSB).
            for i in range(NBits):
                # i goes 0..7
                # Power of 2: NBits - 1 - i.
                # i=0 -> 7 (128). i=7 -> 0 (1).
                bit_pos = NBits - 1 - i

                # Extract bit
                # (f >> bit_pos) & 1
                b_img = (f >> bit_pos) & 1
                Bits.append(b_img)

            # Display
            fig, axes = plt.subplots(3, 3, figsize=(10, 10))
            axes = axes.flatten()

            # Plot 1: Original
            axes[0].imshow(f, cmap="gray", vmin=0, vmax=255)
            axes[0].set_title("Original")
            axes[0].axis("off")

            # Plot 2..9: Bit planes
            # MATLAB: title(['b', num2str(i-1)]).
            # i=1 (MSB) -> 'b0'. i=8 (LSB) -> 'b7'.
            # This naming is arguably confusing (usually b7 is MSB), but I will replicate the MATLAB script Output.
            # The MATLAB script uses loop i=1:8.
            # i=1 (MSB). Title 'b0'.

            for i in range(NBits):
                ax = axes[i + 1]
                ax.imshow(Bits[i], cmap="gray")
                ax.set_title(f"b{i}")  # i starts at 0. matches MATLAB 'i-1'.
                ax.axis("off")

            plt.tight_layout()
            plt.savefig("Figure314.png")
            print("Saved Figure314.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure315(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure315.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.libgeneral.ReconstructionUsingBitPlanes import (
                ReconstructionUsingBitPlanes,
            )
            from helpers.libdipum.data_path import dip_data

            # Image loading
            img_name = dip_data("trophozoite.tif")
            f = imread(img_name)
            if f.ndim == 3:
                f = f[:, :, 0]

            NBits = 8

            # 1. Bit Plane Slicing
            # "BitPlanes" needs to be prepared such that index 0 is MSB (iter=1 in MATLAB)
            # MATLAB loop i=1:8. Bit(:,:,i) = ... 2^(NBits-i).
            # i=1 -> 2^7.

            BitPlanes = []
            for i in range(NBits):  # 0..7
                # Power of 2:
                power = NBits - 1 - i
                # i=0 (iter 1) -> 7. 2^7.

                plane = (f >> power) & 1
                BitPlanes.append(plane)

            # 2. Reconstruction
            Recs = []
            SNRs = []

            for iter_count in range(1, NBits + 1):
                # Use first 'iter_count' planes
                Rec, val = ReconstructionUsingBitPlanes(f, BitPlanes, NBits, iter_count)
                Recs.append(Rec)
                SNRs.append(val)

            # Display
            fig, axes = plt.subplots(2, 4, figsize=(15, 8))
            axes = axes.flatten()

            for i in range(NBits):
                ax = axes[i]
                rec_img = Recs[i]
                snr_v = SNRs[i]

                ax.imshow(rec_img, cmap="gray", vmin=0, vmax=255)
                ax.set_title(f"{i + 1} MSB planes\nSNR = {snr_v:.2f} [dB]")
                ax.axis("off")

            plt.tight_layout()
            plt.savefig("Figure315.png")
            print("Saved Figure315.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure316(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure316.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.libdipum.data_path import dip_data

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
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure320(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure320.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.libdipum.data_path import dip_data
            from skimage import exposure

            # Filenames key map matches Figure316
            files_map = [
                ("dark", "Pollen-dark.tif", "Dark"),
                ("light", "Pollen-light.tif", "Light"),
                ("lowcontrast", "pollen-lowcontrast.tif", "Low Contrast"),
                ("highcontrast", "Pollen-high-contrast.tif", "High Contrast"),
            ]

            # Data storage
            originals = []
            equalized = []

            for key, fname, title in files_map:
                img = imread(dip_data(fname))
                if img.ndim == 3:
                    img = img[:, :, 0]

                originals.append(img)

                # Histeq
                # exposure.equalize_hist returns float [0,1].
                # We convert back to uint8 [0,255] for consistency with MATLAB display.
                eq_img_float = exposure.equalize_hist(img)
                eq_img = (eq_img_float * 255).astype(np.uint8)

                equalized.append(eq_img)

            # Display
            # 3x4 grid.
            # Row 1: Originals
            # Row 2: Equalized
            # Row 3: Histograms of Equalized

            titles = [x[2] for x in files_map]

            fig, axes = plt.subplots(3, 4, figsize=(16, 12))

            for i in range(4):
                # Row 1: Original
                ax1 = axes[0, i]
                ax1.imshow(originals[i], cmap="gray", vmin=0, vmax=255)
                ax1.set_title(titles[i])
                ax1.axis("off")

                # Row 2: Equalized
                ax2 = axes[1, i]
                ax2.imshow(equalized[i], cmap="gray", vmin=0, vmax=255)
                # ax2.set_title('Equalized')
                ax2.axis("off")

                # Row 3: Histogram of Equalized
                ax3 = axes[2, i]
                img_eq = equalized[i]
                counts, bins = np.histogram(img_eq.ravel(), bins=256, range=(0, 255))

                ax3.bar(bins[:-1], counts, width=1, color="black", align="edge")
                ax3.set_xlim([0, 255])
                ax3.set_ylim([0, counts.max() * 1.05])
                ax3.set_box_aspect(1)

            plt.tight_layout()
            plt.savefig("Figure320.png")
            print("Saved Figure320.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure324(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure324.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from PIL import Image
            from helpers.libdipum.data_path import dip_data

            # %% Figure 3.24
            # Image of hidden horse and its histogram

            # %% Data
            f = np.array(Image.open(dip_data("hidden-horse.tif")))
            if f.ndim == 3:
                f = f[..., 0]

            # %% Obtain normalized histogram of the image.
            counts, _ = np.histogram(f.ravel(), bins=np.arange(-0.5, 256.5, 1.0))
            pf = counts.astype(float) / f.size

            # %% Display
            fig = plt.figure(1, figsize=(10, 4))

            plt.subplot(1, 2, 1)
            plt.imshow(f, cmap="gray")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.bar(np.arange(256), pf)
            # plt.axis('square')

            plt.tight_layout()
            fig.savefig("Figure324.png", dpi=150, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure325(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure325.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from PIL import Image
            from skimage.exposure import equalize_hist
            from helpers.libdipum.data_path import dip_data

            # %% Fig. 3.25
            # Histogram equalization of hidden horse image

            # %% Data
            f = np.array(Image.open(dip_data("hidden-horse.tif")))
            if f.ndim == 3:
                f = f[..., 0]

            # %% Obtain the histogram equalization intensity transformation function
            hfn, _ = np.histogram(f.ravel(), bins=np.arange(-0.5, 256.5, 1.0))
            hfn = hfn.astype(float) / f.size
            thf = np.cumsum(hfn)

            # %% Histogram equalized image (256 levels)
            g_float = equalize_hist(f, nbins=256)
            g = np.clip(np.round(255 * g_float), 0, 255).astype(np.uint8)

            # %% Normalized histogram of equalized image
            hng, _ = np.histogram(g.ravel(), bins=np.arange(-0.5, 256.5, 1.0))
            hng = hng.astype(float) / g.size

            # %% Display
            fig = plt.figure(1, figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.plot(255 * thf)
            plt.axis("square")
            plt.axis("tight")

            plt.subplot(1, 3, 2)
            plt.imshow(g, cmap="gray")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.bar(np.arange(256), hng)
            plt.axis("square")
            plt.axis("tight")

            plt.tight_layout()
            fig.savefig("Figure325.png", dpi=150, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure326(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure326.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from PIL import Image
            from helpers.libdipum.twomodegauss import twomodegauss
            from helpers.libdipum.data_path import dip_data

            # %% Fig 3.26
            # Histogram specification of hidden horse image

            # %% Data
            f = np.array(Image.open(dip_data("hidden-horse.tif")))
            if f.ndim == 3:
                f = f[..., 0]
            f = f.astype(np.uint8)

            # %% Specified histogram
            p = twomodegauss(0.125, 0.05, 0.9, 0.03, 1, 0.05, 0.0018)
            tsh = np.cumsum(p)

            # %% Obtain the histogram-specified image (MATLAB histeq(f,p)-like CDF matching)
            hf, _ = np.histogram(f.ravel(), bins=np.arange(-0.5, 256.5, 1.0))
            hf = hf.astype(float) / f.size
            cs = np.cumsum(hf)
            ct = np.cumsum(p)

            mapping = np.zeros(256, dtype=np.uint8)
            for r in range(256):
                # First target gray level whose CDF reaches source CDF.
                mapping[r] = np.searchsorted(ct, cs[r], side="left")

            g = mapping[f]

            hns, _ = np.histogram(g.ravel(), bins=np.arange(-0.5, 256.5, 1.0))
            hns = hns.astype(float) / g.size

            # %% Display
            fig = plt.figure(1, figsize=(10, 8))

            plt.subplot(2, 2, 1)
            plt.plot(p)
            plt.axis("square")
            plt.axis("tight")

            plt.subplot(2, 2, 2)
            plt.plot(255 * tsh)
            plt.axis("square")
            plt.axis("tight")

            plt.subplot(2, 2, 3)
            plt.imshow(g, cmap="gray")
            plt.axis("off")

            plt.subplot(2, 2, 4)
            plt.bar(np.arange(256), hns)
            plt.axis("square")
            plt.axis("tight")

            plt.tight_layout()
            fig.savefig("Figure326.png", dpi=150, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure328(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure328.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.exposure import equalize_hist, histogram
            from helpers.libdipum.spechist import spechist
            from helpers.libdipum.fun2hist import fun2hist
            from helpers.libdipum.trapezmf import trapezmf
            from helpers.libdipum.data_path import dip_data

            # Image loading
            img_path = dip_data("hidden-horse.tif")
            f = imread(img_path)
            if f.ndim == 3:
                f = f[:, :, 0]

            M, N = f.shape

            # Construct a non-normalized uniform histogram
            # z = 1:256
            z = np.arange(1, 257)

            # fun = trapezmf(z, 0, 0, 256, 256)
            # This creates a uniform distributions of 1s.
            tmf = trapezmf(z, 0, 0, 256, 256)

            # H = fun2hist(fun, M*N)
            H = fun2hist(tmf, M * N)

            # "Normal" histogram equalized image
            # histeq in MATLAB returns [0, 255]? Or [0, 1]?
            # In skimage, equalize_hist returns [0, 1].
            geq_norm = equalize_hist(f, nbins=256)
            geq = (geq_norm * 255).astype(np.uint8)

            # Histogram specified image
            gsp = spechist(f, H)

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes = axes.flatten()

            # 1. Original (Not explicitly shown in MATLAB script subplot logic which seems slightly off, but typical)
            # MATLAB script writes: subplot(2,2,1); imshow(g).
            # But 'g' corresponds to spechist output in standard notation, OR original 'f'?
            # Given the script variable names: f (input), geq (histeq), gsp (spechist).
            # If variable 'g' is not defined, it might be a typo in user's provided file or referring to previous workspace var.
            # I will assume subplot 1 should show the Original image 'f'.
            axes[0].imshow(f, cmap="gray", vmin=0, vmax=255)
            axes[0].set_title("Original")
            axes[0].axis("off")

            # 2. Hist Eq
            axes[1].imshow(geq, cmap="gray", vmin=0, vmax=255)
            axes[1].set_title("Hist Eq")
            axes[1].axis("off")

            # 3. Specified
            axes[2].imshow(gsp, cmap="gray", vmin=0, vmax=255)
            axes[2].set_title("Hist Specified")
            axes[2].axis("off")

            # 4. Histogram of Specified
            counts, centers = histogram(gsp, nbins=256, source_range="image")
            axes[3].bar(centers, counts, width=1)
            axes[3].set_title("Histogram of Specified")
            axes[3].set_xlim([0, 255])

            plt.tight_layout()
            plt.savefig("Figure328.png")
            print("Saved Figure328.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure329(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure329.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.exposure import histogram
            from helpers.libdipum.exacthist import exacthist
            from helpers.libdipum.fun2hist import fun2hist
            from helpers.libdipum.trapezmf import trapezmf
            from helpers.libdipum.data_path import dip_data

            # Image loading
            img_path = dip_data("hidden-horse.tif")
            f = imread(img_path)
            if f.ndim == 3:
                f = f[:, :, 0]

            M, N = f.shape

            # Generate function: uniform then ramp tail.
            # z = 1:256
            z = np.arange(1, 257)
            # trapezmf(z, 0, 0, 64, 256)
            # This means:
            # a=0, b=0 (flat start)
            # c=64 (start of ramp down)
            # d=256 (end of ramp down)
            # So flat 1.0 from 0 to 64, then ramp down to 0 at 256.
            fun = trapezmf(z, 0, 0, 64, 256)

            # Generate histogram
            H = fun2hist(fun, M * N)

            # Generate image with this histogram
            # exacthist(f, H) returns (g, Hg, lexiOrder)
            # Note: exacthist.m returns [g, Hg, lexiOrder]. My internal implementation of exacthist.py does same.
            # MATLAB script calls: gramp = exacthist(...)
            # If function returns multiple values, MATLAB assigns first to gramp.
            # Python: tuple unpacking or index [0].

            result = exacthist(f, H)
            gramp = result[0]

            # Calculate histogram of result for display check
            counts, centers = histogram(gramp, nbins=256, source_range="image")

            # Display
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # 1. Specified Histogram H
            axes[0].bar(np.arange(256), H, width=1)
            axes[0].set_title("Specified Histogram H")
            axes[0].set_xlim([0, 255])

            # 2. Result Image
            axes[1].imshow(gramp, cmap="gray", vmin=0, vmax=255)
            axes[1].set_title("Result gramp")
            axes[1].axis("off")

            # 3. Result Histogram Hg
            axes[2].bar(centers, counts, width=1)
            axes[2].set_title("Result Histogram Hg")
            axes[2].set_xlim([0, 255])

            plt.tight_layout()
            plt.savefig("Figure329.png")
            print("Saved Figure329.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure330(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure330.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_ubyte
            from helpers.libdipum.exacthist import exacthist
            from helpers.libdipum.fun2hist import fun2hist
            from helpers.libdipum.data_path import dip_data

            def imhist(img: Any):
                """Compute histogram for uint8 image with 256 bins [0, 255]."""
                # Create 256 bins
                hist, _ = np.histogram(img.flatten(), 256, [0, 256])
                return hist

            # Data Loading
            img_path = dip_data("mars_moon_phobos.tif")
            f = imread(img_path)
            if f.ndim == 3:
                f = f[:, :, 0]

            # Ensure f is uint8? exacthist expects uint8 usually, or converts internally?
            # exacthist requires integer values usually (0-255).
            # If file is uint8, fine.
            # MATLAB `imhist` works on uint8.

            if f.dtype != np.uint8:
                # Convert to uint8 only if range allows, but exact specs might work on other ranges.
                # However, typically exacthist for images assumes discrete levels.
                # Let's assume input is uint8 or convertible.
                f = img_as_ubyte(f)

            M, N = f.shape

            # Histogram of f
            Hf = imhist(f)

            # Histogram Specification
            # H(1:256) = M*N/256
            H_target = np.full(256, (M * N) / 256.0)

            # H = fun2hist(H, M*N)
            # fun2hist standardizes the histogram to sum exactly to MN and be integers.
            H_target = fun2hist(H_target, M * N)

            # g = exacthist(f, H)
            g = exacthist(f, H_target)
            g = g[0]

            # Hz = imhist(g)
            Hg = imhist(g)

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            axes[0, 0].imshow(f, cmap="gray", vmin=0, vmax=255)
            axes[0, 0].set_title("Original Image")
            axes[0, 0].axis("off")

            # Plot histogram. MATLAB uses bar or stem or plot? "plot(Hf)"
            axes[0, 1].plot(Hf, color="black")
            axes[0, 1].set_title("Histogram of f")
            axes[0, 1].set_xlim([0, 255])

            axes[1, 0].plot(Hg, color="black")
            axes[1, 0].set_title("Histogram of g")
            axes[1, 0].set_xlim([0, 255])

            axes[1, 1].imshow(g, cmap="gray", vmin=0, vmax=255)
            axes[1, 1].set_title("Exact Hist. Spec. Result")
            axes[1, 1].axis("off")

            plt.tight_layout()
            plt.savefig("Figure330.png")
            print("Saved Figure330.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure331(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure331.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_ubyte

            from helpers.libdipum.exacthist import exacthist
            from helpers.libdipum.fun2hist import fun2hist
            from helpers.libdipum.sigmamf import sigmamf  # Assuming this exists based on ls
            from helpers.libdipum.data_path import dip_data

            def imhist(img: Any, bins: Any = 256):
                """imhist."""
                hist, _ = np.histogram(img.flatten(), bins, [0, bins])
                return hist

            # Data Loading
            img_path = dip_data("mars_moon_phobos.tif")
            f = imread(img_path)
            if f.ndim == 3:
                f = f[:, :, 0]
            # Ensure uint8 for exacthist
            if f.dtype != np.uint8:
                f = img_as_ubyte(f)

            M, N = f.shape

            # Mask
            # idx = find(f == 0); mask = ones; mask(idx) = 0;
            # Convert to boolean mask where True means "Process this pixel"
            # exacthist mask: "Only pixels where mask > 0 are processed."
            # So mask should be 0 for background.
            mask = np.ones((M, N), dtype=bool)
            mask[f == 0] = False

            # Part 1: Exact Histogram Equalization (Uniform)
            # H(1:256) = M*N/256 roughly.
            # Note: mask reduces number of pixels?
            # MATLAB: H(1:256) = M*N/256.
            # Wait, if we use a mask, exacthist processes only masked pixels.
            # The target H usually sums to the number of *masked* pixels or *total* pixels?
            # exacthist documentation/implementation (which I wrote) checks sum against `num_active`.
            # And normalizes H if mismatch.
            # MATLAB script sets H based on M*N (total pixels).
            # `fun2hist` (MATLAB) normalizes to M*N usually.
            # If `exacthist` sees mask, it will likely see mismatch (Sum(H)=MN != NumMasked).
            # My `exacthist.py` implementation handles this by rescaling H to match `num_active`.
            # So strictly following MATLAB script is fine.

            H_uniform = np.full(256, (M * N) / 256.0)
            H_uniform = fun2hist(H_uniform, M * N)

            # gmasked = exacthist(f, H, mask)
            gmasked, _, _ = exacthist(f, H_uniform, mask)

            # Part 2: Custom Histogram (Sigmoid)
            # z = 1:256;
            # sig = 0.065 + sigmamf(z, 32, 256);
            # Check sigmamf signature in Python.
            # Assuming sigmamf(x, a, c) or sigmamf(x, [a, c])?
            # MATLAB: sigmamf(x, [a c]).
            # MATLAB script: sigmamf(z, 32, 256). This implies 2 separate args?
            # Or maybe custom `sigmamf`?
            # I'll try calling with separate args `sigmamf(z, 32, 256)`. If it fails, I'll try list.
            # Actually, let's peek lexiorder logic or similar? No time.
            # I will assume standard python transcription of typical usage.

            z = np.arange(1, 257)  # 1 to 256

            # Try calling sigmamf.
            try:
                # Assuming arguments a=32, c=256?
                # MATLAB sigmamf(x, [a c]).
                # But script says sigmamf(z,32,256).
                # Maybe the script uses a custom version that takes unpacked args.
                sig_vals = sigmamf(z, 32, 256)
            except TypeError:
                # Retry with list
                sig_vals = sigmamf(z, [32, 256])

            sig = 0.065 + sig_vals

            # Hup = fun2hist(sig, M*N)
            Hup = fun2hist(sig, M * N)

            # [gup, Hg] = exacthist(f, Hup, mask)
            gup, Hg, _ = exacthist(f, Hup, mask)

            # Display
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # 1. Mask
            axes[0, 0].imshow(mask, cmap="gray")
            axes[0, 0].set_title("Mask")
            axes[0, 0].axis("off")

            # 2. Masked Equalized
            axes[0, 1].imshow(gmasked, cmap="gray")
            axes[0, 1].set_title("Masked Hist. Eq.")
            axes[0, 1].axis("off")

            # 3. Hup Bar
            axes[0, 2].bar(np.arange(256), Hup, color="black", width=1)
            axes[0, 2].set_title("Target Histogram Hup")
            axes[0, 2].set_xlim([0, 255])

            # 4. Hg Bar (Result Hist)
            axes[1, 0].bar(np.arange(256), Hg, color="black", width=1)
            axes[1, 0].set_title("Result Histogram Hg")
            axes[1, 0].set_xlim([0, 255])

            # 5. gup (Result Image)
            axes[1, 1].imshow(gup, cmap="gray")
            axes[1, 1].set_title("Masked Specified Result (gup)")
            axes[1, 1].axis("off")

            # Empty slot
            axes[1, 2].axis("off")

            plt.tight_layout()
            plt.savefig("Figure331.png")
            print("Saved Figure331.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure332(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure332.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage import exposure
            from helpers.libdipum.data_path import dip_data

            # Image loading
            img_name = dip_data("hidden-symbols.tif")

            f = imread(img_name)
            if f.ndim == 3:
                f = f[:, :, 0]

            NR, NC = f.shape
            Local = 3

            # Global HE
            # skimage histeq returns float
            g1_float = exposure.equalize_hist(f)
            g1 = (g1_float * 255).astype(np.uint8)

            # Local HE
            # MATLAB: NumTiles = [floor(NR/Local), floor(NC/Local)]
            # If Local=3, Tile Size is approx 3x3.
            # skimage equalize_adapthist takes kernel_size.
            # kernel_size = [Local, Local] (integer dimensions of the tile/window).

            # ClipLimit = 1.
            # In skimage, clip_limit is normalized between 0 and 1.
            # MATLAB 'ClipLimit' is also 0 to 1.
            # So we use clip_limit=1.0.

            # Note: equalize_adapthist acts on float [0,1] image usually or converts it.
            # It returns float [0,1].

            # tile_size calculation:
            # MATLAB: NumTiles.
            # skimage: kernel_size.
            # If NumTiles = NR/3. Then pixel size of tile is NR/(NR/3) = 3.
            # So kernel_size = (3, 3).

            # However, skimage's kernel_size corresponds to the window size for sliding window?
            # No, equalize_adapthist uses CLAHE which operates on tiles.
            # "kernel_size: integer or list-like, optional. Defines the shape of contextual regions used in the algorithm."
            # Default is image_shape/8.

            # Implementation:
            g2_float = exposure.equalize_adapthist(
                f, kernel_size=(Local, Local), clip_limit=1.0
            )
            g2 = (g2_float * 255).astype(np.uint8)

            # Display
            fig, axes = plt.subplots(1, 3, figsize=(10, 10))
            axes = axes.flatten()

            # 1. Original
            axes[0].imshow(f, cmap="gray", vmin=0, vmax=255)
            axes[0].set_title("Original")
            axes[0].axis("off")

            # 2. Global HE
            axes[1].imshow(g1, cmap="gray", vmin=0, vmax=255)
            axes[1].set_title("Global hist eq.")
            axes[1].axis("off")

            # 3. Local HE
            axes[2].imshow(g2, cmap="gray", vmin=0, vmax=255)
            axes[2].set_title(f"Local Size = {Local}")
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig("Figure332.png")
            print("Saved Figure332.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure333(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure333.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from helpers.libdipum.locstats import locstats
            from helpers.libdipum.data_path import dip_data

            # Data Loading

            img_path = dip_data("hidden-symbols.tif")
            f = imread(img_path)
            if f.ndim == 3:
                f = f[:, :, 0]
            f = img_as_float(f)

            # Process
            # param = [22.8, 0, 0.1, 0, 0.1];
            # [g, GMF, GSTDF] = locstats(f, 3, 3, param);
            # Note: locstats implementation might expect params as list or separate args?
            # Usually MATLAB 'param' vector implies a single list/array argument.
            # Checking if locstats.py follows this. Most transcoded funcs do.

            param = [22.8, 0, 0.1, 0, 0.1]

            g, GMF, GSTDF = locstats(f, 3, 3, param)

            # Display
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            axes[0].imshow(f, cmap="gray")
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            axes[1].imshow(g, cmap="gray")
            axes[1].set_title("Locally Enhanced Image")
            axes[1].axis("off")

            plt.tight_layout()
            plt.savefig("Figure333.png")
            print("Saved Figure333.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure339(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure339.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from scipy.ndimage import uniform_filter
            from helpers.libdipum.data_path import dip_data

            # Image loading
            img_path = dip_data("characterTestPattern688.tif")
            f = imread(img_path)
            if f.ndim == 3:
                f = f[:, :, 0]

            f_float = img_as_float(f)

            # Box filters
            # MATLAB imfilter default boundary is 0.
            # scipy.ndimage.uniform_filter default mode is 'reflect'.
            # We should set mode='constant', cval=0.0 to match MATLAB default 'imfilter'.

            gbox3 = uniform_filter(f_float, size=3, mode="constant", cval=0.0)
            gbox11 = uniform_filter(f_float, size=11, mode="constant", cval=0.0)
            gbox21 = uniform_filter(f_float, size=21, mode="constant", cval=0.0)

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes = axes.flatten()

            axes[0].imshow(f, cmap="gray")  # Original
            axes[0].set_title("Original")
            axes[0].axis("off")

            axes[1].imshow(gbox3, cmap="gray")
            axes[1].set_title("Box 3x3")
            axes[1].axis("off")

            axes[2].imshow(gbox11, cmap="gray")
            axes[2].set_title("Box 11x11")
            axes[2].axis("off")

            axes[3].imshow(gbox21, cmap="gray")
            axes[3].set_title("Box 21x21")
            axes[3].axis("off")

            plt.tight_layout()
            plt.savefig("Figure339.png")
            print("Saved Figure339.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure342(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure342.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from scipy.ndimage import correlate
            from helpers.libdipum.gaussiankernel import gaussiankernel
            from helpers.libdipum.data_path import dip_data

            # Image loading
            img_name = dip_data("testpattern1024.tif")
            f = imread(img_name)
            if f.ndim == 3:
                f = f[:, :, 0]

            f = img_as_float(f)

            # 21x21 Gaussian kernel. sig = 3.5
            # MATLAB: gaussiankernel(21,'sampled',3.5,1);
            # Python: returns (w, S)
            # The last argument '1' in MATLAB likely is 'K'.
            # My python implementation: args are (sigma, K).
            gauss3pt5, S = gaussiankernel(21, "sampled", 3.5, 1.0)

            # Normalize
            # gauss3pt5 = gauss3pt5/sum(gauss3pt5(:))
            gauss3pt5 = gauss3pt5 / np.sum(gauss3pt5)

            # Filter. Default padding is zero padding.
            # MATLAB imfilter corresponds to correlation.
            # scipy.ndimage.correlate with mode='constant', cval=0.0 corresponds to zero padding.
            ggauss3pt5 = correlate(f, gauss3pt5, mode="constant", cval=0.0)

            # sig = 7, size 43x43
            gauss7, S2 = gaussiankernel(43, "sampled", 7.0, 1.0)
            gauss7 = gauss7 / np.sum(gauss7)

            ggauss7 = correlate(f, gauss7, mode="constant", cval=0.0)

            # Display
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(f, cmap="gray")
            axes[0].set_title("Original")
            axes[0].axis("off")

            axes[1].imshow(ggauss3pt5, cmap="gray")
            axes[1].set_title("Gaussian 21x21, sigma=3.5")
            axes[1].axis("off")

            axes[2].imshow(ggauss7, cmap="gray")
            axes[2].set_title("Gaussian 43x43, sigma=7")
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig("Figure342.png")
            print("Saved Figure342.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure343(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure343.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from scipy.ndimage import correlate
            from helpers.libdipum.gaussiankernel import gaussiankernel
            from helpers.libdipum.data_path import dip_data

            # Image loading
            img_name = dip_data("testpattern1024.tif")
            f = imread(img_name)
            if f.ndim == 3:
                f = f[:, :, 0]

            f = img_as_float(f)

            # Kernels
            # gauss43 = gaussiankernel(43,'sampled',7,1); % Approx 6 sig
            # gauss85 = gaussiankernel(85,'sampled',7,1); % Approx 12 sig

            gauss43, _ = gaussiankernel(43, "sampled", 7.0, 1.0)
            gauss85, _ = gaussiankernel(85, "sampled", 7.0, 1.0)

            # Normalize the filters
            gauss43 = gauss43 / np.sum(gauss43)
            gauss85 = gauss85 / np.sum(gauss85)

            # Filter (default padding is zero padding)
            ggauss43 = correlate(f, gauss43, mode="constant", cval=0.0)
            ggauss85 = correlate(f, gauss85, mode="constant", cval=0.0)

            # Compare
            # diff = imsubtract(ggauss43, ggauss85);
            # Since images are floats, simple subtraction.
            diff = ggauss43 - ggauss85
            # MATLAB imsubtract on doubles clip negatives?
            # "Z = imsubtract(X,Y) subtracts each element ...
            # If the output array is of integer class, then all negative results are truncated to zero."
            # If double, negative values are preserved.
            # HOWEVER, the script later computes `255 * max(diff(:))`.
            # Wait, usually for display one might care about abs difference, but `imsubtract` on double is just minus.
            # The title implies we look at the deviation.
            # I will stick to simple subtraction as f is double.

            max_diff = 255 * np.max(diff)

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes = axes.flatten()

            axes[0].imshow(f, cmap="gray")
            axes[0].set_title("Original")
            axes[0].axis("off")

            axes[1].imshow(ggauss43, cmap="gray")
            axes[1].set_title("Gaussian 43x43")
            axes[1].axis("off")

            axes[2].imshow(ggauss85, cmap="gray")
            axes[2].set_title("Gaussian 85x85")
            axes[2].axis("off")

            # Subplot 4: diff
            axes[3].imshow(diff, cmap="gray")
            axes[3].set_title(f"Diff = {max_diff:.4f}")
            axes[3].axis("off")

            plt.tight_layout()
            plt.savefig("Figure343.png")
            print("Saved Figure343.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure344(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure344.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy.ndimage import correlate, uniform_filter
            from helpers.libdipum.gaussiankernel import gaussiankernel

            # Data: Synthetic image
            f = np.zeros((1024, 1024), dtype=np.uint8)
            f[127:896, 191:320] = 255

            # Kernels
            # box = ones(71); box = box/sum(box(:));
            # We can use uniform_filter.

            # Gaussian
            # gaussian = gaussiankernel(151, 'sampled', 25, 1);
            gaussian, _ = gaussiankernel(151, "sampled", 25.0, 1.0)
            gaussian = gaussian / np.sum(gaussian)

            # Filtering
            # MATLAB: imfilter(f, box, 'replicate')
            # scipy: uniform_filter or correlated. uniform_filter is faster for box.
            # mode='nearest' corresponds to 'replicate'.

            f_float = f.astype(float) / 255.0

            # Box filtering
            # uniform_filter size=71
            gbox = uniform_filter(f_float, size=71, mode="nearest")

            # Gaussian filtering
            ggauss = correlate(f_float, gaussian, mode="nearest")

            # Display
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(f, cmap="gray", vmin=0, vmax=255)
            axes[0].set_title("Original")
            axes[0].axis("off")

            axes[1].imshow(gbox, cmap="gray")
            axes[1].set_title("Box Filter 71x71")
            axes[1].axis("off")

            axes[2].imshow(ggauss, cmap="gray")
            axes[2].set_title("Gaussian Filter 151x151, sigma=25")
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig("Figure344.png")
            print("Saved Figure344.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure345(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure345.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from scipy.ndimage import correlate
            from helpers.libdipum.gaussiankernel import gaussiankernel
            from helpers.libdipum.data_path import dip_data

            # Image loading
            img_name = dip_data("testpattern1024.tif")
            f = imread(img_name)
            if f.ndim == 3:
                f = f[:, :, 0]

            f = img_as_float(f)

            # Kernel
            # gauss187 = gaussiankernel(187, 'sampled', 31, 1);
            gauss187, _ = gaussiankernel(187, "sampled", 31.0, 1.0)
            gauss187 = gauss187 / np.sum(gauss187)

            # Filtering with different boundary conditions

            # 1. Zero padding
            # MATLAB: imfilter(f, h) (default is zero)
            # scipy: mode='constant', cval=0.0
            gzeropad = correlate(f, gauss187, mode="constant", cval=0.0)

            # 2. Symmetric padding
            # MATLAB: imfilter(f, h, 'symmetric')
            # MATLAB 'symmetric' pads with mirror reflections of itself.
            # padarray([1 2 3], 2, 'symmetric') -> [2 1 1 2 3 3 2]. (Repeats edge pixel).
            # scipy.ndimage.correlate mode='reflect' -> d c b a | a b c d | d c b a (Repeats edge pixel).
            # mode='mirror' -> d c b | a b c d | c b a (Does not repeat edge pixel).
            # So MATLAB 'symmetric' matches scipy 'reflect'.
            gsymmpad = correlate(f, gauss187, mode="reflect")

            # 3. Replicate padding
            # MATLAB: imfilter(f, h, 'replicate')
            # scipy: mode='nearest'
            greplpad = correlate(f, gauss187, mode="nearest")

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes = axes.flatten()

            axes[0].imshow(f, cmap="gray", vmin=0, vmax=1)
            axes[0].set_title("Original")
            axes[0].axis("off")

            axes[1].imshow(gzeropad, cmap="gray", vmin=0, vmax=1)
            axes[1].set_title("Zero Padding")
            axes[1].axis("off")

            axes[2].imshow(gsymmpad, cmap="gray", vmin=0, vmax=1)
            axes[2].set_title("Symmetric Padding")
            axes[2].axis("off")

            axes[3].imshow(greplpad, cmap="gray", vmin=0, vmax=1)
            axes[3].set_title("Replicate Padding")
            axes[3].axis("off")

            plt.tight_layout()
            plt.savefig("Figure345.png")
            print("Saved Figure345.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure346(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure346.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from scipy.ndimage import correlate1d
            from helpers.libdipum.data_path import dip_data

            # Data
            img_name = dip_data("testpattern4096.tif")
            f = imread(img_name)
            if f.ndim == 3:
                f = f[:, :, 0]
            f = img_as_float(f)

            def gaussian_1d(size: Any, sigma: Any):
                """gaussian_1d."""
                radius = (size - 1) // 2
                x = np.arange(-radius, radius + 1, dtype=np.float64)
                w = np.exp(-(x**2) / (2.0 * sigma * sigma))
                w /= np.sum(w)
                return w

            # Separable kernels equivalent to normalized 2D Gaussian kernels
            w187 = gaussian_1d(187, 31.0)
            w745 = gaussian_1d(745, 124.0)

            # Filtering (MATLAB imfilter(..., 'symmetric') -> mode='reflect')
            g187 = correlate1d(f, w187, axis=1, mode="reflect")
            g187 = correlate1d(g187, w187, axis=0, mode="reflect")

            g745 = correlate1d(f, w745, axis=1, mode="reflect")
            g745 = correlate1d(g745, w745, axis=0, mode="reflect")

            # Display
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.imshow(f, cmap="gray", vmin=0, vmax=1)
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(g187, cmap="gray", vmin=0, vmax=1)
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(g745, cmap="gray", vmin=0, vmax=1)
            plt.axis("off")

            plt.tight_layout()
            plt.savefig("Figure346.png")
            print("Saved Figure346.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure347(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure347.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from scipy.ndimage import correlate1d
            from helpers.libdipum.data_path import dip_data

            # Image loading
            img_path = dip_data("hickson-compact-group.tif")
            f = imread(img_path)
            if f.ndim == 3:
                f = f[:, :, 0]

            f = img_as_float(f)

            # Kernel (separable Gaussian, spatial domain)
            SIG = 25.0
            ksize = 151
            radius = (ksize - 1) // 2
            x = np.arange(-radius, radius + 1, dtype=np.float64)
            w1d = np.exp(-(x**2) / (2.0 * SIG * SIG))
            w1d /= np.sum(w1d)

            # Filtering (equivalent to 2D Gaussian correlation, but much faster)
            # MATLAB default imfilter boundary is zero padding.
            g = correlate1d(f, w1d, axis=1, mode="constant", cval=0.0)
            g = correlate1d(g, w1d, axis=0, mode="constant", cval=0.0)

            # Thresholding
            gT = g > 0.4

            # Display
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(f, cmap="gray")
            axes[0].set_title("Original")
            axes[0].axis("off")

            axes[1].imshow(g, cmap="gray")
            axes[1].set_title("Smoothed (Lowpass)")
            axes[1].axis("off")

            axes[2].imshow(gT, cmap="gray")
            axes[2].set_title("Thresholded > 0.4")
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig("Figure347.png")
            print("Saved Figure347.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure348fourier(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure348Fourier.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from skimage.filters import threshold_otsu
            from scipy.signal import fftconvolve
            from helpers.libdipum.data_path import dip_data

            def fspecial(type_filter: Any, *args: Any):
                """
                Mimics MATLAB's fspecial function for 'gaussian'.
                """
                if type_filter == "gaussian":
                    hsize = args[0]
                    sigma = args[1]

                    if isinstance(hsize, (int, float)):
                        hsize = (int(hsize), int(hsize))

                    m, n = hsize
                    y, x = np.ogrid[
                        -(m - 1) // 2 : (m - 1) // 2 + 1,
                        -(n - 1) // 2 : (n - 1) // 2 + 1,
                    ]
                    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
                    h[h < np.finfo(h.dtype).eps * h.max()] = 0
                    sumh = h.sum()
                    if sumh != 0:
                        h /= sumh
                    return h

                raise NotImplementedError(
                    f"Filter type '{type_filter}' not implemented."
                )

            def imfilter(img: Any, kernel: Any, mode: Any = "constant"):
                """
                Mimics MATLAB's imfilter.
                Uses FFT-based convolution for speed with large kernels.
                """
                if mode == "replicate":
                    pad_mode = "edge"
                elif mode == "symmetric":
                    pad_mode = "reflect"
                elif mode == "circular":
                    pad_mode = "wrap"
                else:
                    pad_mode = "constant"

                kh, kw = kernel.shape

                # For output size preservation with mode='valid':
                # need total padding kh-1 and kw-1, split asymmetrically when even.
                pad_top = (kh - 1) // 2
                pad_bottom = (kh - 1) - pad_top
                pad_left = (kw - 1) // 2
                pad_right = (kw - 1) - pad_left

                if pad_mode == "constant":
                    padded_img = np.pad(
                        img,
                        ((pad_top, pad_bottom), (pad_left, pad_right)),
                        mode=pad_mode,
                        constant_values=0,
                    )
                else:
                    padded_img = np.pad(
                        img,
                        ((pad_top, pad_bottom), (pad_left, pad_right)),
                        mode=pad_mode,
                    )

                output = fftconvolve(padded_img, kernel, mode="valid")
                return np.real(output)

            print("Running Figure348 (Shading Correction)...")

            Sigma = 64
            HSize = 512

            img_name = dip_data("checkerboard1024-shaded.tif")
            f_orig = imread(img_name)
            if f_orig.ndim == 3:
                f_orig = f_orig[:, :, 0]

            f = img_as_float(f_orig)

            h = fspecial("gaussian", HSize, Sigma)

            fs = imfilter(f, h, "replicate")

            epsilon = 1e-6
            g = f / (fs + epsilon)

            try:
                thresh = threshold_otsu(g)
                X = g > thresh
            except ValueError:
                X = np.zeros_like(g, dtype=bool)

            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            axes[0, 0].imshow(f, cmap="gray")
            axes[0, 0].set_title(f"Original Image f\nSize={f.shape}")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(fs, cmap="gray")
            axes[0, 1].set_title(f"Smoothed Image fs\nSigma={Sigma}, Size={HSize}")
            axes[0, 1].axis("off")

            axes[1, 0].imshow(g, cmap="gray")
            axes[1, 0].set_title("Shading Corrected g = f/fs")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(X, cmap="gray")
            axes[1, 1].set_title("Otsu Thresholded (g)")
            axes[1, 1].axis("off")

            plt.tight_layout()
            plt.savefig("Figure348.png")
            print("Saved Figure348.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure348separable(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure348Separable.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from skimage.filters import threshold_otsu
            from scipy.ndimage import correlate

            def fspecial(type_filter: Any, *args: Any):
                """
                Mimics MATLAB's fspecial function for 'gaussian'.
                """
                if type_filter == "gaussian":
                    hsize = args[0]
                    sigma = args[1]

                    if isinstance(hsize, (int, float)):
                        hsize = (hsize, hsize)

                    m, n = hsize
                    # Use ogrid ensuring exact shape m, n
                    # Center is at (m-1)/2, (n-1)/2
                    y, x = np.ogrid[0:m, 0:n]
                    y = y - (m - 1) / 2.0
                    x = x - (n - 1) / 2.0

                    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
                    h[h < np.finfo(h.dtype).eps * h.max()] = 0
                    sumh = h.sum()
                    if sumh != 0:
                        h /= sumh
                    return h
                else:
                    raise NotImplementedError(
                        f"Filter type '{type_filter}' not implemented."
                    )

            from scipy.ndimage import correlate1d
            from helpers.libdipum.data_path import dip_data

            def imfilter(img: Any, kernel: Any, mode: Any = "constant"):
                """
                Mimics MATLAB's imfilter.
                Uses separable convolution if kernel is rank 1.
                """
                # Map MATLAB modes to scipy.ndimage modes
                if mode == "replicate":
                    scipy_mode = "nearest"
                elif mode == "symmetric":
                    scipy_mode = "reflect"
                elif mode == "circular":
                    scipy_mode = "wrap"
                else:
                    scipy_mode = mode  # 'constant'

                # Check for separability using SVD
                # kernel shape (M, N)
                if kernel.ndim == 2:
                    try:
                        u, s, vh = np.linalg.svd(kernel)
                        # Check if rank 1 approx
                        # Ratio of first singular value to total energy or second singular value
                        if s[1] < 1e-5 * s[0]:
                            # Separable
                            # k = s[0] * u[:, 0] * vh[0, :]
                            # Vertical kernel (u[:, 0] * sqrt(s[0]))
                            # Horizontal kernel (vh[0, :] * sqrt(s[0]))

                            # Assign sign/magnitude factors
                            scale = np.sqrt(s[0])
                            k_vert = u[:, 0] * scale
                            k_horz = vh[0, :] * scale

                            # Convolve columns (axis 0) then rows (axis 1)
                            temp = correlate1d(img, k_vert, axis=0, mode=scipy_mode)
                            return correlate1d(temp, k_horz, axis=1, mode=scipy_mode)
                    except Exception:
                        pass  # Fallback to 2D

                # Fallback to standard 2D correlation (slow for large kernels)
                return correlate(img, kernel, mode=scipy_mode)

            # Parameters
            Sigma = 64
            HSize = 512

            # Data Loading
            img_name = dip_data("checkerboard1024-shaded.tif")
            f_orig = imread(img_name)
            if f_orig.ndim == 3:
                f_orig = f_orig[:, :, 0]

            # Convert to float
            f = img_as_float(f_orig)

            # Kernel
            # h = fspecial ('gaussian', HSize, Sigma);
            h = fspecial("gaussian", HSize, Sigma)

            # Filtering
            # fs = imfilter (f, h, 'replicate');
            fs = imfilter(f, h, "replicate")

            # Pointwise division
            # g = double(f)./double(fs);
            epsilon = 1e-6
            g = f / (fs + epsilon)

            # Thresholding
            try:
                thresh = threshold_otsu(g)
                X = g > thresh
            except ValueError:
                X = np.zeros_like(g, dtype=bool)

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            axes[0, 0].imshow(f, cmap="gray")
            axes[0, 0].set_title(f"Original Image f\nSize={f.shape}")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(fs, cmap="gray")
            axes[0, 1].set_title(f"Smoothed Image fs\nSigma={Sigma}, Size={HSize}")
            axes[0, 1].axis("off")

            axes[1, 0].imshow(g, cmap="gray")
            axes[1, 0].set_title("Shading Corrected g = f/fs")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(X, cmap="gray")
            axes[1, 1].set_title("Otsu Thresholded (g)")
            axes[1, 1].axis("off")

            plt.tight_layout()
            plt.savefig("Figure348.png")
            print("Saved Figure348.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure349(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure349.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from scipy.ndimage import correlate
            from scipy.signal import medfilt2d

            from helpers.libdipum.imnoise2 import imnoise2
            from helpers.libdipum.gaussiankernel import gaussiankernel
            from helpers.libdipum.data_path import dip_data

            # Data
            f = imread(dip_data("circuitboard.tif"))
            if f.ndim == 3:
                f = f[:, :, 0]

            # Add noise
            fn, _ = imnoise2(f, "salt & pepper")

            # Linear filtering
            w, _ = gaussiankernel(7, "sampled", 3.0, 1.0)
            w = w / np.sum(w)
            gG = correlate(fn, w, mode="reflect")  # MATLAB 'symmetric'

            # Non linear filtering
            gM = medfilt2d(fn, kernel_size=7)

            # Display
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(fn, cmap="gray", vmin=0, vmax=1)
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(gG, cmap="gray", vmin=0, vmax=1)
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(gM, cmap="gray", vmin=0, vmax=1)
            plt.axis("off")

            plt.tight_layout()
            plt.savefig("Figure349.png")
            print("Saved Figure349.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure352(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure352.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from scipy.ndimage import correlate
            from helpers.libdip.intScaling4e import intScaling4e
            from helpers.libdipum.data_path import dip_data

            # Data
            f = imread(dip_data("blurry-moon.tif"))
            if f.ndim == 3:
                f = f[:, :, 0]
            f = img_as_float(f)

            # Convolution kernels
            w4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
            w8 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=float)

            # Spatial filtering (MATLAB imfilter default: zero padding)
            lap4 = correlate(f, w4, mode="constant", cval=0.0)
            lap4s = intScaling4e(lap4)

            lap8 = correlate(f, w8, mode="constant", cval=0.0)

            # Enhanced images
            g4 = f - lap4
            g8 = f - lap8

            # Display figure 1
            fig1 = plt.figure(figsize=(8, 8))
            plt.subplot(2, 2, 1)
            plt.imshow(f, cmap="gray")
            plt.axis("off")

            plt.subplot(2, 2, 2)
            plt.imshow(np.abs(lap4), cmap="gray")
            plt.axis("off")

            plt.subplot(2, 2, 3)
            plt.imshow(g4, cmap="gray")
            plt.axis("off")

            plt.subplot(2, 2, 4)
            plt.imshow(g8, cmap="gray")
            plt.axis("off")

            plt.tight_layout()
            fig1.savefig("Figure352.png")
            print("Saved Figure352.png")

            # Display figure 2
            fig2 = plt.figure(figsize=(6, 6))
            plt.imshow(lap4s, cmap="gray")
            plt.axis("off")
            plt.tight_layout()
            fig2.savefig("Figure353.png")
            print("Saved Figure353.png")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure355(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure355.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.libdip.intScaling4e import intScaling4e
            from helpers.libdipum.unsharp import unsharp
            from helpers.libdipum.data_path import dip_data

            # Parameters
            k = [1, 2, 3]
            N = 31
            Sigma = 5.0

            # Data
            f = imread(dip_data("girl-blurred.tif"))
            if f.ndim == 3:
                f = f[:, :, 0]

            # Unsharp masking (k = 1)
            g, gb, gmask_raw = unsharp(f, k[0], N, Sigma)
            gmask = intScaling4e(gmask_raw)

            # Highboost filtering
            ghb2, _, _ = unsharp(f, k[1], N, Sigma)
            ghb3, _, _ = unsharp(f, k[2], N, Sigma)

            # Display
            plt.figure(figsize=(12, 8))

            plt.subplot(2, 3, 1)
            plt.imshow(f, cmap="gray")
            plt.title("Original")
            plt.axis("off")

            plt.subplot(2, 3, 2)
            plt.imshow(gb, cmap="gray", vmin=0, vmax=1)
            plt.title("blurred")
            plt.axis("off")

            plt.subplot(2, 3, 3)
            plt.imshow(gmask, cmap="gray")
            plt.title("Mask")
            plt.axis("off")

            plt.subplot(2, 3, 4)
            plt.imshow(g, cmap="gray", vmin=0, vmax=1)
            plt.title(f"unsharp, k = {k[0]}")
            plt.axis("off")

            plt.subplot(2, 3, 5)
            plt.imshow(ghb2, cmap="gray", vmin=0, vmax=1)
            plt.title(f"unsharp, k = {k[1]}")
            plt.axis("off")

            plt.subplot(2, 3, 6)
            plt.imshow(ghb3, cmap="gray", vmin=0, vmax=1)
            plt.title(f"unsharp, k = {k[2]}")
            plt.axis("off")

            plt.tight_layout()
            plt.savefig("Figure355.png")
            print("Saved Figure355.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure357(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure357.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from scipy.ndimage import correlate
            from helpers.libdipum.data_path import dip_data

            # Sobel kernels
            wh = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)
            wv = wh.T

            # Data
            f = imread(dip_data("contact-lens.tif"))
            if f.ndim == 3:
                f = f[:, :, 0]

            # Filtering (MATLAB 'symmetric' -> scipy 'reflect')
            gx = np.abs(correlate(f.astype(float), wv, mode="reflect"))
            gy = np.abs(correlate(f.astype(float), wh, mode="reflect"))
            g = gx + gy

            # Display
            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.imshow(f, cmap="gray")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(g, cmap="gray")
            plt.axis("off")

            plt.tight_layout()
            plt.savefig("Figure357.png")
            print("Saved Figure357.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure359(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure359.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from helpers.libdipum.zoneplate import zoneplate

            # Data
            f = zoneplate(8.2, 0.0275, 0)
            print(f.shape)

            # Display
            plt.figure(figsize=(6, 6))
            plt.imshow(f, cmap="gray", vmin=0, vmax=1)
            plt.axis("off")
            plt.tight_layout()

            # Print to file
            plt.savefig("Figure359.png")
            print("Saved Figure359.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure360(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure360.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import os
            import sys
            import numpy as np
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            # Allow imports from project root (General, DIP4eFigures, libDIPUM)
            ROOT = str(_Path(__file__).resolve().parents[2])
            if ROOT not in sys.path:
                sys.path.append(ROOT)

            from helpers.libgeneral.fir1 import fir1
            from helpers.libgeneral.ftrans2 import ftrans2

            # Lowpass filtering
            lp, _ = fir1(128, 0.1)

            # 2-D separable lowpass filter
            lp2s = np.outer(lp, lp)

            # 2-D Circularly symmetric lowpass filter
            lp2c = ftrans2(lp)

            # Display
            fig = plt.figure(figsize=(12, 5))

            ax1 = fig.add_subplot(1, 2, 1)
            ax1.plot(lp, color="k")
            ax1.set_box_aspect(1)  # MATLAB-like axis square (subplot box square)
            ax1.margins(x=0)

            ax2 = fig.add_subplot(1, 2, 2, projection="3d")
            Z = lp2c[::2, ::2]
            X, Y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
            ax2.plot_wireframe(X, Y, Z, color="k", linewidth=0.5)
            ax2.set_axis_off()

            plt.tight_layout()
            plt.savefig("Figure360.png")
            print("Saved Figure360.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure361(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure361.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import os
            import sys
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy.ndimage import correlate

            # Allow imports from project root (General, DIP4eFigures, libDIPUM)
            ROOT = str(_Path(__file__).resolve().parents[2])
            if ROOT not in sys.path:
                sys.path.append(ROOT)

            from helpers.libdipum.zoneplate import zoneplate
            from helpers.libgeneral.fir1 import fir1
            from helpers.libgeneral.ftrans2 import ftrans2

            # Data
            f = zoneplate(8.2, 0.0275, 0)  # ~597 x 597

            # Gaussian prefilter
            w = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float) / 16.0
            f = correlate(f, w, mode="nearest")  # MATLAB 'replicate'

            # Lowpass filtering
            lp, _ = fir1(128, 0.1)

            # 2-D separable lowpass filter
            lp2s = np.outer(lp, lp)

            # 2-D circularly symmetric lowpass filter
            lp2c = ftrans2(lp)

            # Filter image with both filters (MATLAB 'symmetric')
            glps = correlate(f, lp2s, mode="reflect")
            glpc = correlate(f, lp2c, mode="reflect")

            # Display
            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.imshow(glps, cmap="gray", vmin=0, vmax=1)
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(glpc, cmap="gray", vmin=0, vmax=1)
            plt.axis("off")

            plt.tight_layout()
            plt.savefig("Figure361.png")
            print("Saved Figure361.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure362(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure362.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import os
            import sys
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy.ndimage import correlate

            # Allow imports from project root (General, DIP4eFigures, libDIPUM)
            ROOT = str(_Path(__file__).resolve().parents[2])
            if ROOT not in sys.path:
                sys.path.append(ROOT)

            from helpers.libdipum.zoneplate import zoneplate
            from helpers.libgeneral.fir1 import fir1
            from helpers.libgeneral.ftrans2 import ftrans2
            from helpers.libdip.intScaling4e import intScaling4e

            # Data
            f = zoneplate(8.2, 0.0275, 0)  # ~597 x 597

            # Lowpass filtering
            lp, _ = fir1(128, 0.1)

            # 2-D separable lowpass filter
            lp2s = np.outer(lp, lp)

            # 2-D circularly symmetric lowpass filter
            lp2c = ftrans2(lp)

            # Apply filters (MATLAB 'symmetric')
            glps = correlate(f, lp2s, mode="reflect")
            glpc = correlate(f, lp2c, mode="reflect")

            # Highpass from lowpass
            ghp = f - glpc

            # Alternative highpass from impulse - lowpass
            M = lp.size
            center = int(np.ceil(M / 2.0)) - 1  # zero-based index
            _delta = np.zeros(M, dtype=float)
            _delta[center] = 1.0
            hp = _delta - lp

            hp2c = ftrans2(hp)
            ghpc = correlate(f, hp2c, mode="reflect")

            # Bandreject from lowpass/highpass
            lp1, _ = fir1(128, 0.06)
            lp2, _ = fir1(128, 0.12)
            hp2 = _delta - lp2

            hbr = lp1 + hp2
            hbrc = ftrans2(hbr)
            gbr = correlate(f, hbrc, mode="reflect")

            # Bandpass from bandreject
            hbp = _delta - hbr
            hbpc = ftrans2(hbp)
            gbp = correlate(f, hbpc, mode="reflect")

            # Display
            plt.figure(figsize=(12, 8))

            plt.subplot(2, 3, 1)
            plt.imshow(glpc, cmap="gray", vmin=0, vmax=1)
            plt.axis("off")

            plt.subplot(2, 3, 2)
            plt.imshow(ghpc, cmap="gray", vmin=0, vmax=1)
            plt.axis("off")

            plt.subplot(2, 3, 3)
            plt.imshow(intScaling4e(ghpc), cmap="gray")
            plt.axis("off")

            plt.subplot(2, 3, 4)
            plt.imshow(gbr, cmap="gray", vmin=0, vmax=1)
            plt.axis("off")

            plt.subplot(2, 3, 5)
            plt.imshow(gbp, cmap="gray", vmin=0, vmax=1)
            plt.axis("off")

            plt.subplot(2, 3, 6)
            plt.imshow(intScaling4e(gbp), cmap="gray")
            plt.axis("off")

            plt.tight_layout()
            plt.savefig("Figure362.png")
            print("Saved Figure362.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure363(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure363.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from scipy.ndimage import correlate
            from helpers.libdipum.data_path import dip_data

            print("Running Figure363 (Skeleton Bone Scan Enhancement)...")

            # Image loading
            img_path = dip_data("bonescan.tif")
            f = imread(img_path)
            if f.ndim == 3:
                f = f[:, :, 0]

            f = img_as_float(f)

            # Laplacian
            # w = [-1 -1 -1;-1 8 -1;-1 -1 -1];
            w = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

            # gL = imfilter(f, w, 'symmetric');
            gL = correlate(f, w, mode="reflect")

            # Image needs scaling for display (gLs)
            # Replicate intScaling4e behavior: map min..max to 0..1 (or full range)
            gLs = gL.copy()
            gL_min, gL_max = gLs.min(), gLs.max()
            if gL_max > gL_min:
                gLs = (gLs - gL_min) / (gL_max - gL_min)

            # Sharpen
            # gSharp = f + gL;
            gSharp = f + gL
            # Clip to verify valid range?
            # Usually we don't clip intermediate results unless display.
            # But later calculation uses gSharp.

            # Sobel Gradient
            # gx = [-1 -2 -1;0 0 0;1 2 1];
            # gy = [-1 0 1;-2 0 2;-1 0 1];
            gx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            gy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

            # G = abs(imfilter(f, gx, 'symmetric')) + abs(imfilter(f, gy, 'symmetric'));
            Gx = correlate(f, gx, mode="reflect")
            Gy = correlate(f, gy, mode="reflect")
            G = np.abs(Gx) + np.abs(Gy)

            # Smooth gradient
            # waverage = ones(5)/25
            waverage = np.ones((5, 5)) / 25.0
            Gaverage = correlate(G, waverage, mode="reflect")

            # Product
            # LG = gSharp.*Gaverage;
            LG = gSharp * Gaverage

            # Edge Enhanced
            # gEE = f + LG;
            gEE = f + LG

            # Display
            # Two figures in MATLAB script. We can combine into one large figure or save as two files.
            # The script saves Figure363.png and Figure363Bis.png

            # Figure 1
            fig1, axes1 = plt.subplots(2, 2, figsize=(10, 10))
            axes1 = axes1.flatten()

            axes1[0].imshow(f, cmap="gray")
            axes1[0].set_title("Original")
            axes1[0].axis("off")

            axes1[1].imshow(gLs, cmap="gray")
            axes1[1].set_title("Scaled Laplacian")
            axes1[1].axis("off")

            axes1[2].imshow(gSharp, cmap="gray")  # Might need clipping for display
            axes1[2].set_title("Image + Laplacian")
            axes1[2].axis("off")

            axes1[3].imshow(G, cmap="gray")
            axes1[3].set_title("Sobel Gradient")
            axes1[3].axis("off")

            plt.tight_layout()
            plt.savefig("Figure363.png")
            print("Saved Figure363.png")

            # Figure 2
            fig2, axes2 = plt.subplots(2, 2, figsize=(10, 10))
            axes2 = axes2.flatten()

            axes2[0].imshow(Gaverage, cmap="gray")
            axes2[0].set_title("Smoothed Sobel Gradient")
            axes2[0].axis("off")

            axes2[1].imshow(LG, cmap="gray")  # Might need clipping
            axes2[1].set_title("ProductMask * Sharpened")
            axes2[1].axis("off")

            axes2[2].imshow(gEE, cmap="gray")  # Might need clipping
            axes2[2].set_title("Result (f + LG)")
            axes2[2].axis("off")

            # 4th subplot was gamma commented out
            axes2[3].axis("off")

            plt.tight_layout()
            plt.savefig("Figure363Bis.png")
            print("Saved Figure363Bis.png")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure366(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure366.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from helpers.libdipum.triangmf import triangmf
            from helpers.libdipum.trapezmf import trapezmf
            from helpers.libdipum.sigmamf import sigmamf
            from helpers.libdipum.smf import smf
            from helpers.libdipum.bellmf import bellmf
            from helpers.libdipum.truncgaussmf import truncgaussmf

            print("Running Figure366 (Fuzzy Membership Functions)...")

            z = np.linspace(0, 255, 500)

            # 1. Triangle
            # u.triangle = triangmf (z, 20, 70, 200);
            u_triangle = triangmf(z, 20, 70, 200)

            # 2. Trapezoid
            # u.trapez = trapezmf (z, 20, 50, 200, 220);
            u_trapez = trapezmf(z, 20, 50, 200, 220)

            # 3. Sigma
            # u.sigma = sigmamf (z, 30, 70);
            u_sigma = sigmamf(z, 30, 70)

            # 4. S-shape
            # u.s = smf (z, 30, 226);
            u_s = smf(z, 30, 226)

            # 5. Bell
            # u.bell = bellmf (z, 50, 100);
            u_bell = bellmf(z, 50, 100)

            # 6. Truncated Gaussian
            # u.gauss = truncgaussmf (z, 50, 100, 20);
            u_gauss = truncgaussmf(z, 50, 100, 20)

            # Display
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes = axes.flatten()

            plots = [
                (u_triangle, "Triangle"),
                (u_trapez, "Trapezoid"),
                (u_sigma, "Sigma"),
                (u_s, "S-shape"),
                (u_bell, "Bell"),
                (u_gauss, "Truncated Gaussian"),
            ]

            for i, (data, title) in enumerate(plots):
                ax = axes[i]
                ax.plot(z, data)
                ax.set_title(title)
                ax.set_xlim([0, 255])
                ax.set_ylim([0, 1.05])
                # ax.axis('tight') # MATLAB axis tight

            plt.tight_layout()
            plt.savefig("Figure366.png")
            print("Saved Figure366.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure371(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure371.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt

            from helpers.libdip.fuzzymf3e import fuzzymf3e
            from helpers.libdip.fuzzyeval3e import fuzzyeval3e
            from helpers.libdip.fuzzyimp3e import fuzzyimp3e
            from helpers.libdip.fuzzyad3e import fuzzyad3e

            ugreen, _ = fuzzymf3e('triang', 200, 0, 1, [0.2, 0.2, 0.2])
            uyellow, _ = fuzzymf3e('triang', 200, 0, 1, [0.5, 0.25, 0.25])
            ured, _ = fuzzymf3e('triang', 200, 0, 1, [0.78, 0.22, 0.22])

            uverd, _ = fuzzymf3e('trapez', 101, 0, 100, [0, 10, 0, 20])
            uhalf, _ = fuzzymf3e('trapez', 101, 0, 100, [38, 52, 19, 18])
            umat, _ = fuzzymf3e('trapez', 101, 0, 100, [80, 100, 25, 0])

            inmf = np.vstack([ugreen, uyellow, ured])
            outmf = np.vstack([uverd, uhalf, umat])
            R = np.array([[1], [2], [3]], dtype=int)

            z = 0.705
            rule_strength, _ = fuzzyeval3e(R, inmf, [z], 1)
            q = fuzzyimp3e(rule_strength, outmf)
            defuzz = fuzzyad3e(q)
            print(f'defuzz = {defuzz}')

            m = np.maximum(np.maximum(q[0, :], q[1, :]), q[2, :])

            plt.figure(figsize=(8, 10))
            plt.subplot(3, 1, 1)
            plt.plot(ugreen, 'g')
            plt.title('green : g, yellow : y, red : r')
            plt.xlabel('Color (wavelength)')
            plt.ylabel('Membership')
            plt.plot(uyellow, 'y')
            plt.plot(ured, 'r')

            plt.subplot(3, 1, 2)
            plt.plot(q[0, :], 'r')
            plt.title('Q1 : r, Q2 : g, Q3 : b')
            plt.xlabel('Maturity')
            plt.ylabel('Membership')
            plt.plot(q[1, :], 'g')
            plt.plot(q[2, :], 'b')

            plt.subplot(3, 1, 3)
            plt.plot(m)
            plt.xlabel('Maturity')
            plt.ylabel('Membership')

            plt.tight_layout()
            plt.savefig('Figure371.png', dpi=150)
            plt.show()

        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure37(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure37.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.libdipum.intensityTransformations import intensityTransformations
            from helpers.libdipum.data_path import dip_data

            # Data
            img_path = dip_data("retina.tif")
            f = imread(img_path)

            # Simulate monitor gamma
            gmonitor = intensityTransformations(f, "gamma", 2.5)

            # Gamma correction with 1/2.5
            ggammacorrected = intensityTransformations(f, "gamma", 0.4)

            # Put corrected image through monitor
            gmonitorcorrected = intensityTransformations(ggammacorrected, "gamma", 2.5)

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            axes[0, 0].imshow(f, cmap="gray")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(gmonitor, cmap="gray")
            axes[0, 1].axis("off")

            axes[1, 0].imshow(ggammacorrected, cmap="gray")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(gmonitorcorrected, cmap="gray")
            axes[1, 1].axis("off")

            plt.tight_layout()
            plt.savefig("Figure37.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure374(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure374.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage import exposure
            from skimage.io import imread
            from helpers.libdipum.sigmamf import sigmamf
            from helpers.libdipum.triangmf import triangmf
            from helpers.libdipum.data_path import dip_data

            print("Running Figure374 (Fuzzy Contrast Enhancement)...")

            # Image loading (Exact path)
            img_name = dip_data("einstein-low-contrast.tif")
            f = imread(img_name)
            if f.ndim == 3:
                f = f[:, :, 0]

            if f.ndim == 3:
                f = f[:, :, 0]
            f = f.astype(float)

            # 1. Histogram Equalization
            f_uint8 = f.astype(np.uint8)
            g1_float = exposure.equalize_hist(f_uint8)
            g1 = (g1_float * 255).astype(np.uint8)

            # 2. Fuzzy Enhancement
            # Note: sigmamf expects values. If z is 0-255.
            udark_val = 1 - sigmamf(f, 74, 127)
            ugray_val = triangmf(f, 74, 127, 180)
            ubright_val = sigmamf(f, 127, 180)

            vd = 0.0
            vg = 127.0
            vb = 255.0

            numerator = udark_val * vd + ugray_val * vg + ubright_val * vb
            denominator = udark_val + ugray_val + ubright_val

            denominator[denominator == 0] = 1e-6

            g2 = numerator / denominator
            g2 = np.clip(g2, 0, 255).astype(np.uint8)

            # Setup Plot 1 (Images)
            fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
            axes1[0].imshow(f, cmap="gray", vmin=0, vmax=255)
            axes1[0].set_title("Original")
            axes1[0].axis("off")

            axes1[1].imshow(g1, cmap="gray", vmin=0, vmax=255)
            axes1[1].set_title("Histogram Equalization")
            axes1[1].axis("off")

            axes1[2].imshow(g2, cmap="gray", vmin=0, vmax=255)
            axes1[2].set_title("Fuzzy Logic Enhancement")
            axes1[2].axis("off")

            plt.savefig("Figure374.png")

            # Setup Plot 2 (Histograms and MFs)
            fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))

            # Hist Original
            axes2[0, 0].hist(
                f.ravel(), bins=256, range=(0, 255), color="k", histtype="stepfilled"
            )
            axes2[0, 0].set_title("Hist: Original")
            axes2[0, 0].set_xlim([0, 255])

            # Hist Equalized
            axes2[0, 1].hist(
                g1.ravel(), bins=256, range=(0, 255), color="k", histtype="stepfilled"
            )
            axes2[0, 1].set_title("Hist: Equalized")
            axes2[0, 1].set_xlim([0, 255])

            # Hist Original with MFs
            ax_hist_mf = axes2[1, 0]
            counts, bins = np.histogram(f.ravel(), bins=256, range=(0, 255))
            counts_norm = counts / (counts.max() + 1e-6)
            ax_hist_mf.bar(bins[:-1], counts_norm, width=1, color="gray", alpha=0.5)

            # Plot MFs
            z = np.linspace(0, 255, 500)
            mf_dark = 1 - sigmamf(z, 74, 127)
            mf_gray = triangmf(z, 74, 127, 180)
            mf_bright = sigmamf(z, 127, 180)

            ax_hist_mf.plot(z, mf_dark, "k-", linewidth=2, label="Dark")
            ax_hist_mf.plot(z, mf_gray, "k--", linewidth=2, label="Gray")
            ax_hist_mf.plot(z, mf_bright, "k:", linewidth=2, label="Bright")
            ax_hist_mf.set_title("Hist & MFs")
            ax_hist_mf.set_xlim([0, 255])
            ax_hist_mf.legend(loc="upper right")

            # Hist Fuzzy Enhanced
            axes2[1, 1].hist(
                g2.ravel(), bins=256, range=(0, 255), color="k", histtype="stepfilled"
            )
            axes2[1, 1].set_title("Hist: Fuzzy Enhanced")
            axes2[1, 1].set_xlim([0, 255])

            plt.tight_layout()
            plt.savefig("Figure374Bis.png")
            print("Saved Figure374.png and Figure374Bis.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure379(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter03 script `Figure379.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from scipy.ndimage import convolve
            from helpers.libdipum.bellmf import bellmf
            from helpers.libdipum.onemf import onemf
            from helpers.libdipum.triangmf import triangmf
            from helpers.libdipum.fuzzysysfcn import fuzzysysfcn
            from helpers.libdipum.approxfcn import approxfcn
            from helpers.libdipum.data_path import dip_data

            # MATLAB-like tofloat/revertClass behavior for this script
            f_in = imread(dip_data("headCT.tif"))
            if f_in.ndim == 3:
                f_in = f_in[:, :, 0]
            orig_dtype = f_in.dtype
            f = img_as_float(f_in)

            def revertClass(x: Any):
                """revertClass."""
                if np.issubdtype(orig_dtype, np.integer):
                    info = np.iinfo(orig_dtype)
                    y = np.clip(x, 0.0, 1.0)
                    y = np.round(y * info.max).astype(orig_dtype)
                    return y
                return x.astype(orig_dtype)

            # The fuzzy system has four inputs: differences with neighbors.
            z1 = convolve(f, np.array([[0, -1, 1]], dtype=float), mode="nearest")
            z2 = convolve(f, np.array([[0], [-1], [1]], dtype=float), mode="nearest")
            z3 = convolve(f, np.array([[1], [-1], [0]], dtype=float), mode="nearest")
            z4 = convolve(f, np.array([[1, -1, 0]], dtype=float), mode="nearest")

            # Input membership functions.
            zero = lambda z: bellmf(z, -0.3, 0)  # noqa: E731
            not_used = lambda z: onemf(z)  # noqa: E731

            # Output membership functions.
            black = lambda z: triangmf(z, 0, 0, 0.75)  # noqa: E731
            white = lambda z: triangmf(z, 0.25, 1, 1)  # noqa: E731

            # 4 rules x 4 inputs
            inmf = [
                [zero, not_used, zero, not_used],
                [not_used, not_used, zero, zero],
                [not_used, zero, not_used, zero],
                [zero, zero, not_used, not_used],
            ]

            # Extra output MF gives automatic else-rule behavior
            outmf = [white, white, white, white, black]

            # Output range
            vrange = [0, 1]

            # Build fuzzy system and LUT approximation
            F = fuzzysysfcn(inmf, outmf, vrange)
            G = approxfcn(
                F, np.array([[-1, 1], [-1, 1], [-1, 1], [-1, 1]], dtype=float)
            )
            z1c = np.clip(z1, -1.0, 1.0)
            z2c = np.clip(z2, -1.0, 1.0)
            z3c = np.clip(z3, -1.0, 1.0)
            z4c = np.clip(z4, -1.0, 1.0)
            gf = G(z1c, z2c, z3c, z4c)

            # Convert output back to class of input image
            g = revertClass(gf)

            # Display 1
            fig1 = plt.figure(figsize=(10, 8))
            plt.subplot(2, 2, 1)
            plt.imshow(z1, cmap="gray")
            plt.title("d6")
            plt.axis("off")

            plt.subplot(2, 2, 2)
            plt.imshow(z2, cmap="gray")
            plt.title("d2")
            plt.axis("off")

            plt.subplot(2, 2, 3)
            plt.imshow(z3, cmap="gray")
            plt.title("d8")
            plt.axis("off")

            plt.subplot(2, 2, 4)
            plt.imshow(z4, cmap="gray")
            plt.title("d4")
            plt.axis("off")

            plt.tight_layout()
            fig1.savefig("Figure379.png")
            print("Saved Figure379.png")

            # Display 2
            fig2 = plt.figure(figsize=(10, 8))
            ax1 = plt.subplot(2, 2, 1)
            plt.imshow(f, cmap="gray", vmin=0, vmax=1)
            plt.title("Original")
            plt.axis("off")

            ax2 = plt.subplot(2, 2, 2)
            if np.issubdtype(g.dtype, np.integer):
                plt.imshow(g, cmap="gray", vmin=0, vmax=np.iinfo(g.dtype).max)
            else:
                plt.imshow(g, cmap="gray", vmin=0, vmax=1)
            plt.title("Fuzzy edge enhancement")
            plt.axis("off")

            ax3 = plt.subplot(2, 2, 3)
            gm = gf - np.min(gf)
            if np.max(gm) > 0:
                gs = np.uint8(255.0 * (gm / np.max(gm)))
            else:
                gs = np.zeros_like(gm, dtype=np.uint8)
            plt.imshow(gs, cmap="gray")
            plt.title("After scaling")
            plt.axis("off")

            # MATLAB linkaxes equivalent for image limits consistency
            for ax in [ax1, ax2, ax3]:
                ax.set_xlim(0, f.shape[1] - 1)
                ax.set_ylim(f.shape[0] - 1, 0)

            plt.tight_layout()
            fig2.savefig("Figure379Bis.png")
            print("Saved Figure379Bis.png")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

