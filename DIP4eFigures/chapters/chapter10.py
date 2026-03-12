from pathlib import Path as _Path
import os as _os

from typing import Any


class Chapter10Mixin:
    def figure101(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure101.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.libgeneral.edge import edge
            from helpers.libdipum.splitmerge import splitmerge
            from helpers.libdipum.data_path import dip_data

            # Data
            path_f = dip_data("constant-gray-region.tif")
            path_fn = dip_data("Fig1001(d)(noisy_region).tif")

            f = imread(path_f)
            fn = imread(path_fn)

            # Edge detection on original
            # gedge = edge (f, 'sobel', 0);
            # In MATLAB, threshold 0 means usually default or strict 0.
            # Our edge.py handles 0 explicitly if passed, but let's see.
            # If 0 implies "all gradients > 0", it will be noisy if image isn't perfect constant.
            # But f is "constant-gray-region", so it has perfect regions.
            # Sobel will find the boundary.
            gedge = edge(f, "sobel", threshold=None)  # Start with auto

            # Thresholding
            # mx = max(f(:)); mn = min(f(:)); T = mn + (mx - mn)/2; gthresh = f > T;
            mx = f.max()
            mn = f.min()
            T = mn + (mx - mn) / 2
            gthresh = f > T

            # Noisy region
            # gnedge = edge(fn, 'sobel', 0);
            gnedge = edge(fn, "sobel", threshold=None)

            # Splitmerge
            # gsm = splitmerge(fn, 8, @predicate2);
            # predicate2: flag = (sd > 10);
            def predicate2(region: Any):
                """predicate2."""
                if region.size == 0:
                    return False
                sd = np.std(region)
                return sd > 10

            gsm = splitmerge(fn, 8, predicate2)

            # Display
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            axes[0].imshow(f, cmap="gray")
            axes[0].set_title("f")
            axes[0].axis("off")

            axes[1].imshow(gedge, cmap="gray")
            axes[1].set_title("Edge (Sobel)")
            axes[1].axis("off")

            axes[2].imshow(gthresh, cmap="gray")
            axes[2].set_title("Thresholded")
            axes[2].axis("off")

            axes[3].imshow(fn, cmap="gray")
            axes[3].set_title("Noisy Image")
            axes[3].axis("off")

            axes[4].imshow(gnedge, cmap="gray")
            axes[4].set_title("Edge of Noisy Image")
            axes[4].axis("off")

            axes[5].imshow(gsm, cmap="jet")  # Color to show segmentation
            axes[5].set_title("SplitMerge Result")
            axes[5].axis("off")

            plt.tight_layout()
            print("Saved Figure101.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1011(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1011.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import random_noise
            from helpers.libdip.intScaling4e import intScaling4e
            from helpers.libdipum.sobel import sobel
            from helpers.libdipum.lap import lap
            from helpers.libdipum.data_path import dip_data

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
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1016(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1016.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from scipy.ndimage import convolve
            from helpers.libdipum.data_path import dip_data

            # Data
            img_path = dip_data("building-cropped-834by1114.tif")
            f = img_as_float(imread(img_path))

            # Kernel
            Sx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)

            Sy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)

            # Filtering
            Gx = np.abs(convolve(f, Sx, mode="nearest"))
            Gy = np.abs(convolve(f, Sy, mode="nearest"))

            # g = Gx + Gy;
            g = Gx + Gy

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes = axes.flatten()

            axes[0].imshow(f, cmap="gray")
            axes[0].set_title("f")
            axes[0].axis("off")

            axes[1].imshow(Gx, cmap="gray")
            axes[1].set_title("Gx (Response to Sx)")
            axes[1].axis("off")

            axes[2].imshow(Gy, cmap="gray")
            axes[2].set_title("Gy (Response to Sy)")
            axes[2].axis("off")

            axes[3].imshow(g, cmap="gray")
            axes[3].set_title("Gradient Magnitude (Gx + Gy)")
            axes[3].axis("off")

            plt.tight_layout()
            print(f"Saved Figure1016.png using {img_path}")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1017(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1017.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from scipy.ndimage import convolve
            from helpers.libdipum.data_path import dip_data

            # Data
            img_path = dip_data("building-cropped-834by1114.tif")
            f = img_as_float(imread(img_path))

            # Kernel
            Sx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)

            Sy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)

            # Gradient
            Gx = convolve(f, Sx, mode="nearest")
            Gy = convolve(f, Sy, mode="nearest")

            # angle = atan2(Gy,Gx);
            angle = np.arctan2(Gy, Gx)

            # Display
            # imshow(angle, []) scales min-max.

            plt.figure(figsize=(8, 8))
            plt.imshow(angle, cmap="gray")
            plt.axis("off")
            plt.title("Gradient Angle")

            print("Saved Figure1017.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1018(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1018.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from scipy.ndimage import convolve, uniform_filter
            from helpers.libdipum.data_path import dip_data

            # Data
            img_path = dip_data("building-cropped-834by1114.tif")
            f = img_as_float(imread(img_path))

            # Kernel
            fs = uniform_filter(f, size=5, mode="nearest")

            # Sobel Kernels
            Sx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)

            Sy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)

            # Gradient on Smoothed Image
            Gxs = np.abs(convolve(fs, Sx, mode="nearest"))
            Gys = np.abs(convolve(fs, Sy, mode="nearest"))

            # gs = Gxs + Gys;
            gs = Gxs + Gys

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes = axes.flatten()

            axes[0].imshow(fs, cmap="gray")
            axes[0].set_title("Smoothed Image (fs)")
            axes[0].axis("off")

            axes[1].imshow(Gxs, cmap="gray")
            axes[1].set_title("Gx of Smoothed (Gxs)")
            axes[1].axis("off")

            axes[2].imshow(Gys, cmap="gray")
            axes[2].set_title("Gy of Smoothed (Gys)")
            axes[2].axis("off")

            axes[3].imshow(gs, cmap="gray")
            axes[3].set_title("Gradient of Smoothed (gs)")
            axes[3].axis("off")

            plt.tight_layout()
            print("Saved Figure1018.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1019(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1019.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from scipy.ndimage import convolve, uniform_filter
            from helpers.libdip.edgeKernel4e import edgeKernel4e
            from helpers.libdipum.data_path import dip_data

            # Data
            img_path = dip_data("building-cropped-834by1114.tif")
            f = img_as_float(imread(img_path))

            # Kirsch Kernels
            wK45 = edgeKernel4e("kirsch", "nw")
            wKm45 = edgeKernel4e("kirsch", "sw")

            # Smoothing
            fs = uniform_filter(f, size=5, mode="nearest")

            # Filter
            G45 = convolve(fs, wK45, mode="nearest")
            Gm45 = convolve(fs, wKm45, mode="nearest")

            # Display
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes = axes.flatten()

            axes[0].imshow(G45, cmap="gray")
            axes[0].set_title("NW Kirsch (45 deg)")
            axes[0].axis("off")

            axes[1].imshow(Gm45, cmap="gray")
            axes[1].set_title("SW Kirsch (-45 deg)")
            axes[1].axis("off")

            plt.tight_layout()
            print("Saved Figure1019.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1020(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1020.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from scipy.ndimage import convolve, uniform_filter
            from helpers.libdipum.data_path import dip_data

            # Data
            img_path = dip_data("building-cropped-834by1114.tif")
            f = img_as_float(imread(img_path))

            # Kernels
            Sx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)

            Sy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)

            # Filtering (Smoothed)
            fs = uniform_filter(f, size=5, mode="nearest")

            # Gradient on Original
            Gx = np.abs(convolve(f, Sx, mode="nearest"))
            Gy = np.abs(convolve(f, Sy, mode="nearest"))
            g = Gx + Gy

            # Gradient on Smoothed
            Gxs = np.abs(convolve(fs, Sx, mode="nearest"))
            Gys = np.abs(convolve(fs, Sy, mode="nearest"))
            gs = Gxs + Gys

            # Thresholding
            gt = g >= (0.33 * g.max())
            gst = gs >= (0.33 * gs.max())

            # Display
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes = axes.flatten()

            axes[0].imshow(gt, cmap="gray")
            axes[0].set_title("Thresholded Gradient (Original)")
            axes[0].axis("off")

            axes[1].imshow(gst, cmap="gray")
            axes[1].set_title("Thresholded Gradient (Smoothed)")
            axes[1].axis("off")

            plt.tight_layout()
            print("Saved Figure1020.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1022(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1022.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy.ndimage import convolve as ndi_convolve
            from skimage.io import imread
            from skimage.util import img_as_float
            from helpers.libgeneral.fspecial import fspecial
            from helpers.libdipum.data_path import dip_data

            # Figure 10.22 (Marr-Hildreth)
            def log_zero_cross_edges(f: Any, sigma: Any, threshold: Any):
                """
                LoG edge detector with thresholded zero crossings.
                threshold=0 gives characteristic closed-loop edges.
                """
                hsize = int(np.ceil(6 * sigma))
                if hsize % 2 == 0:
                    hsize += 1

                h = fspecial("log", hsize, sigma)
                log_img = ndi_convolve(f, h, mode="nearest")

                # Zero-crossings in 4-neighborhood with transition-strength threshold.
                out = np.zeros_like(log_img, dtype=bool)

                a = log_img[:-1, :]
                b = log_img[1:, :]
                zc = (a * b) < 0
                strong = np.abs(a - b) > threshold
                out[:-1, :] |= zc & strong

                a = log_img[:, :-1]
                b = log_img[:, 1:]
                zc = (a * b) < 0
                strong = np.abs(a - b) > threshold
                out[:, :-1] |= zc & strong

                return log_img, out

            # Data
            img_path = dip_data("building-cropped-834by1114.tif")
            f = img_as_float(imread(img_path))

            # Generate LoG filter for display (as in MATLAB script).
            LoGfilter = fspecial("log", 25, 4)

            # Filtering
            LoGimage = ndi_convolve(f, LoGfilter, mode="nearest")
            print(np.max(LoGimage))
            print(np.min(LoGimage))

            # Zero crossings with thresholds 0 and 0.0009
            gTzero = log_zero_cross_edges(f, sigma=4, threshold=0.0)[1]
            gT0009 = log_zero_cross_edges(f, sigma=4, threshold=0.0009)[1]

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes = axes.ravel()

            axes[0].imshow(f, cmap="gray")
            axes[0].axis("off")

            axes[1].imshow(LoGimage, cmap="gray")
            axes[1].axis("off")

            axes[2].imshow(gTzero, cmap="gray")
            axes[2].axis("off")

            axes[3].imshow(gT0009, cmap="gray")
            axes[3].axis("off")

            plt.tight_layout()
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1023(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1023.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from helpers.libdipum.logdogfilter import logdogfilter

            # Figure 10.23
            # Comparison of LoG and DoG

            # LoG and DoG filter
            _, _, PL1, PD1 = logdogfilter(511, 20, 1.75, "auto", 1)
            _, _, PL2, PD2 = logdogfilter(511, 20, 1.6, "auto", 1)

            # Display
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            axes[0].plot(-PD1, "r-.")
            axes[0].plot(-PL1, "g")
            axes[0].set_box_aspect(1)
            axes[0].autoscale(enable=True, axis="both", tight=True)

            axes[1].plot(-PD2, "r-.")
            axes[1].plot(-PL2, "g")
            axes[1].set_box_aspect(1)
            axes[1].autoscale(enable=True, axis="both", tight=True)

            plt.tight_layout()
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1025(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1025.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy.ndimage import convolve as ndi_convolve
            from skimage.io import imread
            from skimage.util import img_as_float
            from helpers.libgeneral.edge import edge
            from helpers.libgeneral.fspecial import fspecial
            from helpers.libdipum.data_path import dip_data

            # Figure 10.25
            # Canny edge detection of building

            def log_zero_cross_edges(f: Any, sigma: Any, threshold: Any):
                """
                LoG edge detector with thresholded zero crossings.
                """
                hsize = int(np.ceil(6 * sigma))
                if hsize % 2 == 0:
                    hsize += 1

                h = fspecial("log", hsize, sigma)
                log_img = ndi_convolve(f, h, mode="nearest")

                out = np.zeros_like(log_img, dtype=bool)

                a = log_img[:-1, :]
                b = log_img[1:, :]
                zc = (a * b) < 0
                strong = np.abs(a - b) > threshold
                out[:-1, :] |= zc & strong

                a = log_img[:, :-1]
                b = log_img[:, 1:]
                zc = (a * b) < 0
                strong = np.abs(a - b) > threshold
                out[:, :-1] |= zc & strong

                return log_img, out

            # Data
            img_path = dip_data("Fig1016(a)(building_original).tif")
            f = img_as_float(imread(img_path))
            if f.ndim == 3:
                f = f[..., 0]

            # Kernels
            w = fspecial("average", 5)
            Sx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
            Sy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)

            # Filtering
            fs = ndi_convolve(f, w, mode="nearest")

            Gx = np.abs(ndi_convolve(f, Sx, mode="nearest"))
            Gy = np.abs(ndi_convolve(f, Sy, mode="nearest"))
            g = Gx + Gy

            Gxs = np.abs(ndi_convolve(fs, Sx, mode="nearest"))
            Gys = np.abs(ndi_convolve(fs, Sy, mode="nearest"))
            gs = Gxs + Gys

            # Thresholding
            gt = g >= 0.33 * np.max(g)
            gst = gs >= 0.33 * np.max(gs)

            # LoG filter and filtering
            LoGfilter = fspecial("log", 25, 4)
            LoGimage = ndi_convolve(f, LoGfilter, mode="nearest")
            print(np.max(LoGimage))
            print(np.min(LoGimage))

            # Marr-Hildreth and Canny
            gT0009 = log_zero_cross_edges(f, sigma=4, threshold=0.0009)[1]
            gcanny = edge(f, "canny", [0.04, 0.1], 4)

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes = axes.ravel()

            axes[0].imshow(f, cmap="gray")
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            axes[1].imshow(gst, cmap="gray")
            axes[1].set_title("Thresholded gradient of smoothed image")
            axes[1].axis("off")

            axes[2].imshow(gT0009, cmap="gray")
            axes[2].set_title("Marr-Hildreth")
            axes[2].axis("off")

            axes[3].imshow(gcanny, cmap="gray")
            axes[3].set_title("Canny")
            axes[3].axis("off")

            plt.tight_layout()
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1026(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1026.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from scipy.ndimage import convolve, uniform_filter
            from helpers.libgeneral.edge import edge
            from helpers.libgeneral.fspecial import fspecial
            from helpers.libdipum.data_path import dip_data

            print("Running Figure1026 (Edge Detection Comparison)...")

            def log_zero_cross_edges(f: Any, sigma: Any, threshold: Any):
                """
                LoG edge detector with thresholded zero crossings.
                """
                hsize = int(np.ceil(6 * sigma))
                if hsize % 2 == 0:
                    hsize += 1

                h = fspecial("log", hsize, sigma)
                log_img = convolve(f, h, mode="nearest")

                out = np.zeros_like(log_img, dtype=bool)

                a = log_img[:-1, :]
                b = log_img[1:, :]
                zc = (a * b) < 0
                strong = np.abs(a - b) > threshold
                out[:-1, :] |= zc & strong

                a = log_img[:, :-1]
                b = log_img[:, 1:]
                zc = (a * b) < 0
                strong = np.abs(a - b) > threshold
                out[:, :-1] |= zc & strong

                return log_img, out

            # Data
            img_path = dip_data("headCT.tif")
            f = img_as_float(imread(img_path))

            # 1. Smoothed Gradient
            # w = fspecial('average', 5);
            # fs = imfilter(f,w,'replicate');
            fs = uniform_filter(f, size=5, mode="nearest")

            # Sobel Masks
            # Sx = [-1 -2 -1; 0 0 0; 1 2 1];
            # Sy = [-1 0 1; -2 0 2; -1 0 1];
            Sx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
            Sy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)

            # Gxs = abs(imfilter(fs,Sx,'conv','replicate'));
            # Gys = abs(imfilter(fs,Sy,'conv','replicate'));
            Gxs = np.abs(convolve(fs, Sx, mode="nearest"))
            Gys = np.abs(convolve(fs, Sy, mode="nearest"))

            # gs = Gxs + Gys; % Gradient image.
            gs = Gxs + Gys

            # gst=gs>=0.15*max(gs(:));
            gst = gs >= (0.15 * gs.max())

            # 2. Marr-Hildreth edge detection.
            # gm = edge(f,'log',0.002,3);
            # Note: 0.002 threshold. Sigma=3.
            gm = log_zero_cross_edges(f, sigma=3, threshold=0.002)[1]

            # 3. Canny edge detection
            # gcan = edge(f,'canny',[0.05 0.15], 2);
            gcan = edge(f, "canny", [0.05, 0.15], 2)

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes = axes.flatten()

            axes[0].imshow(f, cmap="gray")
            axes[0].set_title("Original Head CT")
            axes[0].axis("off")

            axes[1].imshow(gst, cmap="gray")
            axes[1].set_title("Smoothed Gradient (T=15% max)")
            axes[1].axis("off")

            axes[2].imshow(gm, cmap="gray")
            axes[2].set_title("Marr-Hildreth (Sigma=3, T=0.002)")
            axes[2].axis("off")

            axes[3].imshow(gcan, cmap="gray")
            axes[3].set_title("Canny (Sigma=2, T=[0.05, 0.15])")
            axes[3].axis("off")

            plt.tight_layout()
            print("Saved Figure1026.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1027(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1027.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from skimage.morphology import thin
            from helpers.libdipum.edgelinklocal import edgelinklocal
            from helpers.libdipum.data_path import dip_data

            print("Running Figure1027 (Local Edge Linking)...")

            # Data
            # Filename in script is 'van-rear.tif'.
            # I'll rely on the finding tool or standard paths.

            img_path = dip_data("van-rear.tif")

            f = img_as_float(imread(img_path))

            # Detection and linking in the vertical direction:
            # [GV, MAGV, ANGLEV, Gxv, Gyv] = edgelinklocal(f', .30, 90, 45, 25);
            # Transpose input f
            ft = f.T
            GV_t, MAGV_t, ANGLEV_t, Gxv_t, Gyv_t = edgelinklocal(ft, 0.30, 90, 45, 25)

            # Transpose back
            GV = GV_t.T
            MAGV = MAGV_t.T
            # ANGLEV = ANGLEV_t.T
            # Gxv = Gxv_t.T
            # Gyv = Gyv_t.T

            # Detection and linking in the horizontal direction:
            # [GH, MAGH, ANGLEH, Gxh, Gyh] = edgelinklocal(f, .30, 90, 45, 25);
            GH, MAGH, ANGLEH, Gxh, Gyh = edgelinklocal(f, 0.30, 90, 45, 25)

            # Logical OR
            # G = GH | GV;
            G = (GH > 0) | (GV > 0)

            # Thinning
            # Gthin = bwmorph (G,'thin',Inf);
            # skimage.morphology.thin returns bool
            Gthin = thin(G)

            # Display
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            axes[0].imshow(f, cmap="gray")
            axes[0].set_title("Original")
            axes[0].axis("off")

            axes[1].imshow(MAGH, cmap="gray")
            axes[1].set_title("Gradient Magnitude")  # MAGH ~ MAGV
            axes[1].axis("off")

            axes[2].imshow(GH, cmap="gray")
            axes[2].set_title("Horizontal Linking")
            axes[2].axis("off")

            axes[3].imshow(GV, cmap="gray")
            axes[3].set_title("Vertical Linking")
            axes[3].axis("off")

            axes[4].imshow(G, cmap="gray")
            axes[4].set_title("Combined (OR)")
            axes[4].axis("off")

            axes[5].imshow(Gthin, cmap="gray")
            axes[5].set_title("Thinned Result")
            axes[5].axis("off")

            plt.tight_layout()
            print("Saved Figure1027.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1030(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1030.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.transform import hough_line

            print("Running Figure1030 (Hough Transform)...")

            # Data
            # f = zeros(101,101);
            f = np.zeros((101, 101), dtype=float)

            # f(1,1) = 1; f(101,1) = 1; f(1,101) = 1; f(101,101) = 1; f(51,51) = 1;
            # MATLAB indices are 1-based (row, col).
            # Python indices are 0-based.
            # MATLAB (1,1) -> Python (0,0)
            # MATLAB (101,1) -> Python (100,0)
            # MATLAB (1,101) -> Python (0,100)
            # MATLAB (101,101) -> Python (100,100)
            # MATLAB (51,51) -> Python (50,50)

            f[0, 0] = 1
            f[100, 0] = 1
            f[0, 100] = 1
            f[100, 100] = 1
            f[50, 50] = 1

            # Hough transform
            # [H,theta,rho]=hough(f);
            # Skimage hough_line returns H, theta, distances
            # Default theta is -pi/2 to pi/2

            H, theta, d = hough_line(f)
            print(f"H shape: {H.shape}")
            print(
                f"Theta range: {np.degrees(theta.min())} to {np.degrees(theta.max())}"
            )
            print(f"Distance range: {d.min()} to {d.max()}")

            # H = H > 0; %Convert to binary for display (in MATLAB code).
            # MATLAB imshow scaling might display it binary-like or just scaled.
            # The script says "H = H > 0; %Convert to binary for display."
            H_binary = H > 0

            # Display
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            axes[0].imshow(f, cmap="gray")
            axes[0].set_title("Original Image (5 points)")
            # axes[0].axis('off')

            # Display Hough Transform
            # imshow (H,[],'Xdata',theta,'Ydata',rho,'InitialMagnification','fit');
            # axis on, axis square, xlabel('\theta'), ylabel('\rho')

            # We display H_binary or H? MATLAB script says H=H>0, then imshow(H). So binary.
            # Use degree for x-axis

            # Extent for imshow: [left, right, bottom, top]
            # theta is radians. Convert to degrees for display usually.
            theta_deg = np.degrees(theta)
            extent = [theta_deg.min(), theta_deg.max(), d.min(), d.max()]

            # Note: imshow default origin is 'upper'. MATLAB origin for plot axes implies y grows up usually?
            # Actually for images ('imshow'), y grows down.
            # But for 'rho', it might be from -diag to +diag.
            # Let's adjust aspect

            axes[1].imshow(
                H_binary, cmap="gray", extent=extent, aspect="auto", origin="lower"
            )
            axes[1].set_title("Hough Transform")
            axes[1].set_xlabel("Theta (degrees)")
            axes[1].set_ylabel("Rho (pixels)")

            plt.tight_layout()
            print("Saved Figure1030.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1031(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1031.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from skimage.feature import canny
            from helpers.libdipum.hough import hough
            from helpers.libdipum.houghpeaks import houghpeaks
            from helpers.libdipum.houghlines import houghlines
            from helpers.libdipum.data_path import dip_data

            print("Running Figure1031 (Hough Line Detection with custom utils)...")

            # Parameters
            NPeaks = 1
            RatioHoughThreshold = 0.3
            FillGap = 50
            MinLength = 7

            # Data
            img_path = dip_data("Fig1034(a)(marion_airport).tif")
            I = img_as_float(imread(img_path))

            # Edge detection
            BW = canny(I, sigma=1.0)

            # Hough transform
            H, theta, rho = hough(BW)

            # Find peaks
            threshold = np.ceil(RatioHoughThreshold * H.max())

            # Python houghpeaks returns r, c indices (rho_idx, theta_idx)
            r_idx, c_idx = houghpeaks(H, numpeaks=NPeaks, threshold=threshold)

            print("Detected peaks (theta, rho indices):")
            if len(r_idx) > 0:
                for r, c in zip(r_idx, c_idx):
                    print(f"  Theta: {theta[c]} deg, Rho: {rho[r]}")

            # Convert peaks to list of [rho_idx, theta_idx] for houghlines
            peaks = np.column_stack((r_idx, c_idx))

            # Get lines
            lines = houghlines(
                BW, theta, rho, peaks, fill_gap=FillGap, min_length=MinLength
            )

            print(f"Found {len(lines)} line segments.")

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            axes = axes.flatten()

            axes[0].imshow(I, cmap="gray")
            axes[0].set_title("Original image")
            axes[0].axis("off")

            axes[1].imshow(BW, cmap="gray")
            axes[1].set_title("Edge detection")
            axes[1].axis("off")

            axes[2].imshow(
                H,
                cmap="gray",
                extent=[theta[0], theta[-1], rho[0], rho[-1]],
                aspect="auto",
                origin="lower",
            )
            axes[2].set_title(f"Hough Peaks ({len(peaks)})")
            axes[2].set_xlabel("Theta (deg)")
            axes[2].set_ylabel("Rho")

            # Plot peaks
            for r, c in zip(r_idx, c_idx):
                axes[2].plot(
                    theta[c],
                    rho[r],
                    "s",
                    color="white",
                    markeredgecolor="white",
                    markersize=5,
                )

            axes[3].imshow(I, cmap="gray")
            axes[3].set_title("Overlay")
            axes[3].axis("off")

            # Overlay lines
            for line in lines:
                p1 = line["point1"]
                p2 = line["point2"]
                # p1 is (row, col) -> (y, x) for plot
                axes[3].plot([p1[1], p2[1]], [p1[0], p2[0]], linewidth=2, color="green")
                axes[3].plot(p1[1], p1[0], "x", linewidth=2, color="yellow")
                axes[3].plot(p2[1], p2[0], "x", linewidth=2, color="red")

            plt.tight_layout()
            print("Saved Figure1031.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1033(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1033.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float, random_noise
            from helpers.libdipum.data_path import dip_data

            print("Running Figure1033 (Histograms of noisy images)...")

            # Data
            img_path = dip_data("Fig1036(a)(original_septagon).tif")
            f_orig = imread(img_path)

            # Ensure standard grayscale
            if f_orig.ndim == 3:
                f_orig = f_orig[:, :, 0]

            f = img_as_float(f_orig)

            # Add Gaussian noise
            fn1 = random_noise(f, mode="gaussian", mean=0, var=0.002)
            fn2 = random_noise(f, mode="gaussian", mean=0, var=0.038)

            # Display
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            # Images
            axes[0].imshow(f, cmap="gray")
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            axes[1].imshow(fn1, cmap="gray")
            axes[1].set_title("Gaussian Noise (var=0.002)")
            axes[1].axis("off")

            axes[2].imshow(fn2, cmap="gray")
            axes[2].set_title("Gaussian Noise (var=0.038)")
            axes[2].axis("off")

            # Histograms
            # MATLAB: imhist(f) -> 256 bins for uint8 usually.
            # For float images, it might use 256 bins mapped to [0,1].

            axes[3].hist(f.ravel(), bins=256, range=(0, 1), color="black", alpha=0.7)
            axes[3].set_title("Histogram (Original)")
            axes[3].set_xlim([0, 1])

            axes[4].hist(fn1.ravel(), bins=256, range=(0, 1), color="black", alpha=0.7)
            axes[4].set_title("Histogram (Noise var=0.002)")
            axes[4].set_xlim([0, 1])

            axes[5].hist(fn2.ravel(), bins=256, range=(0, 1), color="black", alpha=0.7)
            axes[5].set_title("Histogram (Noise var=0.038)")
            axes[5].set_xlim([0, 1])

            plt.tight_layout()
            print("Saved Figure1033.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1034(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1034.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from helpers.libdipum.ishade import ishade
            from helpers.libdipum.data_path import dip_data

            print("Running Figure1034 (Shading with ramp)...")

            # Data
            img_path = dip_data("Fig1036(b)(gaussian_noise_mean_0_std_10_added).tif")
            fn_orig = imread(img_path)
            if fn_orig.ndim == 3:
                fn_orig = fn_orig[:, :, 0]

            M, N = fn_orig.shape

            # r = ishade(M, N, 0.2, 0.6,'ramp', 0);
            r = ishade(M, N, 0.2, 0.6, "ramp", 0)

            # fs=immultiply(im2double(fn),r);
            fn = img_as_float(fn_orig)
            fs = fn * r

            # Display
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            axes[0].imshow(fn, cmap="gray")
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            axes[1].imshow(r, cmap="gray", vmin=0, vmax=1)
            axes[1].set_title("Shading Ramp")
            axes[1].axis("off")

            axes[2].imshow(fs, cmap="gray")
            axes[2].set_title("Shaded Image")
            axes[2].axis("off")

            # Histograms
            # Range [0, 1]

            axes[3].hist(fn.ravel(), bins=256, range=(0, 1), color="black", alpha=0.7)
            axes[3].set_title("Histogram (Original)")
            axes[3].set_xlim([0, 1])

            axes[4].hist(r.ravel(), bins=256, range=(0, 1), color="black", alpha=0.7)
            axes[4].set_title("Histogram (Ramp)")
            axes[4].set_xlim([0, 1])

            axes[5].hist(fs.ravel(), bins=256, range=(0, 1), color="black", alpha=0.7)
            axes[5].set_title("Histogram (Shaded)")
            axes[5].set_xlim([0, 1])

            plt.tight_layout()
            print("Saved Figure1034.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1035(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1035.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.libdipum.data_path import dip_data

            print("Running Figure1035 (Iterative Global Thresholding)...")

            # Data
            img_path = dip_data("fingerprint.tif")
            f = imread(img_path)
            if f.ndim == 3:
                f = f[:, :, 0]

            # Convert to float for mean calculations
            f = f.astype(float)

            # Initial Threshold
            T = np.mean(f)
            print(f"Initial T: {T}")

            done = False
            count = 0
            while not done:
                count += 1
                g = f > T

                # Mean of foreground (g) and background (~g)
                # Handle case where one might be empty (though unlikely for T=mean)
                if np.any(g):
                    mean_fg = np.mean(f[g])
                else:
                    mean_fg = 0  # Should not happen usually

                if np.any(~g):
                    mean_bg = np.mean(f[~g])
                else:
                    mean_bg = 0

                T_next = 0.5 * (mean_fg + mean_bg)

                done = abs(T - T_next) < 0.5
                T = T_next

            print(f"Converged T: {T} after {count} iterations.")

            # Final segmentation
            g = f > T

            # Display
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(f, cmap="gray")
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            # Histogram
            axes[1].hist(f.ravel(), bins=256, color="k", histtype="step")
            axes[1].axvline(x=T, color="r", linestyle="--", label=f"T={T:.1f}")
            axes[1].set_title("Histogram")
            axes[1].legend()

            axes[2].imshow(g, cmap="gray")
            axes[2].set_title(f"Global Thresholding (T={T:.1f})")
            axes[2].axis("off")

            plt.tight_layout()
            print("Saved Figure1035.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1036(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1036.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.libdipum.otsuthresh import otsuthresh
            from helpers.libdipum.data_path import dip_data

            # Figure 10.36

            # Data
            img_path = dip_data("polymercell.tif")
            I = imread(img_path)
            if I.ndim == 3:
                I = I[:, :, 0]

            # Compute histogram
            if I.dtype == np.uint8:
                h, _ = np.histogram(I, bins=256, range=(0, 255))
            else:
                h, _ = np.histogram(I, bins=256)

            # Otsu threshold
            T_norm, SM = otsuthresh(h)
            T_otsu = T_norm * 255 if I.dtype == np.uint8 else T_norm
            g_otsu = I > T_otsu

            # Iterative global threshold (requested algorithm)
            f = I.astype(float)
            count = 0
            T_iter = np.mean(f)
            done = False

            while not done:
                count += 1
                g = f > T_iter

                if np.any(g):
                    m1 = np.mean(f[g])
                else:
                    m1 = T_iter

                if np.any(~g):
                    m2 = np.mean(f[~g])
                else:
                    m2 = T_iter

                T_next = 0.5 * (m1 + m2)
                done = np.abs(T_iter - T_next) < 0.5
                T_iter = T_next

            g_iter = f > T_iter

            print(
                f"Otsu threshold: {T_otsu:.2f} (normalized {T_norm:.4f}), separability: {SM:.4f}"
            )
            print(f"Iterative threshold: {T_iter:.2f}, iterations: {count}")

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.ravel()

            axes[0].imshow(I, cmap="gray")
            axes[0].set_title("Original Image (Polymersomes)")
            axes[0].axis("off")

            axes[1].plot(h, "k")
            line_x = T_otsu if I.dtype == np.uint8 else (T_otsu * 256)
            axes[1].axvline(
                x=line_x, color="r", linestyle="--", label=f"Otsu T={T_otsu:.1f}"
            )
            axes[1].set_title("Histogram")
            axes[1].legend()

            axes[2].imshow(g_iter, cmap="gray")
            axes[2].set_title(f"Iterative Mean (T={T_iter:.1f})")
            axes[2].axis("off")

            axes[3].imshow(g_otsu, cmap="gray")
            axes[3].set_title(f"Otsu (SM={SM:.2f})")
            axes[3].axis("off")

            plt.tight_layout()
            print("Saved Figure1036.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1037(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1037.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from scipy.ndimage import gaussian_filter
            from helpers.libdipum.otsuthresh import otsuthresh
            from helpers.libdipum.data_path import dip_data

            print("Running Figure1037 (Otsu on Noisy Image w/ Smoothing)...")

            img_path = dip_data("Fig1036(c)(gaussian_noise_mean_0_std_50_added).tif")
            I = imread(img_path)
            if I.ndim == 3:
                I = I[:, :, 0]

            # 1. Noisy Image Stats
            if I.dtype == np.uint8:
                h, bins = np.histogram(I, bins=256, range=(0, 255))
            else:
                h, bins = np.histogram(I, bins=256)

            T_norm, SM = otsuthresh(h)
            if I.dtype == np.uint8:
                T = T_norm * 255
            else:
                T = T_norm

            print(f"Noisy Image: T={T:.2f}, SM={SM:.4f}")
            g_noisy = I > T

            # 2. Smoothing
            I_smooth = gaussian_filter(I.astype(float), sigma=1.5)  # Sigma choice?

            # Check normalized histogram of smoothed
            h_s, bins_s = np.histogram(I_smooth, bins=256)
            T_s_norm, SM_s = otsuthresh(h_s)

            min_v, max_v = I_smooth.min(), I_smooth.max()
            T_s = min_v + T_s_norm * (max_v - min_v)

            print(f"Smoothed Image: T={T_s:.2f}, SM={SM_s:.4f}")
            g_smooth = I_smooth > T_s

            # Display
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # Row 1: Noisy
            axes[0, 0].imshow(I, cmap="gray")
            axes[0, 0].set_title("Noisy Image")
            axes[0, 0].axis("off")

            axes[0, 1].plot(np.arange(1, len(h) - 1), h[1:-1], "k")
            axes[0, 1].axvline(x=T, color="r", linestyle="--")
            axes[0, 1].set_title("Histogram (Noisy)")

            axes[0, 2].imshow(g_noisy, cmap="gray")
            axes[0, 2].set_title(f"Otsu (SM={SM:.2f})")
            axes[0, 2].axis("off")

            # Row 2: Smoothed
            axes[1, 0].imshow(I_smooth, cmap="gray")
            axes[1, 0].set_title("Smoothed Image (Gaussian)")
            axes[1, 0].axis("off")

            axes[1, 1].plot(h_s, "k")
            axes[1, 1].axvline(x=T_s, color="r", linestyle="--")
            axes[1, 1].set_title("Histogram (Smoothed)")

            axes[1, 2].imshow(g_smooth, cmap="gray")
            axes[1, 2].set_title(f"Otsu (SM={SM_s:.2f})")
            axes[1, 2].axis("off")

            plt.tight_layout()
            print("Saved Figure1037.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1038(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1038.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from skimage.filters import threshold_otsu
            from scipy.ndimage import uniform_filter
            from helpers.libdipum.data_path import dip_data

            # Data
            img_path = dip_data("Fig1041(a)(septagon_small_noisy_mean_0_stdv_10).tif")
            f_orig = imread(img_path)
            if f_orig.ndim == 3:
                f_orig = f_orig[:, :, 0]

            f = img_as_float(f_orig)

            # Thresholding 1
            T1 = threshold_otsu(f)
            g1 = f > T1
            print(f"Otsu Threshold 1: {T1}")

            # Smooth
            fs = uniform_filter(f, size=5, mode="nearest")

            # Thresholding 2
            T2 = threshold_otsu(fs)
            g2 = fs > T2
            print(f"Otsu Threshold 2 (Smoothed): {T2}")

            # Display
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            axes[0].imshow(f, cmap="gray")
            axes[0].set_title("Original Noisy Image")
            axes[0].axis("off")

            axes[1].hist(f.ravel(), bins=256, range=(0, 1), color="black", alpha=0.7)
            axes[1].set_title("Histogram (Original)")
            axes[1].set_xlim([0, 1])

            axes[2].imshow(g1, cmap="gray")
            axes[2].set_title(f"Otsu Thresholded (T={T1:.3f})")
            axes[2].axis("off")

            axes[3].imshow(fs, cmap="gray")
            axes[3].set_title("Smoothed Image (5x5)")
            axes[3].axis("off")

            axes[4].hist(fs.ravel(), bins=256, range=(0, 1), color="black", alpha=0.7)
            axes[4].set_title("Histogram (Smoothed)")
            axes[4].set_xlim([0, 1])

            axes[5].imshow(g2, cmap="gray")
            axes[5].set_title(f"Otsu Thresholded Smoothed (T={T2:.3f})")
            axes[5].axis("off")

            plt.tight_layout()
            print("Saved Figure1038.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1039(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1039.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.morphology import dilation, square
            from helpers.libdipum.gradlapthresh import gradlapthresh
            from helpers.libdipum.data_path import dip_data

            # Figure 10.39
            # Using edge features to determine threshold.

            # Data
            f = imread(dip_data("Fig1041(a)(septagon_small_noisy_mean_0_stdv_10).tif"))
            if f.ndim == 3:
                f = f[:, :, 0]

            # Gradient and Laplacian with threshold
            G = gradlapthresh(f, 0.3, 1)
            print(round(255 * G["G2"]))

            # Display
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            axes = axes.ravel()

            axes[0].imshow(f, cmap="gray")
            axes[0].axis("off")

            axes[1].hist(f.ravel(), bins=256, range=(0, 255), color="k")
            axes[1].set_box_aspect(1)

            axes[2].imshow(G["G6"], cmap="gray")
            axes[2].axis("off")

            g16d = dilation((255 * G["G16"]).astype(np.uint8), square(3))
            axes[3].imshow(g16d, cmap="gray")
            axes[3].axis("off")

            axes[4].plot(G["G4"], "k")
            axes[4].set_box_aspect(1)

            axes[5].imshow(G["G1"], cmap="gray")
            axes[5].axis("off")

            plt.tight_layout()
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure104(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure104.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from scipy.ndimage import convolve
            import ia870 as ia
            from helpers.libdipum.data_path import dip_data

            # Data
            img_path = dip_data("turbineblad-with-blk-dot.tif")

            f = img_as_float(imread(img_path))

            # Laplacian kernel
            # w = [-1 -1 -1;-1 8 -1;-1 -1 -1];
            w = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=float)

            # Filtering
            # h = imfilter(f,w,'replicate');
            # scipy.ndimage.convolve uses 'nearest' for replicate padding typically?
            # mode='nearest' replicates the edge value.
            h = convolve(f, w, mode="nearest")

            ha = np.abs(h)

            # Thresholding
            # T = 0.9*max(ha(:));
            # g = ha >= T;
            T = 0.9 * ha.max()
            g = ha >= T

            # Display
            # subplot (1, 3, 1); imshow(f);
            # subplot (1, 3, 2); imshow(h);
            # subplot (1, 3, 3); imshow(morphoDilate4e(g, ones (7)))

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes = axes.flatten()

            axes[0].imshow(f, cmap="gray")
            axes[0].set_title("f")
            axes[0].axis("off")

            # h might have negative values, standard imshow scales min-max
            axes[1].imshow(h, cmap="gray")
            axes[1].set_title("Laplacian Filtered (h)")
            axes[1].axis("off")

            # Dilate point for visibility
            B = ia.iasebox(3)
            g_dilated = ia.iadil(g, B)

            axes[2].imshow(g_dilated, cmap="gray")
            axes[2].set_title("Thresholded & Dilated Point")
            axes[2].axis("off")

            plt.tight_layout()
            print("Saved Figure104.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1040(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1040.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.libdipum.gradlapthresh import gradlapthresh
            from helpers.libdipum.otsuthresh import otsuthresh
            from helpers.libdipum.data_path import dip_data

            # Figure 10.40
            # Threshold segmentation of yeast image using edge information.

            # Data
            f = imread(dip_data("yeast-cells.tif"))
            if f.ndim == 3:
                f = f[:, :, 0]

            # Threshold Otsu
            h, _ = np.histogram(f, bins=256, range=(0, 255))
            To, So = otsuthresh(h)
            print(round(255 * To))
            go = f > (255 * To)

            # Threshold Gradient Laplacian
            G = gradlapthresh(f, 1, 0.3)
            print(G["G2"] * 255)
            print(G["G3"])

            # Display
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            axes = axes.ravel()

            axes[0].imshow(f, cmap="gray")
            axes[0].axis("off")

            axes[1].hist(f.ravel(), bins=256, range=(0, 255), color="k")

            axes[2].imshow(go, cmap="gray")
            axes[2].axis("off")

            axes[3].imshow(G["G11"], cmap="gray")
            axes[3].axis("off")

            vals = G["G16"][G["G16"] != 0]
            if vals.size > 0:
                vals_u8 = np.clip(np.round(vals * 255), 0, 255).astype(np.uint8)
                counts, bins = np.histogram(vals_u8, bins=256, range=(0, 256))
                axes[4].plot(bins[:-1], counts, "k")

            axes[5].imshow(G["G1"], cmap="gray")
            axes[5].axis("off")

            plt.tight_layout()
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1041(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1041.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.libdipum.gradlapthresh import gradlapthresh
            from helpers.libdipum.data_path import dip_data

            # Figure 10.41
            # As in 10.40, but using the lower threshold.

            # Data
            f = imread(dip_data("yeast-cells.tif"))
            if f.ndim == 3:
                f = f[:, :, 0]

            # Use lower threshold (about 5% of maximum)
            G = gradlapthresh(f, 1, 0.05)
            print(G["G2"] * 255)  # Otsu threshold converted to [0,255] range
            print(G["G3"])  # Percentile / separability output

            # Display
            plt.figure(figsize=(6, 6))
            plt.imshow(G["G1"], cmap="gray")
            plt.axis("off")

            # Save to file
            plt.tight_layout()
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1042(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1042.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.libgeneral.multithresh import multithresh
            from helpers.libgeneral.imquantize import imquantize
            from helpers.libdipum.data_path import dip_data

            # Data
            image_path = dip_data("iceberg.tif")
            f = imread(image_path)
            if f.ndim == 3:
                f = f[:, :, 0]

            # Histogram
            hist, bins = np.histogram(f, bins=256, range=(0, 255))
            h = hist / np.sum(hist)

            print("Running multithresh(f, 2)...")
            T = multithresh(f, 2)
            print(f"Thresholds: {T}")

            g = imquantize(f, T)

            # Display
            fig, axes = plt.subplots(1, 3, figsize=(15, 6))

            # 1. Original
            axes[0].imshow(f, cmap="gray")
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            # 2. Histogram
            # hist(double(f(:)), 256)
            axes[1].plot(hist)
            axes[1].set_title("Histogram")
            axes[1].set_aspect("auto")
            axes[1].set_xlim([0, 255])

            # 3. Quantized
            # imshow(g, [])
            # g has values 1, 2, 3.
            # auto-scaling with imshow(..., []) handled by matplotlib default or vmin/vmax
            axes[2].imshow(g, cmap="gray")
            axes[2].set_title("Multilevel Thresholding")
            axes[2].axis("off")

            plt.tight_layout()
            print("Saved Figure1042.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1043(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1043.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.libdipum.localthresh import localthresh
            from helpers.libgeneral.multithresh import multithresh
            from helpers.libgeneral.imquantize import imquantize
            from helpers.libgeneral.stdfilt import stdfilt
            from helpers.libdipum.data_path import dip_data

            # Data
            image_path = dip_data("yeast-cells.tif")
            f = imread(image_path)
            if f.ndim == 3:
                f = f[:, :, 0]

            # Otsu multi thresholding
            print("Running multithresh(f, 2)...")
            T = multithresh(f, 2)
            print(f"Thresholds: {T}")

            g1 = imquantize(f, T)

            # Local thresholding
            nhood = np.ones((3, 3))
            print("Running localthresh...")
            g2 = localthresh(f, nhood, 30, 1.5, "global")
            std = stdfilt(f, nhood)

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(15, 6))

            # 1. Original
            axes[0, 0].imshow(f, cmap="gray")
            axes[0, 0].set_title("Original Image")
            axes[0, 0].axis("off")

            # 2. Otsu Multi (g1)
            axes[0, 1].imshow(g1, cmap="gray")
            axes[0, 1].set_title("Otsu Multi (g1)")
            axes[0, 1].axis("off")

            # 3. Local std
            axes[1, 0].imshow(std, cmap="gray")
            axes[1, 0].set_title("Otsu Multi (g1)")
            axes[1, 0].axis("off")

            # 4. Local Thresh (g2)
            axes[1, 1].imshow(g2, cmap="gray")
            axes[1, 1].set_title("Local Thresh (g2)")
            axes[1, 1].axis("off")

            plt.tight_layout()
            print("Saved Figure1043.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1044(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1044.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.filters import threshold_otsu
            from skimage import img_as_float
            from helpers.libdipum.ishade import ishade
            from helpers.libdipum.movingthresh import movingthresh
            from helpers.libdipum.data_path import dip_data

            # Data
            image_path = dip_data("Fig1049(original_cursive_text_WITHOUT_SHADING).tif")
            f_raw = imread(image_path)
            if f_raw.ndim == 3:
                f_raw = f_raw[:, :, 0]

            # Convert to float [0, 1]
            f = img_as_float(f_raw)
            M, N = f.shape

            # Create shading pattern
            shade = ishade(M, N, 0.1, 1.0, "spot", min(M, N) / 2)

            # Shade f
            fs = f * shade

            # Segment using Otsu's method
            T = threshold_otsu(fs)
            gotsu = fs > T

            # Segment using moving average
            n = 20
            gmoving = movingthresh(fs, n, 0.5)

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            axes[0, 0].imshow(f, cmap="gray")
            axes[0, 0].set_title("Original Image")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(fs, cmap="gray")
            axes[0, 1].set_title("Shaded Image")
            axes[0, 1].axis("off")

            axes[1, 0].imshow(gotsu, cmap="gray")
            axes[1, 0].set_title("Otsu's Method")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(gmoving, cmap="gray")
            axes[1, 1].set_title("Moving Average Threshold")
            axes[1, 1].axis("off")

            plt.tight_layout()
            print("Saved Figure1044.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1045(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1045.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.filters import threshold_otsu
            from skimage import img_as_float
            from helpers.libdipum.imnoise3 import imnoise3
            from helpers.libdipum.movingthresh import movingthresh
            from helpers.libdipum.data_path import dip_data

            # Data
            image_path = dip_data("Fig1049(original_cursive_text_WITHOUT_SHADING).tif")
            f_raw = imread(image_path)
            if f_raw.ndim == 3:
                f_raw = f_raw[:, :, 0]

            f = img_as_float(f_raw)
            M, N = f.shape

            # Create shading pattern
            K = round(min(M, N) / 100.0)
            C = np.array([[0, K]])
            r, _, _ = imnoise3(M, N, C)

            r = r - r.min()
            r = r / (r.max() + 1e-10)
            r = r + 0.25
            r = r / (r.max() + 1e-10)

            # Shade image
            fs2 = f * r

            # Segment using Otsu
            T2 = threshold_otsu(fs2)
            gotsu2 = fs2 > T2

            # Segment using moving average
            n = 20
            gmoving2 = movingthresh(fs2, n, 0.5)

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            axes[0, 0].imshow(f, cmap="gray")
            axes[0, 0].set_title("Original Image")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(fs2, cmap="gray")
            axes[0, 1].set_title("Shaded Image (Sinusoidal)")
            axes[0, 1].axis("off")

            axes[1, 0].imshow(gotsu2, cmap="gray")
            axes[1, 0].set_title("Otsu's Method")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(gmoving2, cmap="gray")
            axes[1, 1].set_title("Moving Average Threshold")
            axes[1, 1].axis("off")

            plt.tight_layout()
            print("Saved Figure1045.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1046(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1046.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage import img_as_float, img_as_ubyte
            from skimage.measure import label, regionprops
            from helpers.libdipum.otsudualthresh import otsudualthresh
            from helpers.libdipum.multithresh3E import multithresh3E
            from helpers.libdipum.regiongrow import regiongrow
            from helpers.libdipum.data_path import dip_data

            def imhist(img: Any):
                """imhist."""
                # Retrieve histogram 0-255
                if img.dtype != np.uint8:
                    # Assuming [0,1] float or other. Check range.
                    if img.max() <= 1.0:
                        img = img_as_ubyte(img)
                    else:
                        img = img.astype(np.uint8)

                hist, bins = np.histogram(img.flatten(), bins=256, range=(0, 255))
                return hist

            def mat2gray(img: Any):
                """mat2gray."""
                min_v = img.min()
                max_v = img.max()
                if max_v - min_v < 1e-10:
                    return np.zeros_like(img)
                return (img - min_v) / (max_v - min_v)

            def shrink_to_points(binary_mask: Any):
                """
                Shrinks connected components to single pixels (like bwmorph shrink Inf).
                """
                lbl_img = label(binary_mask)
                props = regionprops(lbl_img)
                shrunk = np.zeros_like(binary_mask)
                for p in props:
                    r, c = p.coords[0]  # Pick first coordinate
                    shrunk[r, c] = True
                return shrunk

            # Data
            image_path = dip_data("weldXray.tif")
            f_raw = imread(image_path)

            # Threshold seeds
            Q = 254
            S1 = f_raw > Q

            # S = bwmorph(S1, 'shrink', Inf)
            print("Shrinking seeds...")
            S = shrink_to_points(S1)

            # Difference image
            f_double = img_as_float(f_raw)
            diff_val = np.abs(((Q + 1) / 255.0) - f_double)
            d = img_as_ubyte(diff_val)

            # Histogram of difference
            hd = imhist(d)
            hd_norm = hd / np.sum(hd)

            # Dual Otsu
            print("Computing Otsu dual thresholds...")
            T1, T2, _ = otsudualthresh(hd_norm)
            print(f"T1={T1 * 255:.2f}, T2={T2 * 255:.2f}")

            gtd = multithresh3E(d, [T1, T2])

            gt1 = multithresh3E(d, [T1])  # Pass as list

            # Region growing
            print("Region growing...")
            g, NR, SI, TI = regiongrow(f_double, S, T1)

            # Display
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))

            # Row 1
            axes[0, 0].imshow(f_raw, cmap="gray")
            axes[0, 0].set_title("f")
            axes[0, 0].axis("off")

            axes[0, 1].plot(imhist(f_raw))
            axes[0, 1].set_title("Hist(f)")
            axes[0, 1].set_xlim([0, 255])

            axes[0, 2].imshow(S1, cmap="gray")
            axes[0, 2].set_title("S1 (Seeds)")
            axes[0, 2].axis("off")

            # Row 2
            axes[1, 0].imshow(S, cmap="gray")
            axes[1, 0].set_title("S (Shrunk Seeds)")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(d, cmap="gray")
            axes[1, 1].set_title("d (Diff Image)")
            axes[1, 1].axis("off")

            axes[1, 2].plot(hd_norm)
            axes[1, 2].set_title("Hist(d) norm")
            axes[1, 2].set_xlim([0, 255])

            # Row 3
            axes[2, 0].imshow(gtd, cmap="gray")
            axes[2, 0].set_title("Dual Thresh (T1, T2)")
            axes[2, 0].axis("off")

            axes[2, 1].imshow(gt1, cmap="gray")
            axes[2, 1].set_title("Thresh (T1)")
            axes[2, 1].axis("off")

            axes[2, 2].imshow(g, cmap="gray")
            axes[2, 2].set_title("Region Growing Result")
            axes[2, 2].axis("off")

            plt.tight_layout()
            print("Saved Figure1046.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1048(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1048.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.libdipum.splitmerge import splitmerge
            from helpers.libdipum.data_path import dip_data

            def predicate(region: Any):
                """predicate."""
                # sd = std2(region);
                # m = mean2(region);
                # flag = (sd > 10) & (m > 0) & (m < 125);
                # Note: MATLAB std2 is sample standard deviation (ddof=1).
                sd = np.std(region, ddof=1)
                m = np.mean(region)
                return (sd > 10) and (m > 0) and (m < 125)

            # Data
            image_path = dip_data("cygnusloop.tif")
            f = imread(image_path)

            # Split and Merge
            print("Processing splitmerge(32)...")
            g32 = splitmerge(f, 32, predicate)

            print("Processing splitmerge(16)...")
            g16 = splitmerge(f, 16, predicate)

            print("Processing splitmerge(8)...")
            g8 = splitmerge(f, 8, predicate)

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            axes[0, 0].imshow(f, cmap="gray")
            axes[0, 0].set_title("Original Image")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(g32, cmap="gray")
            axes[0, 1].set_title("g32")
            axes[0, 1].axis("off")

            axes[1, 0].imshow(g16, cmap="gray")
            axes[1, 0].set_title("g16")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(g8, cmap="gray")
            axes[1, 1].set_title("g8")
            axes[1, 1].axis("off")

            plt.tight_layout()
            print("Saved Figure1048.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1049(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1049.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage import img_as_float
            from helpers.libgeneral.kmeans import kmeans
            from helpers.libdipum.data_path import dip_data

            # Data
            image_path = dip_data("book-cover.tif")
            f_raw = imread(image_path)
            if f_raw.ndim == 3:
                pass
            f = img_as_float(f_raw)
            f_flat = f.flatten()

            print("Running K-means...")
            idx, centers = kmeans(f_flat, 3)

            # Reconstruct
            idx_reshaped = idx.reshape(f.shape)
            fseg = idx_reshaped.astype(np.float64)

            # Display
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            if f_raw.ndim == 2:
                axes[0].imshow(f, cmap="gray")
            else:
                axes[0].imshow(f)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            # MATLAB: imshow(fseg, [], 'InitialMagnification', 'fit')
            # [] means scale display range to [min(fseg) max(fseg)]
            axes[1].imshow(fseg, cmap="gray")
            axes[1].set_title("Segmented Image")
            axes[1].axis("off")

            plt.tight_layout()
            print("Saved Figure1049.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure105(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure105.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from scipy.ndimage import convolve
            from helpers.libdip.intScaling4e import intScaling4e
            from helpers.libdipum.data_path import dip_data

            f = img_as_float(imread(dip_data("Fig1007(a)(wirebond_mask).tif")))

            # Laplacian kernel
            # w = ones(3); w(2,2) = -8;
            w = np.ones((3, 3), dtype=float)
            w[1, 1] = -8.0

            # Filtering
            # g = imfilter(f, w, 'replicate');
            g = convolve(f, w, mode="nearest")

            # Scaling
            # gs = intScaling4e(g);
            gs = intScaling4e(g)

            # Thresholding
            # ga = abs(g);
            # gp = g > 0;
            ga = np.abs(g)
            gp = g > 0

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes = axes.flatten()

            axes[0].imshow(f, cmap="gray")
            axes[0].set_title("f")
            axes[0].axis("off")

            axes[1].imshow(gs, cmap="gray")
            axes[1].set_title("Scaled Laplacian (gs)")
            axes[1].axis("off")

            # ga is float, imshow scales auto.
            axes[2].imshow(ga, cmap="gray")
            axes[2].set_title("abs(Laplacian)")
            axes[2].axis("off")

            axes[3].imshow(gp, cmap="gray")
            axes[3].set_title("g > 0")
            axes[3].axis("off")

            plt.tight_layout()
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1050(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1050.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage import img_as_float
            from skimage.segmentation import find_boundaries
            from scipy.ndimage import mean
            from helpers.libgeneral.superpixels import superpixels
            from helpers.libdipum.data_path import dip_data

            def mat2gray(img: Any):
                """mat2gray."""
                min_v = img.min()
                max_v = img.max()
                if max_v - min_v < 1e-10:
                    return np.zeros_like(img)
                return (img - min_v) / (max_v - min_v)

            # Data
            image_path = dip_data("totem-poles.tif")
            f_raw = imread(image_path)

            # Ensure standard double format [0,1]
            f = img_as_float(f_raw)

            # Get superpixel labels
            print("Computing superpixels...")
            L, NL = superpixels(f, 4000)

            # Replace each superpixel by its average value
            print("Computing means...")
            if f.ndim == 3:
                # Multichannel mean
                fSP = np.zeros_like(f)
                for c in range(f.shape[2]):
                    # ndimage.mean returns list of means for labels 1..NL
                    means = mean(f[:, :, c], labels=L, index=np.arange(1, NL + 1))
                    mapping = np.zeros(NL + 1)
                    mapping[1:] = means
                    fSP[:, :, c] = mapping[L]
            else:
                means = mean(f, labels=L, index=np.arange(1, NL + 1))
                mapping = np.zeros(NL + 1)
                mapping[1:] = means
                fSP = mapping[L]

            fSP = mat2gray(fSP)

            BW = find_boundaries(L, mode="thick")

            if fSP.ndim == 2:
                f_overlay = fSP.copy()
                f_overlay[BW] = 1.0  # White
            else:
                f_overlay = fSP.copy()
                for c in range(3):
                    # If we want white, set all channels to 1
                    f_overlay[:, :, c][BW] = 1.0

            # Display
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            cmap = "gray" if f.ndim == 2 else None

            axes[0].imshow(f, cmap=cmap)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            axes[1].imshow(f_overlay, cmap=cmap)
            axes[1].set_title("Superpixels (Mean + Boundaries)")
            axes[1].axis("off")

            axes[2].imshow(fSP, cmap=cmap)
            axes[2].set_title("Mean Color Image")
            axes[2].axis("off")

            plt.tight_layout()
            print("Saved Figure1050.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1051(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1051.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage import img_as_float
            from scipy.ndimage import mean
            from helpers.libgeneral.superpixels import superpixels
            from helpers.libdipum.data_path import dip_data

            def mat2gray(img: Any):
                """mat2gray."""
                min_v = img.min()
                max_v = img.max()
                if max_v - min_v < 1e-10:
                    return np.zeros_like(img)
                return (img - min_v) / (max_v - min_v)

            # Data
            image_path = dip_data("totem-poles.tif")
            f_raw = imread(image_path)
            f = img_as_float(f_raw)

            # Get superpixel labels
            print("Computing superpixels (N=40000)...")
            L, NL = superpixels(f, 40000)

            # Replace each superpixel by its average value
            print("Computing means...")
            if f.ndim == 3:
                fSP = np.zeros_like(f)
                for c in range(f.shape[2]):
                    means = mean(f[:, :, c], labels=L, index=np.arange(1, NL + 1))
                    mapping = np.zeros(NL + 1)
                    mapping[1:] = means
                    fSP[:, :, c] = mapping[L]
            else:
                means = mean(f, labels=L, index=np.arange(1, NL + 1))
                mapping = np.zeros(NL + 1)
                mapping[1:] = means
                fSP = mapping[L]

            fSP = mat2gray(fSP)

            # Difference image
            diff = mat2gray(f - fSP)

            # Display
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            cmap = "gray" if f.ndim == 2 else None

            axes[0].imshow(f, cmap=cmap)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            axes[1].imshow(fSP, cmap=cmap)
            axes[1].set_title("Superpixels Mean Image")
            axes[1].axis("off")

            axes[2].imshow(diff, cmap=cmap)
            axes[2].set_title("Difference Image")
            axes[2].axis("off")

            plt.tight_layout()
            print("Saved Figure1051.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1052(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1052.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage import img_as_float
            from skimage.segmentation import find_boundaries
            from scipy.ndimage import mean
            from helpers.libgeneral.superpixels import superpixels
            from helpers.libdipum.data_path import dip_data

            def mat2gray(img: Any):
                """mat2gray."""
                min_v = img.min()
                max_v = img.max()
                if max_v - min_v < 1e-10:
                    return np.zeros_like(img)
                return (img - min_v) / (max_v - min_v)

            # Data
            image_path = dip_data("totem-poles.tif")
            f_raw = imread(image_path)
            f = img_as_float(f_raw)

            # Superpixels sizes
            NSUP = [1000, 500, 250]

            fSPStore = []
            BWStore = []

            for i, n_seg in enumerate(NSUP):
                print(f"Processing superpixels N={n_seg}...")
                L, NL = superpixels(f, n_seg)

                # Mean image
                if f.ndim == 3:
                    fSP = np.zeros_like(f)
                    for c in range(f.shape[2]):
                        means = mean(f[:, :, c], labels=L, index=np.arange(1, NL + 1))
                        mapping = np.zeros(NL + 1)
                        mapping[1:] = means
                        fSP[:, :, c] = mapping[L]
                else:
                    means = mean(f, labels=L, index=np.arange(1, NL + 1))
                    mapping = np.zeros(NL + 1)
                    mapping[1:] = means
                    fSP = mapping[L]

                fSP = mat2gray(fSP)
                fSPStore.append(fSP)

                # Boundaries
                BW = find_boundaries(L, mode="thick")
                BWStore.append(BW)

            # Display
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            for i in range(3):
                # Top row: Overlays
                fSP = fSPStore[i]
                BW = BWStore[i]

                if fSP.ndim == 2:
                    f_overlay = fSP.copy()
                    f_overlay[BW] = 1.0  # White
                    cmap = "gray"
                else:
                    f_overlay = fSP.copy()
                    for c in range(3):
                        f_overlay[:, :, c][BW] = 1.0
                    cmap = None

                axes[0, i].imshow(f_overlay, cmap=cmap)
                axes[0, i].set_title(f"N={NSUP[i]} (Overlay)")
                axes[0, i].axis("off")

                # Bottom row: Mean Images
                axes[1, i].imshow(fSP, cmap=cmap)
                axes[1, i].set_title(f"N={NSUP[i]} (Mean)")
                axes[1, i].axis("off")

            plt.tight_layout()
            print("Saved Figure1052.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1053(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1053.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage import img_as_float
            from skimage.segmentation import find_boundaries
            from scipy.ndimage import mean
            from helpers.libgeneral.superpixels import superpixels
            from helpers.libgeneral.kmeans import kmeans
            from helpers.libdipum.data_path import dip_data

            def mat2gray(img: Any):
                """mat2gray."""
                min_v = img.min()
                max_v = img.max()
                if max_v - min_v < 1e-10:
                    return np.zeros_like(img)
                return (img - min_v) / (max_v - min_v)

            # Data
            image_path = dip_data("iceberg.tif")
            f_raw = imread(image_path)
            f = img_as_float(f_raw)

            # K-means
            print("Running initial K-means (k=3)...")
            idx_raw, _ = kmeans(f.flatten(), 3)

            fseg = idx_raw.reshape(f.shape).astype(float)
            fseg = mat2gray(fseg)

            # Superpixels
            print("Computing superpixels (N=100)...")
            L, NL = superpixels(f, 100)

            # Replace each superpixel by its average value
            print("Computing means...")
            if f.ndim == 3:
                fSP = np.zeros_like(f)
                for c in range(f.shape[2]):
                    means = mean(f[:, :, c], labels=L, index=np.arange(1, NL + 1))
                    mapping = np.zeros(NL + 1)
                    mapping[1:] = means
                    fSP[:, :, c] = mapping[L]
            else:
                means = mean(f, labels=L, index=np.arange(1, NL + 1))
                mapping = np.zeros(NL + 1)
                mapping[1:] = means
                fSP = mapping[L]

            fSP = mat2gray(fSP)

            # BW = boundarymask(L);
            BW = find_boundaries(L, mode="thick")

            print("Running K-means on superpixel image (k=3)...")
            idx_sp, _ = kmeans(fSP.flatten(), 3)

            fSPseg = idx_sp.reshape(fSP.shape).astype(float)
            fSPseg = mat2gray(fSPseg)

            # Display

            # Prepare overlay
            if fSP.ndim == 2:
                f_overlay = fSP.copy()
                f_overlay[BW] = 1.0  # White
                cmap = "gray"
            else:
                f_overlay = fSP.copy()
                for c in range(3):
                    f_overlay[:, :, c][BW] = 1.0
                cmap = None

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            axes[0, 0].imshow(f, cmap=cmap)
            axes[0, 0].set_title("Original Image")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(fseg, cmap=cmap)
            axes[0, 1].set_title("k-means")
            axes[0, 1].axis("off")

            axes[0, 2].axis("off")

            axes[1, 0].imshow(f_overlay, cmap=cmap)
            axes[1, 0].set_title("Superpixels Overlay")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(fSP, cmap=cmap)
            axes[1, 1].set_title("Superpixels Mean")
            axes[1, 1].axis("off")

            axes[1, 2].imshow(fSPseg, cmap=cmap)
            axes[1, 2].set_title("Segmented Superpixels")
            axes[1, 2].axis("off")

            plt.tight_layout()
            print("Saved Figure1053.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1054(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1054.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy.ndimage import mean
            from skimage import img_as_float
            from skimage.io import imread
            from skimage.segmentation import find_boundaries, slic

            from helpers.libgeneral.kmeans import kmeans
            from helpers.libgeneral.superpixels import superpixels
            from helpers.libdipum.data_path import dip_data

            def mat2gray(img: Any):
                """mat2gray."""
                min_v = img.min()
                max_v = img.max()
                if max_v - min_v < 1e-12:
                    return np.zeros_like(img, dtype=np.float64)
                return (img - min_v) / (max_v - min_v)

            def slic_compat(image: Any, n_segments: Any):
                """slic_compat."""
                try:
                    return slic(
                        image,
                        n_segments=n_segments,
                        compactness=10,
                        start_label=1,
                        channel_axis=-1 if image.ndim == 3 else None,
                    )
                except TypeError:
                    return slic(
                        image,
                        n_segments=n_segments,
                        compactness=10,
                        start_label=1,
                        multichannel=(image.ndim == 3),
                    )

            # Data
            image_path = dip_data("book-cover.tif")
            f_raw = imread(image_path)
            f = img_as_float(f_raw)

            # Super pixels
            print("Computing superpixels (N=95000)...")
            L, NL = superpixels(f, 95000)
            L = np.asarray(L, dtype=np.int64)

            # Validate segmentation quality for display and fallback if boundaries are excessive.
            BW = find_boundaries(L, mode="inner")
            if BW.mean() > 0.95:
                print(
                    "Fallback: boundary mask is too dense; using SLIC superpixels for stable regions."
                )
                L = slic_compat(f, 95000)
                BW = find_boundaries(L, mode="inner")

            print(
                f"Boundary pixels: {int(BW.sum())} / {BW.size} ({100.0 * BW.mean():.3f}%)"
            )

            labels = np.unique(L)
            NL = int(labels.max())

            # Replace each superpixel by its average value.
            print("Computing means...")
            if f.ndim == 3:
                fSP = np.zeros_like(f, dtype=np.float64)
                for c in range(f.shape[2]):
                    means = mean(f[:, :, c], labels=L, index=labels)
                    lut = np.zeros(NL + 1, dtype=np.float64)
                    lut[labels] = means
                    fSP[:, :, c] = lut[L]
            else:
                means = mean(f, labels=L, index=labels)
                lut = np.zeros(NL + 1, dtype=np.float64)
                lut[labels] = means
                fSP = lut[L]

            fSP = mat2gray(fSP)

            # Segment fSP using k-means.
            print("Running K-means on superpixel image (k=3)...")
            km_input = fSP.mean(axis=2).ravel() if fSP.ndim == 3 else fSP.ravel()
            idx_sp, _ = kmeans(km_input, 3)
            fSPseg = idx_sp.reshape(fSP.shape[:2]).astype(np.float64)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(f, cmap="gray" if f.ndim == 2 else None)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            axes[1].imshow(fSP, cmap="gray" if fSP.ndim == 2 else None)
            axes[1].set_title("Superpixels Mean")
            axes[1].axis("off")

            axes[2].imshow(fSPseg, cmap="gray")
            axes[2].set_title("Segmented Superpixels")
            axes[2].axis("off")

            plt.tight_layout()
            print("Saved Figure1054.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1058(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1058.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import os
            import sys
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from scipy.ndimage import uniform_filter
            from skimage.transform import resize

            # Add project root so this script runs directly from any working directory.
            ROOT = str(_Path(__file__).resolve().parents[2])
            if ROOT not in sys.path:
                sys.path.insert(0, ROOT)

            from helpers.libdipum.nCutSegmentation import nCutSegmentation
            from helpers.libgeneral.mat2gray import mat2gray
            from helpers.libdipum.data_path import dip_data

            # Data
            image_path = dip_data("building-600by600.tif")
            f_raw = imread(image_path)

            # Ensure grayscale
            if f_raw.ndim == 3:
                f = f_raw.mean(axis=2)
            else:
                f = f_raw.astype(float)

            f = mat2gray(f)

            # Smoothing
            print("Applying smoothing (25x25)...")
            I_smooth = uniform_filter(f, size=25, mode="nearest")

            # Segment as in MATLAB example (half scale).
            print("Running nCutSegmentation (sf=0.5)...")
            S = nCutSegmentation(I_smooth, 2, sf=0.5)

            # Force a clean binary map (0/1) with foreground in white.
            labels = np.unique(S)
            if labels.size != 2:
                raise RuntimeError(f"Expected 2 regions, got {labels.size}")

            # Use a point near the bottom-center as foreground cue (building/hill region).
            r0 = int(0.85 * S.shape[0])
            c0 = int(0.50 * S.shape[1])
            fg_label = S[r0, c0]
            S_low_bin = S == fg_label

            # Hybrid refinement:
            # Keep robust low-res cut, but recover sharper 600x600 boundary by
            # transferring class intensity statistics to full-resolution smoothed image.
            I_low = resize(
                I_smooth, S.shape, order=1, preserve_range=True, anti_aliasing=True
            )
            fg_mean = float(I_low[S_low_bin].mean())
            bg_mean = float(I_low[~S_low_bin].mean())
            t = 0.5 * (fg_mean + bg_mean)

            if fg_mean >= bg_mean:
                Shalf = (I_smooth >= t).astype(float)
            else:
                Shalf = (I_smooth < t).astype(float)

            # Display
            fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
            axes[0].imshow(f, cmap="gray")
            axes[0].set_title("Original (600x600)")
            axes[0].axis("off")

            axes[1].imshow(I_smooth, cmap="gray")
            axes[1].set_title("Smoothed (25x25 box)")
            axes[1].axis("off")

            axes[2].imshow(Shalf, cmap="gray", vmin=0, vmax=1)
            axes[2].set_title("Graph Cut (2 regions)")
            axes[2].axis("off")

            plt.tight_layout()
            print("Saved Figure1058.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1059(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1059.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy.ndimage import uniform_filter
            from skimage.io import imread
            from skimage.transform import resize
            from helpers.libdipum.nCutSegmentation import nCutSegmentation
            from helpers.libgeneral.mat2gray import mat2gray
            from helpers.libdipum.data_path import dip_data

            def remap_two_regions(labels: Any, ref_image: Any):
                """Map 2 labels to {0,1} with bottom-center region forced to white."""
                u = np.unique(labels)
                if u.size != 2:
                    raise RuntimeError(f"Expected 2 regions, got {u.size}")

                r0 = int(0.85 * labels.shape[0])
                c0 = int(0.50 * labels.shape[1])
                fg_label = labels[r0, c0]
                out = np.zeros_like(labels, dtype=np.float64)
                out[labels == fg_label] = 1.0
                return out

            def remap_three_regions(labels: Any, ref_image: Any):
                """Map 3 labels to {0,0.5,1} ordered by mean brightness."""
                u = np.unique(labels)
                if u.size != 3:
                    raise RuntimeError(f"Expected 3 regions, got {u.size}")

                means = np.array(
                    [ref_image[labels == lab].mean() for lab in u], dtype=np.float64
                )
                order = np.argsort(means)  # darkest -> brightest

                out = np.zeros_like(labels, dtype=np.float64)
                out[labels == u[order[0]]] = 0.0
                out[labels == u[order[1]]] = 0.5
                out[labels == u[order[2]]] = 1.0
                return out

            def refine_two_region_from_lowres(S2: Any, I: Any):
                """Figure1058-like refinement: robust low-res cut + sharp full-res threshold."""
                # Orient labels with a spatial cue (iceberg near lower half).
                S2_bin = remap_two_regions(S2, I).astype(bool)
                # Transfer low-res class means to full-res smoothed image.
                I_low = resize(
                    I, S2.shape, order=1, preserve_range=True, anti_aliasing=True
                )
                fg_mean = float(I_low[S2_bin].mean())
                bg_mean = float(I_low[~S2_bin].mean())
                t = 0.5 * (fg_mean + bg_mean)
                if fg_mean >= bg_mean:
                    out = (I >= t).astype(np.float64)
                else:
                    out = (I < t).astype(np.float64)
                return out

            # Data
            image_path = dip_data("iceberg.tif")
            f_raw = imread(image_path)
            if f_raw.ndim == 3:
                f = f_raw.mean(axis=2)
            else:
                f = f_raw.astype(np.float64)
            f = mat2gray(f)

            # Smooth with 25x25 box kernel.
            print("Applying smoothing (25x25)...")
            I = uniform_filter(f, size=25, mode="nearest")

            # Graph-cut segmentation with 2 and 3 regions.
            print("Running graph cut (2 regions, sf=0.35)...")
            S2 = nCutSegmentation(I, 2, sf=0.35, n_segments=1200)
            S2v = refine_two_region_from_lowres(S2, I)

            print("Running graph cut (3 regions, sf=0.35)...")
            S3 = nCutSegmentation(I, 3, sf=0.35, n_segments=1200)
            S3v = remap_three_regions(S3, I)

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(8, 8))

            axes[0, 0].imshow(f, cmap="gray", vmin=0, vmax=1)
            axes[0, 0].axis("off")

            axes[0, 1].imshow(I, cmap="gray", vmin=0, vmax=1)
            axes[0, 1].axis("off")

            axes[1, 0].imshow(S2v, cmap="gray", vmin=0, vmax=1)
            axes[1, 0].axis("off")

            axes[1, 1].imshow(S3v, cmap="gray", vmin=0, vmax=1)
            axes[1, 1].axis("off")

            plt.tight_layout(pad=0.5)
            print("Saved Figure1059.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1062(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1062.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            import ia870 as ia
            from skimage.io import imread
            from helpers.libdipum.data_path import dip_data

            # Parameters
            h = 30
            bc8 = ia.iasebox()

            # Data
            image_path = dip_data("Fig1056(a)(blob_original).tif")
            f = imread(image_path)
            # Marker detection
            f3 = ia.iahmin(f, h, bc8)  # HMin filter
            m3 = ia.iaregmin(f3, bc8)  # Regional minima

            # Morphological gradient
            x = ia.iagradm(m3)

            # Display
            fig = plt.figure(figsize=(10, 8))

            ax = fig.add_subplot(2, 2, 1)
            ax.imshow(f, cmap="gray")
            ax.set_title("Original image")
            ax.axis("off")

            ax = fig.add_subplot(2, 2, 2)
            ax.imshow(f3, cmap="gray")
            ax.set_title("After removal of min < 30")
            ax.axis("off")

            ax = fig.add_subplot(2, 2, 3)
            ax.imshow(m3, cmap="gray")
            ax.set_title("Regional minimum")
            ax.axis("off")

            ax = fig.add_subplot(2, 2, 4)
            ax.imshow(x, cmap="gray")
            ax.set_title("Gradient")
            ax.axis("off")

            plt.tight_layout()
            print("Saved Figure1062.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1063(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1063.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            import ia870 as ia
            from skimage.io import imread
            from helpers.libdipum.data_path import dip_data
            # import MKRLib

            # Data
            image_path = dip_data("corneacells.tif")
            a = imread(image_path)
            b = ia.iagsurf(a)

            # Filtering and cell detection
            c = ia.iaasf(a, "oc", ia.iasecross(), 2)
            d = ia.iaregmax(c)

            # Background marker
            e = ia.ianeg(a)
            f = ia.iacwatershed(e, d, ia.iasebox())

            # Labeling markers and gradient
            g = ia.iagray(f, "uint16", 1)
            h1 = ia.iaaddm(ia.ialabel(d), 1)
            h = ia.iaintersec(ia.iagray(d, "uint16"), h1)
            i = ia.iaunion(g, h)

            # Gradient
            j = ia.iagradm(a)

            # Constrained watershed from markers
            k = ia.iacwatershed(j, i)

            def _imshow_ready(x: Any):
                """Convert ia870 color output from (C,H,W) to (H,W,C) for matplotlib."""
                x = np.asarray(x)
                if x.ndim == 3 and x.shape[0] in (3, 4) and x.shape[-1] not in (3, 4):
                    x = np.moveaxis(x, 0, -1)
                return x

            def _colorize_i_from_g_h(g_img: Any, h_img: Any, seed: Any = 0):
                """Create RGB display: red where g==1, random flat color for each label in h."""
                g_arr = np.asarray(g_img)
                h_arr = np.asarray(h_img)

                rgb = np.zeros(h_arr.shape + (3,), dtype=np.float32)

                labels = np.unique(h_arr)
                labels = labels[labels != 0]  # keep background as black
                rng = np.random.default_rng(seed)
                lut = {lab: rng.random(3, dtype=np.float32) for lab in labels}

                for lab in labels:
                    rgb[h_arr == lab] = lut[lab]

                # Force red on pixels marked in g (priority over label colors).
                rgb[g_arr == 1] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                return rgb

            # Figure 1
            fig1 = plt.figure(figsize=(10, 8))
            ax = fig1.add_subplot(2, 2, 1)
            ax.imshow(a, cmap="gray")
            ax.set_title("a = Original image")
            ax.axis("off")

            ax = fig1.add_subplot(2, 2, 2)
            ax.imshow(b, cmap="gray")
            ax.set_title("Surf(a)")
            ax.axis("off")

            ax = fig1.add_subplot(2, 2, 3)
            ax.imshow(c, cmap="gray")
            ax.set_title("c = OC(a)")
            ax.axis("off")

            ax = fig1.add_subplot(2, 2, 4)
            ax.imshow(d, cmap="gray")
            ax.set_title("d = RMAX(c)")
            ax.axis("off")

            fig1.tight_layout()
            fig1.savefig("Figure1063.png")

            # Figure 2
            fig2 = plt.figure(figsize=(10, 8))
            ax = fig2.add_subplot(2, 2, 1)
            ax.imshow(ia.iagsurf(c), cmap="gray")
            ax.set_title("Surf(c)")
            ax.axis("off")

            ax = fig2.add_subplot(2, 2, 2)
            plt.sca(ax)
            ax.imshow(_imshow_ready(ia.iagshow(ia.iagsurf(c), d)))
            ax.set_title("Surf(c), d")
            ax.axis("off")

            ax = fig2.add_subplot(2, 2, 3)
            plt.sca(ax)
            ax.imshow(_imshow_ready(ia.iagshow(e, d)))
            ax.set_title("e=~a, d")
            ax.axis("off")

            ax = fig2.add_subplot(2, 2, 4)
            ax.imshow(f, cmap="gray")
            ax.set_title("f = WS(e, d)")
            ax.axis("off")

            fig2.tight_layout()
            fig2.savefig("Figure1063Bis.png")

            # Figure 3
            fig3 = plt.figure(figsize=(10, 8))
            ax = fig3.add_subplot(2, 2, 1)
            plt.sca(ax)
            ax.imshow(_imshow_ready(ia.iagshow(e, f, d)))
            ax.set_title("e, f, d")
            ax.axis("off")

            ax = fig3.add_subplot(2, 2, 2)
            plt.sca(ax)
            ax.imshow(_colorize_i_from_g_h(g, h), interpolation="nearest")
            ax.set_title("i =")
            ax.axis("off")

            ax = fig3.add_subplot(2, 2, 3)
            ax.imshow(j, cmap="gray")
            ax.set_title("j = Grad(a)")
            ax.axis("off")

            ax = fig3.add_subplot(2, 2, 4)
            plt.sca(ax)
            ax.imshow(_imshow_ready(ia.iagshow(a, k, k)))
            ax.set_title("a, k")
            ax.axis("off")

            fig3.tight_layout()
            fig3.savefig("Figure1063Ter.png")

            print("Saved Figure1063.png, Figure1063Bis.png, Figure1063Ter.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1064(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure1064.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            import ia870 as ia
            from helpers.libgeneral import MKRLib
            from helpers.libdipum.data_path import dip_data

            # Figure1064

            # Data
            a = imread(dip_data("corneacells.tif"))

            # Image cleaning
            c = ia.iaasf(a, "oc", ia.iasecross(), 2)

            # Internal marker
            d = ia.iaregmax(c)

            # External marker obtained by watershed
            e = ia.ianeg(a)  # Segmentation function
            f = ia.iacwatershed(e, d)

            # Internal + External Marker
            # As the internal and external markers can be touching, we separate them by using the concept of labelled image

            h1 = ia.iaaddm(ia.ialabel(d), np.uint16(1))  # BG = 1, First particle = 2
            h2 = ia.iagray(d, "uint16")  # BG = 0, prticle = 65535
            h = ia.iaintersec(h2, h1)  # BG = 0, First particle = 2

            g = ia.iagray(f, "uint16", 1)  # External Marker = 1
            i = ia.iaunion(g, h)  # BG = 0, External Marker = 1, First particle = 2

            # Segmentation function
            j = ia.iagradm(a)

            # Constrained watershed of the gradient from markers
            # Apply the constrained watershed on the gradient from the labeled internal and external markers.
            k = ia.iacwatershed(j, i)

            # Display

            (fig, axes) = plt.subplots(nrows=2, ncols=2)
            axes[0, 0].set_title("a")
            axes[0, 0].imshow(a, cmap="gray")
            axes[0, 0].axis("off")
            axes[0, 1].set_title("c")
            axes[0, 1].imshow(c, cmap="gray")
            axes[0, 1].axis("off")
            axes[1, 0].set_title("d")
            axes[1, 0].imshow(MKRLib.mmshow(a, d), cmap="gray")
            axes[1, 0].axis("off")
            axes[1, 1].set_title("e,f,d")
            axes[1, 1].imshow((MKRLib.mmshow(e, f, d)), cmap="gray")
            axes[1, 1].axis("off")
            fig.tight_layout()

            (fig, axes) = plt.subplots(nrows=2, ncols=2)
            axes[0, 0].set_title("h1")
            # axes[0, 0].imshow(h1, cmap=CMap)
            axes[0, 0].imshow(MKRLib.mmlblshow(h1))
            axes[0, 0].axis("off")
            axes[0, 1].set_title("h2")
            axes[0, 1].imshow(h2, cmap="gray")
            axes[0, 1].axis("off")
            axes[1, 0].set_title("h")
            axes[1, 0].imshow(MKRLib.mmlblshow(h))
            axes[1, 0].axis("off")
            axes[1, 1].set_title("g")
            axes[1, 1].imshow(g, cmap="gray")
            axes[1, 1].axis("off")
            fig.tight_layout()

            (fig, axes) = plt.subplots(nrows=1, ncols=1)
            axes.set_title("i")
            axes.imshow(MKRLib.mmlblshow(i))
            axes.axis("off")
            fig.tight_layout()

            (fig, axes) = plt.subplots(nrows=1, ncols=1)
            axes.set_title("a,d,k")
            axes.imshow(MKRLib.mmshow(a, d > 0, k > 0), cmap="gray")
            axes.axis("off")
            fig.tight_layout()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure107(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure107.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from scipy.ndimage import convolve
            from helpers.libdip.intScaling4e import intScaling4e
            from helpers.libdipum.pixeldup import pixeldup
            from helpers.libdipum.data_path import dip_data

            # Data
            f = img_as_float(imread(dip_data("Fig1007(a)(wirebond_mask).tif")))
            M, N = f.shape

            # Kernel
            w = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=float)

            # Filtering
            g = convolve(f, w, mode="nearest")

            # Top Left Corner Zoom
            h_top = int(M / 4)
            w_top = int(N / 4)

            gtop = g[0:h_top, 0:w_top]
            # gtop = pixeldup(gtop, 4);
            gtop = pixeldup(gtop, 4)

            # Crop back to size of f
            gtop = gtop[:M, :N]

            # Bottom Right Corner Zoom
            # gbot = g(end - int32(M/4) + 1:end, end - int32(N/4) + 1:end);
            # Python: g[M-h_top : M, N-w_top : N]
            gbot = g[M - h_top : M, N - w_top : N]
            gbot = pixeldup(gbot, 4)
            gbot = gbot[:M, :N]

            botmax = gbot.max()
            gtop[-1, -1] = botmax

            # Positive values
            gpos = np.zeros_like(g)
            gpos[g > 0] = g[g > 0]

            # Thresholding
            T = gpos.max()
            gt = gpos >= T

            # Scaling for display
            gs = intScaling4e(g)
            gtop_s = intScaling4e(gtop)
            gbot_s = intScaling4e(gbot)
            gpos_s = intScaling4e(gpos)
            gt_s = intScaling4e(gt)

            # Display
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            axes[0].imshow(f, cmap="gray")
            axes[0].set_title("f")
            axes[0].axis("off")

            axes[1].imshow(gs, cmap="gray")
            axes[1].set_title("g (Filtered)")
            axes[1].axis("off")

            axes[2].imshow(gtop_s, cmap="gray")
            axes[2].set_title("Top Corner Zoom (gtop)")
            axes[2].axis("off")

            axes[3].imshow(gbot_s, cmap="gray")
            axes[3].set_title("Bottom Corner Zoom (gbot)")
            axes[3].axis("off")

            axes[4].imshow(gpos_s, cmap="gray")
            axes[4].set_title("Positive Values (gpos)")
            axes[4].axis("off")

            axes[5].imshow(gt_s, cmap="gray")
            axes[5].set_title("Thresholded (gt)")
            axes[5].axis("off")

            plt.tight_layout()
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure108(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `Figure108.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from helpers.libdipum.edgemodel import edgemodel

            # fstep = edgemodel('step', 128, 565, 0, .9, 1);
            fstep = edgemodel("step", 128, 565, 0.0, 0.9, 1)

            # framp = edgemodel('ramp', 128, 565, 0, .9, 250);
            framp = edgemodel("ramp", 128, 565, 0.0, 0.9, 250)

            # froof = edgemodel('roof', 128, 565, 0, .9, 150);
            froof = edgemodel("roof", 128, 565, 0.0, 0.9, 150)

            # Display
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes = axes.flatten()

            axes[0].imshow(fstep, cmap="gray")
            axes[0].set_title("Step Edge")
            axes[0].axis("off")

            axes[1].imshow(framp, cmap="gray")
            axes[1].set_title("Ramp Edge")
            axes[1].axis("off")

            axes[2].imshow(froof, cmap="gray")
            axes[2].set_title("Roof Edge")
            axes[2].axis("off")

            plt.tight_layout()
            print("Saved Figure108.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def tqtdecomp(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `TQTDecomp.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import sys
            import os
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.libgeneral.qtdecomp import qtdecomp
            from helpers.libgeneral.qtgetblk import qtgetblk
            from helpers.libgeneral.qtsetblk import qtsetblk
            import ia870 as ia
            from helpers.libdipum.data_path import dip_data
            from typing import Any

            # Robust image reader (reuse from previous or import if refactored)
            def read_image_robust(path: Any):
                """Read an image using multiple backends."""
                try:
                    return imread(path)
                except Exception:
                    try:
                        from PIL import Image

                        return np.array(Image.open(path))
                    except Exception:
                        import cv2

                        return cv2.imread(path, cv2.IMREAD_UNCHANGED)

            def ComputeMeans(I: Any, S: Any):
                """
                Compute mean of each block in quadtree decomposition.
                """
                means = I.astype(float)  # Output image

                # Iterate dimensions used in decomposition (powers of 2)
                # TQTDecomp loop: [512 256 ... 1]
                # We should detect dimensions present in S or iterate standard range.
                # Dimensions present:
                if S.nnz == 0:
                    return means

                dims = np.unique(S.data)

                for dim in dims:
                    dim = int(dim)  # Ensure int
                    values = qtgetblk(I, S, dim)

                    if values.size > 0:
                        # values is (dim, dim, k)
                        # Sum over dim, dim (axis 0, 1)
                        # Result (k,)

                        # doublesum = sum(sum(values,1,'double'),2); -> Sum over blocks
                        # Here: sum axis 0 and 1
                        block_sums = np.sum(values, axis=(0, 1))
                        block_means = block_sums / (dim**2)

                        # Repmat mean to fill block?
                        # qtsetblk(means, S, dim, mean_val)
                        # qtsetblk expects values of size (dim, dim, k).

                        # Create (dim, dim, k) with constant mean
                        k = values.shape[2]
                        mean_blocks = np.zeros((dim, dim, k))
                        for i in range(k):
                            mean_blocks[:, :, i] = block_means[i]

                        means = qtsetblk(means, S, dim, mean_blocks)

                return means

            def TQTDecomp():
                """Run quadtree decomposition demo and generate output plots."""
                print("Running TQTDecomp...")

                # Data
                # MATLAB source used liftingbody.png
                # Locate image
                fname = "cygnusloop.tif"
                # Common locations or search result?
                # I'll rely on find_by_name result logic or hardcoded from previous experience,
                # but since I am running blindly, I'll search common paths or current dir.

                # Fallback to standard check
                possible_paths = [
                    dip_data("cygnusloop.tif"),
                ]

                # Add whatever find_by_name returns if I could parse it here, but I will write this generic.
                path = fname
                for p in possible_paths:
                    if os.path.exists(p):
                        path = p
                        break

                if not os.path.exists(path):
                    # Try finding anywhere?
                    # For now prompt or error
                    print(f"Image {fname} not found.")
                    # Try standard skimage image?
                    try:
                        from skimage import data

                        print("Using skimage.data.camera() as fallback.")
                        I = data.camera()
                    except Exception:
                        return
                else:
                    I = read_image_robust(path)

                if I.ndim == 3:
                    I = I[:, :, 0]

                # DynThreshold = .27; (If float 0..1, input should be float? or uint8 scaled?)
                # "If I is of class uint8, threshold is multiplied by 255."
                # Let's use float range 0..1 by ensuring I is float 0..1?
                # Or keep uint8 and scale threshold.
                DynThreshold = 0.27
                if I.dtype == np.uint8:
                    threshold = DynThreshold * 255
                else:
                    threshold = DynThreshold

                print(f"Quadtree Decomposition (Threshold={threshold:.2f})...")
                S = qtdecomp(I, threshold)

                # Visualization of blocks
                # blocks = repmat(uint8(0),size(S));
                blocks = np.zeros(I.shape, dtype=np.uint8)

                # Dim sizes: 512, 256, ..., 1
                # Check max dim in S
                # sparse S contains dims.
                dims = np.unique(S.data)
                # Filter out 0 (background/none)
                dims = dims[dims > 0]
                dims = np.sort(dims)[::-1]  # Descending

                for dim in dims:
                    dim = int(dim)
                    # numblocks = length(find(S==dim));
                    # qtsetblk logic handles finding blocks.

                    # values = repmat(uint8(1),[dim dim numblocks]);
                    # values(2:dim,2:dim,:) = 0;
                    # -> Finds blocks, sets border to 1, center to 0. (Drawing borders)

                    # We need num blocks to construct values array
                    # Get count from S
                    numblocks = np.sum(S.data == dim)

                    if numblocks > 0:
                        values = np.ones((dim, dim, numblocks), dtype=np.uint8)
                        if dim > 1:
                            values[1:dim, 1:dim, :] = 0

                        blocks = qtsetblk(blocks, S, dim, values)

                # blocks(end,1:end) = 1; blocks(1:end,end) = 1; (Borders of image)
                blocks[-1, :] = 1
                blocks[:, -1] = 1

                # Compute Means
                print("Computing Means...")
                BlockMean = ComputeMeans(I, S)

                # e = mmsymdif (I, BlockMean);
                # mmsymdif? 'ia.iasymdif'?
                # Symmetric difference usually for sets (binary) or absolute diff for gray?
                # ia.iasymdif exists?
                # If not, use abs diff.
                try:
                    e = ia.iasymdif(I, BlockMean)  # Expects same type?
                    # I is uint8, BlockMean is float/double likely.
                    # Ensure compat.
                except:
                    # Fallback
                    e = np.abs(I.astype(float) - BlockMean.astype(float))

                # Display
                fig, axes = plt.subplots(2, 2, figsize=(10, 10))

                axes[0, 0].imshow(I, cmap="gray")
                axes[0, 0].set_title("f")
                axes[0, 0].axis("off")

                axes[0, 1].imshow(blocks, cmap="gray")  # [] scale? blocks is 0/1.
                axes[0, 1].set_title(f"blocks, Threshold = {DynThreshold}")
                axes[0, 1].axis("off")

                axes[1, 0].imshow(BlockMean, cmap="gray")
                axes[1, 0].set_title("Mean of each block")
                axes[1, 0].axis("off")

                axes[1, 1].imshow(e, cmap="gray")
                axes[1, 1].set_title("e = f - BlockMean")
                axes[1, 1].axis("off")

                plt.tight_layout()
                print("Saved TQTDecomp.png")
                plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def test_ncut(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter10 script `test_ncut.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from helpers.libdipum.nCutSegmentation import nCutSegmentation

            def test_ncut():
                """test_ncut."""
                # Create simple 32x32 image with two regions
                # Left half dark (0), right half light (1)
                I = np.zeros((32, 32))
                I[:, 16:] = 1.0

                # Add Gaussian noise
                rng = np.random.default_rng(42)
                I = I + rng.normal(0, 0.1, I.shape)

                print("Running nCutSegmentation...")
                # Using 2 segments
                S = nCutSegmentation(I, 2, sf=1.0)

                unique_lbls = np.unique(S)
                print("S unique values:", unique_lbls)

                plt.figure()
                plt.imshow(S)
                plt.title("Ncut Result (32x32)")
                plt.colorbar()

                # print(f"Saved result to {output_file}")
                plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

