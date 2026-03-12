from pathlib import Path as _Path
import os as _os

from typing import Any


class Chapter05Mixin:
    def figure510(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter05 script `Figure510.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from skimage.color import rgb2gray
            from helpers.libdipum.imnoise2 import imnoise2
            from helpers.libdipum.spfilt import spfilt
            from helpers.libdipum.data_path import dip_data

            # Parameters
            kernel_size = 3
            p_salt = 0.05
            p_pepper = 0.05

            # Data
            img_path = dip_data("circuitboard.tif")
            f_orig = imread(img_path)
            if f_orig.ndim == 3:
                f_orig = rgb2gray(f_orig)
            f = img_as_float(f_orig)

            # Noise adding
            fnSaltPepper, _ = imnoise2(f, "salt & pepper", p_salt, p_pepper)

            # Iterative median filtering (matches MATLAB: each iteration filters the same noisy image)
            fHat = []
            for _ in range(3):
                fHat.append(spfilt(fnSaltPepper, "median", kernel_size, kernel_size))

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            imshow_kwargs = dict(cmap="gray", vmin=0, vmax=1, interpolation="nearest")

            axes[0, 0].imshow(fnSaltPepper, **imshow_kwargs)
            axes[0, 0].set_title("Original image")
            axes[0, 0].axis("off")

            titles = [
                "median filter 1 pass",
                "median filter 2 passes",
                "median filter 3 passes",
            ]
            for idx in range(3):
                ax = axes.flat[idx + 1]
                ax.imshow(fHat[idx], **imshow_kwargs)
                ax.set_title(titles[idx])
                ax.axis("off")

            plt.tight_layout()
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure511(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter05 script `Figure511.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from skimage.color import rgb2gray
            from helpers.libdipum.imnoise2 import imnoise2
            from helpers.libdipum.spfilt import spfilt
            from helpers.libdipum.data_path import dip_data

            # Parameters
            kernel_size = 3
            p_salt = 0.05
            p_pepper = 0.05

            # Data
            img_path = dip_data("circuitboard.tif")
            f_orig = imread(img_path)
            if f_orig.ndim == 3:
                f_orig = rgb2gray(f_orig)
            f = img_as_float(f_orig)

            # Noise adding (computed but not used in MATLAB display)
            fnSaltPepper, _ = imnoise2(f, "salt & pepper", p_salt, p_pepper)

            # Min max filtering
            fMin = spfilt(f, "min", kernel_size, kernel_size)
            fMax = spfilt(f, "max", kernel_size, kernel_size)

            # Display
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            imshow_kwargs = dict(cmap="gray", vmin=0, vmax=1, interpolation="nearest")

            axes[0].imshow(fMax, **imshow_kwargs)
            axes[0].axis("off")
            axes[0].set_title("Max filter")

            axes[1].imshow(fMin, **imshow_kwargs)
            axes[1].axis("off")
            axes[1].set_title("Min filter")

            plt.tight_layout()
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure512(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter05 script `Figure512.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from skimage.color import rgb2gray
            from helpers.libdipum.imnoise2 import imnoise2
            from helpers.libdipum.spfilt import spfilt
            from helpers.libdipum.data_path import dip_data

            # Parameters
            kernel_size = 5
            mean = 0
            sigma = 0.4
            p_salt = 0.05
            p_pepper = 0.05

            # Data
            img_path = dip_data("circuitboard.tif")
            f_orig = imread(img_path)
            if f_orig.ndim == 3:
                f_orig = rgb2gray(f_orig)
            f = img_as_float(f_orig)

            # Add noise
            fnUniform, _ = imnoise2(f, "uniform", mean, sigma)
            fnSaltPepper, _ = imnoise2(f, "salt & pepper", p_salt, p_pepper)

            # Filtering
            fnSaltPepperAMean = spfilt(fnSaltPepper, "amean", kernel_size, kernel_size)
            fnSaltPepperGMean = spfilt(fnSaltPepper, "gmean", kernel_size, kernel_size)
            fnSaltPepperMedian = spfilt(
                fnSaltPepper, "median", kernel_size, kernel_size
            )
            fnSaltPepperAlphaTrimmed = spfilt(
                fnSaltPepper, "atrimmed", kernel_size, kernel_size, 6
            )

            # Display
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            imshow_kwargs = dict(cmap="gray", vmin=0, vmax=1, interpolation="nearest")

            axes[0, 0].imshow(fnUniform, **imshow_kwargs)
            axes[0, 0].set_title("Uniform noise")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(fnSaltPepper, **imshow_kwargs)
            axes[0, 1].set_title("fn = Salt & Pepper noise")
            axes[0, 1].axis("off")

            axes[0, 2].imshow(fnSaltPepperAMean, **imshow_kwargs)
            axes[0, 2].set_title("Arithmetic mean (fn)")
            axes[0, 2].axis("off")

            axes[1, 0].imshow(fnSaltPepperGMean, **imshow_kwargs)
            axes[1, 0].set_title("Geometric mean (fn)")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(fnSaltPepperMedian, **imshow_kwargs)
            axes[1, 1].set_title("Median (fn)")
            axes[1, 1].axis("off")

            axes[1, 2].imshow(fnSaltPepperAlphaTrimmed, **imshow_kwargs)
            axes[1, 2].set_title("Alpha trimmed mean (fn)")
            axes[1, 2].axis("off")

            plt.tight_layout()
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure513(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter05 script `Figure513.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from skimage.color import rgb2gray
            from scipy.ndimage import uniform_filter
            from helpers.libdipum.imnoise2 import imnoise2
            from helpers.libdipum.spfilt import spfilt
            from helpers.libdipum.data_path import dip_data

            def adaptive_noise_reduction_filter(f: Any, w: Any, noise_var: Any):
                """
                Adaptive noise reduction filter (MATLAB-equivalent).
                w: half-window size (window is (2*w+1)x(2*w+1))
                noise_var: noise variance
                """
                f = img_as_float(f)
                size = 2 * w + 1

                # Local mean and variance (population variance, like MATLAB var(...,1))
                local_mean = uniform_filter(f, size=size, mode="reflect")
                local_mean_sq = uniform_filter(f * f, size=size, mode="reflect")
                local_var = local_mean_sq - local_mean**2

                # Apply adaptive formula
                with np.errstate(divide="ignore", invalid="ignore"):
                    g = f - (noise_var / local_var) * (f - local_mean)

                # Where local variance is zero, keep original pixel
                g = np.where(local_var > 0, g, f)
                return g

            # Parameters
            kernel_size = 7
            mean = 0
            sigma = 0.1

            # Data
            img_path = dip_data("circuitboard.tif")
            f_orig = imread(img_path)
            if f_orig.ndim == 3:
                f_orig = rgb2gray(f_orig)
            f = img_as_float(f_orig)

            # Add noise
            fnGaussian, _ = imnoise2(f, "gaussian", mean, sigma)

            # Filtering
            fnGaussianAMean = spfilt(fnGaussian, "amean", kernel_size, kernel_size)
            fnGaussianGMean = spfilt(fnGaussian, "gmean", kernel_size, kernel_size)
            fnGaussianAdaptiveFilter = adaptive_noise_reduction_filter(
                fnGaussian, 3, sigma**2
            )

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            imshow_kwargs = dict(cmap="gray", vmin=0, vmax=1, interpolation="nearest")

            axes[0, 0].imshow(fnGaussian, **imshow_kwargs)
            axes[0, 0].set_title("Gaussian noise")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(fnGaussianAMean, **imshow_kwargs)
            axes[0, 1].set_title("Arithmetic mean")
            axes[0, 1].axis("off")

            axes[1, 0].imshow(fnGaussianGMean, **imshow_kwargs)
            axes[1, 0].set_title("Geometric mean")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(fnGaussianAdaptiveFilter, **imshow_kwargs)
            axes[1, 1].set_title("Adaptive Filter (fn)")
            axes[1, 1].axis("off")

            plt.tight_layout()
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure514(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter05 script `Figure514.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from helpers.libdipum.imnoise2 import imnoise2
            from helpers.libdipum.spfilt import spfilt
            from helpers.libdipum.adpmedian import adpmedian
            from helpers.libdipum.data_path import dip_data

            # Parameters
            KernelSize = 5
            PSalt = 0.25
            PPepper = 0.25
            SMax = 7

            # Data
            img_path = dip_data("circuitboard.tif")
            f = plt.imread(img_path)
            # Convert RGB → grayscale if needed
            if f.ndim == 3:
                f = f[..., 0]
            # im2double equivalent
            f = f.astype(np.float64)
            if f.max() > 1.0:
                f /= 255.0

            # Add noise
            fn, R = imnoise2(f, "salt & pepper", PSalt, PPepper)

            # Filtering
            fHatMedian = spfilt(fn, "median", KernelSize, KernelSize)
            fHatAdaptiveMedian = adpmedian(fn, SMax)

            # Display
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(fn, cmap="gray")
            plt.title("Salt & Pepper noise")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(fHatMedian, cmap="gray")
            plt.title("Median")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(fHatAdaptiveMedian, cmap="gray")
            plt.title("Adaptive median")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

            # ------------------------------------------------------------
            # Save figure
            # ------------------------------------------------------------
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(fn, cmap="gray")
            plt.title("Salt & Pepper noise")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(fHatMedian, cmap="gray")
            plt.title("Median")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(fHatAdaptiveMedian, cmap="gray")
            plt.title("Adaptive median")
            plt.axis("off")

            plt.tight_layout()
            plt.close()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure515(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter05 script `Figure515.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from helpers.libdipum.cnotch import cnotch

            def plot_mesh(ax: Any, H: Any, title: Any):
                """plot_mesh."""
                # Shift for display
                H_disp = np.fft.fftshift(H)
                rows, cols = H_disp.shape

                # Meshgrid for plotting
                X = np.arange(cols)
                Y = np.arange(rows)
                X, Y = np.meshgrid(X, Y)

                ax.plot_wireframe(
                    X, Y, H_disp, color="black", rstride=1, cstride=1, linewidth=0.7
                )

                # Setup axis
                ax.set_title(title)
                ax.set_axis_off()

                # View point? Matlab default.
                ax.view_init(elev=30, azim=-60)  # Default-ish

            # Transfer functions
            # Hideal = double(cnotch('ideal','reject',40,40,[28 11],3));
            Hideal = cnotch("ideal", "reject", 40, 40, [28, 11], 3)

            # Hgauss = double(cnotch('gaussian','reject',40,40,[28 11],3));
            Hgauss = cnotch("gaussian", "reject", 40, 40, [28, 11], 3)

            # Hbw = double(cnotch('butterworth','reject',40,40,[28 11],3,2));
            Hbw = cnotch("butterworth", "reject", 40, 40, [28, 11], 3, n=2)

            # Display
            fig = plt.figure(figsize=(15, 6))

            # Plot 1
            ax1 = fig.add_subplot(1, 3, 1, projection="3d")
            plot_mesh(ax1, Hideal, "Ideal Notch")

            # Plot 2
            ax2 = fig.add_subplot(1, 3, 2, projection="3d")
            plot_mesh(ax2, Hgauss, "Gaussian Notch")

            # Plot 3
            ax3 = fig.add_subplot(1, 3, 3, projection="3d")
            plot_mesh(ax3, Hbw, "Butterworth Notch")

            plt.tight_layout()
            print("Saved Figure515.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure516(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter05 script `Figure516.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from helpers.libdipum.cnotch import cnotch
            from helpers.libdipum.imnoise3 import imnoise3
            from helpers.libdip.intScaling4e import intScaling4e
            from helpers.libdipum.data_path import dip_data

            # Data
            img_name = dip_data("astronaut.tif")
            f_orig = imread(img_name)
            if f_orig.ndim == 3:
                f_orig = f_orig[:, :, 0]  # Use one channel if color
            f = img_as_float(f_orig)

            M, N = f.shape

            # Generate sinusoidal noise pattern
            # [r, R, S] = imnoise3(M, N, [25 25], 0.3);
            # Note: MATLAB [25 25] means C = [25 25].
            # 0.3 is Amplitude A.
            r, R, S = imnoise3(M, N, [[25, 25]], A=[0.3])

            # Noisy image
            g = f + r

            # Scaling for display implies mapping range.
            # intScaling4e handles this.
            gs = intScaling4e(g)

            # Spectrum
            G_complex = np.fft.fft2(g)
            G = np.fft.fftshift(np.abs(G_complex))
            Glog = intScaling4e(1 + np.log(G + 1e-9))  # Avoid log(0)

            # Create notch filters
            impulse_loc = [M // 2 + 25, N // 2 + 25]
            H = cnotch("ideal", "reject", M, N, [impulse_loc], 2)

            # Hc = intScaling4e(fftshift(H));
            # cnotch returns uncentered.
            Hc = intScaling4e(np.fft.fftshift(H))

            # Filter the image
            F_g = np.fft.fft2(g)
            G_filtered = F_g * H
            gf_raw = np.real(np.fft.ifft2(G_filtered))
            gf = intScaling4e(gf_raw)

            # Eliminated noise (what filtering removed)
            noise_eliminated = g - gf_raw
            noise_eliminated_s = intScaling4e(noise_eliminated)

            # Display results
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            axes[0, 0].imshow(gs, cmap="gray")  # Display scaled noisy image
            axes[0, 0].set_title("Noisy Image")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(Glog, cmap="gray")
            axes[0, 1].set_title("Spectrum of Noisy Image")
            axes[0, 1].axis("off")

            axes[1, 0].imshow(Hc, cmap="gray")
            axes[1, 0].set_title("Notch Filter")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(gf, cmap="gray")
            axes[1, 1].set_title("Filtered Image")
            axes[1, 1].axis("off")

            plt.tight_layout()
            print("Saved Figure516.png")

            # Display eliminated noise in separate figure
            plt.figure(figsize=(6, 6))
            plt.imshow(noise_eliminated_s, cmap="gray")
            plt.title("Eliminated Noise")
            plt.axis("off")
            plt.tight_layout()
            print("Saved Figure517.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure518(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter05 script `Figure518.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from helpers.libdipum.recnotch import recnotch
            from helpers.libdip.intScaling4e import intScaling4e
            from helpers.libdipum.data_path import dip_data

            # Data
            img_name = dip_data("satellite_original.tif")
            f_orig = imread(img_name)
            if f_orig.ndim == 3:
                f_orig = f_orig[:, :, 0]
            f = img_as_float(f_orig)

            M, N = f.shape

            # DFT
            F = np.fft.fft2(f)

            S = intScaling4e(np.log10(1 + np.abs(np.fft.fftshift(F))))

            # Notch filter design
            HR = recnotch("reject", "vertical", M, N, 19, 30)

            # Filtering in the Fourier domain
            G = F * HR
            g = np.real(np.fft.ifft2(G))

            # Obtain interference pattern using a bandpass filter
            HP = 1 - HR
            GP = F * HP
            gp = intScaling4e(np.real(np.fft.ifft2(GP)))

            # Spectrum times notch filter
            SF = S * np.fft.fftshift(HR)

            # Display (5 plots in Figure518)
            fig = plt.figure(figsize=(15, 10))

            ax1 = fig.add_subplot(2, 3, 1)
            ax1.imshow(f, cmap="gray")
            ax1.set_title("Original")
            ax1.axis("off")

            ax2 = fig.add_subplot(2, 3, 2)
            ax2.imshow(S, cmap="gray")
            ax2.set_title("Spectrum of the original")
            ax2.axis("off")

            ax3 = fig.add_subplot(2, 3, 3)
            ax3.imshow(np.fft.fftshift(HR), cmap="gray")
            ax3.set_title("Notch filter")
            ax3.axis("off")

            ax4 = fig.add_subplot(2, 3, 4)
            ax4.imshow(SF, cmap="gray")
            ax4.set_title("Spectrum * Notch")
            ax4.axis("off")

            ax5 = fig.add_subplot(2, 3, 5)
            ax5.imshow(g, cmap="gray")
            ax5.set_title("Recovered")
            ax5.axis("off")

            # Leave subplot (2,3,6) empty intentionally
            ax6 = fig.add_subplot(2, 3, 6)
            ax6.axis("off")

            plt.tight_layout()
            print("Saved Figure518.png")

            # Separate figure: interference
            plt.figure(figsize=(6, 6))
            plt.imshow(gp, cmap="gray")
            plt.title("Interference")
            plt.axis("off")
            plt.tight_layout()
            print("Saved Figure519.png")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure525(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter05 script `Figure525.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from skimage.color import rgb2gray
            from helpers.libdip.dftFiltering4e import dftFiltering4e
            from helpers.libgeneral.atmosphturb import atmosphturb
            from helpers.libdipum.data_path import dip_data

            # Parameters
            k_vals = [0.0025, 0.001, 0.00025]

            # Data
            img_path = dip_data("aerial_view_no_turb.tif")
            f_orig = imread(img_path)
            if f_orig.ndim == 3:
                f_orig = rgb2gray(f_orig)
            f = img_as_float(f_orig)

            M, N = f.shape

            # Atmospheric perturbations
            H = []
            for k in k_vals:
                H.append(atmosphturb(2 * M, 2 * N, k))

            # Filtering
            g = []
            for idx in range(len(k_vals)):
                g.append(dftFiltering4e(f, H[idx]))

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            imshow_kwargs = dict(cmap="gray", vmin=0, vmax=1, interpolation="nearest")

            axes[0, 0].imshow(f, **imshow_kwargs)
            axes[0, 0].set_title("Original")
            axes[0, 0].axis("off")

            for i, k in enumerate(k_vals):
                ax = axes.flat[i + 1]
                ax.imshow(g[i], **imshow_kwargs)
                ax.set_title(f"k = {k}")
                ax.axis("off")

            plt.tight_layout()
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure526(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter05 script `Figure526.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from helpers.libdip.motionBlurTF4e import motionBlurTF4e
            from helpers.libdipum.data_path import dip_data

            img_name = dip_data("original_DIP.tif")
            f_orig = imread(img_name)
            if f_orig.ndim == 3:
                f_orig = f_orig[:, :, 0]
            f = img_as_float(f_orig)

            M, N = f.shape

            # Generate blurring filter
            # H = fftshift (motionBlurTF4e (M, N, 0.1, 0.1, 1));
            # motionBlurTF4e returns centered H.
            # fftshift creates standard uncentered FFT format (DC at corner).
            H_centered = motionBlurTF4e(M, N, 0.1, 0.1, 1)
            H = np.fft.fftshift(H_centered)

            # Get FFT of image
            F = np.fft.fft2(f)

            # Multiply
            G = F * H

            # Output
            # g = real(ifft2(G));
            g = np.real(np.fft.ifft2(G))

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            axes[0, 0].imshow(f, cmap="gray")
            axes[0, 0].set_title("Original Image")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(g, cmap="gray")
            axes[0, 1].set_title("Blurred Image")
            axes[0, 1].axis("off")

            axes[1, 0].imshow(np.abs(H), cmap="gray")
            axes[1, 0].set_title("Filter Magnitude |H|")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(np.angle(H), cmap="gray")
            axes[1, 1].set_title("Filter Phase angle(H)")
            axes[1, 1].axis("off")

            plt.tight_layout()
            print("Saved Figure526.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure527(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter05 script `Figure527.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from skimage.color import rgb2gray
            from helpers.libgeneral.atmosphturb import atmosphturb
            from helpers.libdip.lpFilterTF4e import lpFilterTF4e
            from helpers.libdipum.imnoise2 import imnoise2
            from helpers.libdipum.data_path import dip_data

            # Parameters
            k = 0.0025
            mu = 0
            sigma = 10 ** (-10)
            D0 = [40, 70, 85]
            order = 10

            # Data
            img_path = dip_data("aerial_view_no_turb.tif")
            f_orig = imread(img_path)
            if f_orig.ndim == 3:
                f_orig = rgb2gray(f_orig)
            f = img_as_float(f_orig)

            M, N = f.shape

            # Fourier transform (shifted)
            F = np.fft.fftshift(np.fft.fft2(f))

            # Atmospheric perturbations
            H = atmosphturb(M, N, k)

            # Filtering in the frequency domain
            G = H * F

            # Add noise in the frequency domain
            z = np.zeros((M, N))
            zn, _ = imnoise2(z, "gaussian", mu, sigma)
            Gn = np.fft.fftshift(np.fft.fft2(zn))
            G1 = G + Gn

            # Straight inverse filter
            Fh1 = G1 / H
            fHat = [np.abs(np.real(np.fft.ifft2(Fh1)))]

            # Cut off range of values of Fh1
            Temp = []
            for d0 in D0:
                LowPass = lpFilterTF4e("butterworth", M, N, d0, order)
                Temp.append(LowPass * Fh1)
                fHat.append(np.abs(np.real(np.fft.ifft2(Temp[-1]))))

            # Display 1
            fig1, axes1 = plt.subplots(2, 3, figsize=(12, 8))
            axes1[0, 0].imshow(f, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axes1[0, 0].set_title("f")
            axes1[0, 0].axis("off")

            axes1[0, 1].imshow(np.log10(1 + np.abs(F)), cmap="gray")
            axes1[0, 1].set_title("log(1+|F|)")
            axes1[0, 1].axis("off")

            axes1[0, 2].imshow(H, cmap="gray")
            axes1[0, 2].set_title("|H|")
            axes1[0, 2].axis("off")

            axes1[1, 0].imshow(np.log10(1 + np.abs(G)), cmap="gray")
            axes1[1, 0].set_title("log(1+|G|, G = H*F)")
            axes1[1, 0].axis("off")

            axes1[1, 1].imshow(np.log10(1 + np.abs(G1)), cmap="gray")
            axes1[1, 1].set_title("log(1+|G1|, = G + G_n)")
            axes1[1, 1].axis("off")

            axes1[1, 2].axis("off")

            plt.tight_layout()

            # Display 2
            fig2, axes2 = plt.subplots(2, 2, figsize=(10, 10))
            axes2[0, 0].imshow(np.log10(1 + np.abs(Fh1)), cmap="gray")
            axes2[0, 0].set_title("log(1+|Fh1|, = Fh1 = G1 / H)")
            axes2[0, 0].axis("off")

            for i, d0 in enumerate(D0):
                ax = axes2.flat[i + 1]
                ax.imshow(np.log10(1 + np.abs(Temp[i])), cmap="gray")
                ax.set_title("log(1+|Fh1|, = Fh1 = (G1 / H) * LowPass)")
                ax.axis("off")

            plt.tight_layout()

            # Display 3
            fig3, axes3 = plt.subplots(2, 2, figsize=(10, 10))
            axes3[0, 0].imshow(fHat[0], cmap="gray")
            axes3[0, 0].set_title("f_hat = ifft2 (Fh1)")
            axes3[0, 0].axis("off")

            for i, d0 in enumerate(D0):
                ax = axes3.flat[i + 1]
                ax.imshow(fHat[i + 1], cmap="gray")
                ax.set_title(f"f_hat = ifft2 (Fh1 * LowPass(D0, N)), D0 = {d0}")
                ax.axis("off")

            plt.tight_layout()
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure528(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter05 script `Figure528.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from skimage.color import rgb2gray
            from helpers.libgeneral.atmosphturb import atmosphturb
            from helpers.libdip.lpFilterTF4e import lpFilterTF4e
            from helpers.libdipum.imnoise2 import imnoise2
            from helpers.libgeneral.deconvwnr import deconvwnr
            from helpers.libdipum.data_path import dip_data

            # Parameters
            k = 0.0025
            mu = 0
            sigma = 1e-10
            D0 = 85
            order = 10

            # Data
            img_path = dip_data("aerial_view_no_turb.tif")
            f_orig = imread(img_path)
            if f_orig.ndim == 3:
                f_orig = rgb2gray(f_orig)
            f = img_as_float(f_orig)

            M, N = f.shape

            # Fourier transform
            F = np.fft.fftshift(np.fft.fft2(f))

            # Atmospheric perturbations
            H = atmosphturb(M, N, k)

            # Filtering in the frequency domain
            G = H * F
            g = np.fft.ifft2(np.fft.fftshift(G))

            # Add noise in the frequency domain
            z = np.zeros((M, N))
            zn, _ = imnoise2(z, "gaussian", mu, sigma)
            Gn = np.fft.fftshift(np.fft.fft2(zn))
            G1 = G + Gn
            g1 = np.fft.ifft2(np.fft.fftshift(G1))

            # Straight inverse filter
            Fh1 = G1 / H
            fHat = []
            fHat.append(np.abs(np.real(np.fft.ifft2(Fh1))))

            # Low-pass cutoff
            LowPass = lpFilterTF4e("butterworth", M, N, D0, order)
            Temp = LowPass * Fh1
            fHat.append(np.abs(np.real(np.fft.ifft2(Temp))))

            # Wiener
            nspr = (sigma**2) / np.var(f)
            psf = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(H)))
            fHat.append(deconvwnr(g1, psf, nspr))

            # Display
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for i in range(3):
                axes[i].imshow(fHat[i], cmap="gray")
                axes[i].axis("off")

            plt.tight_layout()
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure529(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter05 script `Figure529.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float, random_noise
            from helpers.libdip.motionBlurTF4e import motionBlurTF4e
            from helpers.libdip.pWienerTF4e import pWienerTF4e
            from helpers.libdipum.data_path import dip_data

            # Parameters
            VarNoise = [1e-37, 1e-2, 1e-1]
            VarWiener = [1e-35, 0.15e-1, 0.2e-1]

            # Data
            img_name = dip_data("original_DIP.tif")
            f_orig = imread(img_name)
            if f_orig.ndim == 3:
                f_orig = f_orig[:, :, 0]
            f = img_as_float(f_orig)

            M, N = f.shape
            F = np.fft.fft2(f)

            # Blur filter
            H_centered = motionBlurTF4e(M, N, 0.1, 0.1, 1)
            H = np.fft.fftshift(H_centered)

            # Filtering in Fourier domain
            G = F * H
            g = np.real(np.fft.ifft2(G))

            # Added noise
            z = np.zeros((M, N))

            # Generate noise with clip=True to match MATLAB imnoise behavior on zeros
            zn1 = random_noise(z, mode="gaussian", mean=0, var=VarNoise[0], clip=True)
            zn2 = random_noise(z, mode="gaussian", mean=0, var=VarNoise[1], clip=True)
            zn3 = random_noise(z, mode="gaussian", mean=0, var=VarNoise[2], clip=True)

            Zn1 = np.fft.fft2(zn1)
            Zn2 = np.fft.fft2(zn2)
            Zn3 = np.fft.fft2(zn3)

            Gn1 = G + Zn1
            Gn2 = G + Zn2
            Gn3 = G + Zn3

            gn1 = np.real(np.fft.ifft2(Gn1))
            gn2 = np.real(np.fft.ifft2(Gn2))
            gn3 = np.real(np.fft.ifft2(Gn3))

            # Performs filtering
            # Inverse using pWienerTF4e(H, 0)
            W0 = pWienerTF4e(H, 0)

            fHatInvFilter1 = np.real(np.fft.ifft2(W0 * Gn1))
            fHatInvFilter2 = np.real(np.fft.ifft2(W0 * Gn2))
            fHatInvFilter3 = np.real(np.fft.ifft2(W0 * Gn3))

            # Wiener
            W1 = pWienerTF4e(H, VarWiener[0])
            W2 = pWienerTF4e(H, VarWiener[1])
            W3 = pWienerTF4e(H, VarWiener[2])

            fHatWienerFilter1 = np.real(np.fft.ifft2(W1 * Gn1))
            fHatWienerFilter2 = np.real(np.fft.ifft2(W2 * Gn2))
            fHatWienerFilter3 = np.real(np.fft.ifft2(W3 * Gn3))

            # Display
            # Use vmin=0, vmax=1 to match MATLAB imshow(double) behavior

            fig1, axes1 = plt.subplots(1, 2, figsize=(10, 5))
            axes1[0].imshow(f, cmap="gray", vmin=0, vmax=1)
            axes1[0].set_title("Original")
            axes1[0].axis("off")
            axes1[1].imshow(g, cmap="gray", vmin=0, vmax=1)
            axes1[1].set_title("Blurred")
            axes1[1].axis("off")
            plt.tight_layout()

            fig2, axes2 = plt.subplots(3, 3, figsize=(12, 12))

            # Row 1 High
            axes2[0, 0].imshow(gn3, cmap="gray", vmin=0, vmax=1)
            axes2[0, 0].set_title("Blurred + Noise (High)")
            axes2[0, 0].axis("off")
            axes2[0, 1].imshow(fHatInvFilter3, cmap="gray", vmin=0, vmax=1)
            axes2[0, 1].set_title("Inverse Filter")
            axes2[0, 1].axis("off")
            axes2[0, 2].imshow(fHatWienerFilter3, cmap="gray", vmin=0, vmax=1)
            axes2[0, 2].set_title("Wiener Filter")
            axes2[0, 2].axis("off")

            # Row 2 Med
            axes2[1, 0].imshow(gn2, cmap="gray", vmin=0, vmax=1)
            axes2[1, 0].set_title("Blurred + Noise (Med)")
            axes2[1, 0].axis("off")
            axes2[1, 1].imshow(fHatInvFilter2, cmap="gray", vmin=0, vmax=1)
            axes2[1, 1].set_title("Inverse Filter")
            axes2[1, 1].axis("off")
            axes2[1, 2].imshow(fHatWienerFilter2, cmap="gray", vmin=0, vmax=1)
            axes2[1, 2].set_title("Wiener Filter")
            axes2[1, 2].axis("off")

            # Row 3 Low
            axes2[2, 0].imshow(gn1, cmap="gray", vmin=0, vmax=1)
            axes2[2, 0].set_title("Blurred + Noise (Low)")
            axes2[2, 0].axis("off")
            axes2[2, 1].imshow(fHatInvFilter1, cmap="gray", vmin=0, vmax=1)
            axes2[2, 1].set_title("Inverse Filter")
            axes2[2, 1].axis("off")
            axes2[2, 2].imshow(fHatWienerFilter1, cmap="gray", vmin=0, vmax=1)
            axes2[2, 2].set_title("Wiener Filter")
            axes2[2, 2].axis("off")

            plt.tight_layout()
            print("Saved Figure529_init.png and Figure529.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure53(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter05 script `Figure53.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from helpers.libdipum.data_path import dip_data

            # %% Figure 5.3

            # Data
            f = imread(dip_data("test-pattern.tif"))
            if f.ndim == 3:
                f = f[:, :, 0]

            # MATLAB: f = im2double(f)
            f = img_as_float(f)

            # MATLAB: f = f(1:2:end, 1:2:end)
            f = f[0::2, 0::2]
            M, N = f.shape
            print(f"Size after subsampling: M={M}, N={N}")

            # Display
            plt.figure(figsize=(6, 6))
            plt.imshow(f, cmap="gray", vmin=0, vmax=1)
            plt.axis("off")
            plt.tight_layout()

            # Print to file
            print("Saved Figure53.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure530(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter05 script `Figure530.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from skimage.color import rgb2gray
            from helpers.libdip.motionBlurTF4e import motionBlurTF4e
            from helpers.libdip.constrainedLsTF4e import constrainedLsTF4e
            from helpers.libdipum.data_path import dip_data

            # Parameters
            var_noise = [10 ** (-37), 10 ** (-2), 10 ** (-1)]
            gamma = [10 ** (-35), 0.015, 1.5]

            # Data
            img_path = dip_data("original_DIP.tif")
            f_orig = imread(img_path)
            if f_orig.ndim == 3:
                f_orig = rgb2gray(f_orig)
            f = img_as_float(f_orig)

            M, N = f.shape

            # Fourier transform
            F = np.fft.fftshift(np.fft.fft2(f))

            # Blur filter (centered)
            H = motionBlurTF4e(M, N, 0.1, 0.1, 1)

            # Filtering in the Fourier domain
            G = F * H
            g = np.real(np.fft.ifft2(G))

            # Added noise (3 different variances)
            z = np.zeros((M, N))
            zn1 = np.random.normal(0.0, np.sqrt(var_noise[0]), size=(M, N))
            zn2 = np.random.normal(0.0, np.sqrt(var_noise[1]), size=(M, N))
            zn3 = np.random.normal(0.0, np.sqrt(var_noise[2]), size=(M, N))

            Zn1 = np.fft.fft2(zn1)
            Zn2 = np.fft.fft2(zn2)
            Zn3 = np.fft.fft2(zn3)

            Gn1 = G + Zn1
            Gn2 = G + Zn2
            Gn3 = G + Zn3

            # Case 1. High noise, high Gamma
            L = constrainedLsTF4e(H, gamma[2])
            Fh = L * Gn3
            fHatHighNoiseHighGamma = np.abs(np.real(np.fft.ifft2(Fh)))

            # Case 2. Medium noise, Medium Gamma
            L = constrainedLsTF4e(H, gamma[1])
            Fh = L * Gn2
            fHatMediumNoiseMediumGamma = np.abs(np.real(np.fft.ifft2(Fh)))

            # Case 3. Low noise, low Gamma
            L = constrainedLsTF4e(H, gamma[0])
            Fh = L * Gn1
            fHatLowNoiseLowGamma = np.abs(np.real(np.fft.ifft2(Fh)))

            # Display
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(fHatHighNoiseHighGamma, cmap="gray")
            axes[0].axis("off")

            axes[1].imshow(fHatMediumNoiseMediumGamma, cmap="gray")
            axes[1].axis("off")

            axes[2].imshow(fHatLowNoiseLowGamma, cmap="gray")
            axes[2].axis("off")

            plt.tight_layout()
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure531(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter05 script `Figure531.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from skimage.color import rgb2gray
            from helpers.libgeneral.atmosphturb import atmosphturb
            from helpers.libdipum.imnoise2 import imnoise2
            from helpers.libdip.constrainedLsTF4e import constrainedLsTF4e
            from helpers.libgeneral.deconvreg1 import deconvreg1
            from helpers.libdipum.data_path import dip_data

            # Parameters
            k = 0.0025
            mu = 0
            sigma = 10e-3
            gamma = 5e-5

            # Data
            img_path = dip_data("aerial_view_no_turb.tif")
            f_orig = imread(img_path)
            if f_orig.ndim == 3:
                f_orig = rgb2gray(f_orig)
            f = img_as_float(f_orig)

            M, N = f.shape

            # Fourier transform
            F = np.fft.fftshift(np.fft.fft2(f))

            # Atmospheric perturbations
            H = atmosphturb(M, N, k)
            h = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(H)))

            # Filtering in the frequency domain
            G = H * F
            g = np.fft.ifft2(np.fft.fftshift(G))

            # Add noise in the frequency domain
            z = np.zeros((M, N))
            zn, _ = imnoise2(z, "gaussian", mu, sigma)
            Gn = np.fft.fftshift(np.fft.fft2(zn))
            G1 = G + Gn
            g1 = np.fft.ifft2(np.fft.fftshift(G1))

            # Restoration by using constrained least square (fixed gamma)
            L = constrainedLsTF4e(H, gamma)
            Fh = L * G1
            fHat = np.fft.ifft2(Fh)

            L1 = constrainedLsTF4e(H, gamma / 10.0)
            Fh1 = L1 * G1
            fHat1 = np.fft.ifft2(Fh1)

            # Restoration by using constrained least square (regularized), iterative via fminbnd
            fHat2, Lagra = deconvreg1(np.real(g1), np.real(h))

            # Display
            fig1, axes1 = plt.subplots(2, 2, figsize=(10, 10))
            axes1[0, 0].imshow(np.abs(H), cmap="gray")
            axes1[0, 0].set_title("Blurring transfer function")
            axes1[0, 0].axis("off")

            axes1[0, 1].imshow(np.abs(h), cmap="gray")
            axes1[0, 1].set_title("Point spread function")
            axes1[0, 1].axis("off")

            axes1[1, 0].imshow(f, cmap="gray")
            axes1[1, 0].set_title("Original image")
            axes1[1, 0].axis("off")

            axes1[1, 1].imshow(np.abs(g1), cmap="gray")
            axes1[1, 1].set_title(f"Blurred + noise, sigma = {sigma}")
            axes1[1, 1].axis("off")

            plt.tight_layout()

            fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
            axes2[0].imshow(np.abs(np.real(fHat)), cmap="gray")
            axes2[0].set_title(f"Fixed : gamma = {gamma}")
            axes2[0].axis("off")

            axes2[1].imshow(np.abs(np.real(fHat1)), cmap="gray")
            axes2[1].set_title(f"Fixed : gamma = {gamma / 10.0}")
            axes2[1].axis("off")

            axes2[2].imshow(fHat2, cmap="gray")
            axes2[2].set_title(f"Optimisation : gamma = {Lagra}")
            axes2[2].axis("off")

            plt.tight_layout()
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure532(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter05 script `Figure532.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from helpers.libdipum.lpfilter import lpfilter
            from helpers.libdip.imRecon4e import imRecon4e

            H = lpfilter("ideal", 480, 480, 40)
            f = np.fft.fftshift(H)

            # Horizontal back projection
            gh = imRecon4e(f, 90)

            # Vertical back projection
            gv = imRecon4e(f, 0)

            # Now get a horizontal and a vertical projection.
            ghv = imRecon4e(f, [90, 0])

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            # 1. Original
            axes[0, 0].imshow(f, cmap="gray")
            axes[0, 0].set_title("Original")
            axes[0, 0].axis("off")

            # 2. Horizontal BackProj
            axes[0, 1].imshow(gh, cmap="gray")
            axes[0, 1].set_title("BackProj (90)")
            axes[0, 1].axis("off")

            # 3. Vertical BackProj
            axes[1, 0].imshow(gv, cmap="gray")
            axes[1, 0].set_title("BackProj (0)")
            axes[1, 0].axis("off")

            # 4. Both
            axes[1, 1].imshow(ghv, cmap="gray")
            axes[1, 1].set_title("BackProj (90, 0)")
            axes[1, 1].axis("off")

            plt.tight_layout()
            print("Saved Figure532.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure533(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter05 script `Figure533.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from helpers.libdip.imRecon4e import imRecon4e

            # Parameters
            NR = 256
            Center = NR / 2  # 128.0

            x = np.arange(1, NR + 1)
            y = np.arange(1, NR + 1)
            Col, Row = np.meshgrid(x, y)

            Col = Col - Center
            Row = Row - Center

            Radius = np.sqrt(Col**2 + Row**2)
            X = Radius < 30

            # Angles
            Angles = 5.625 * np.arange(32)

            # Reconstructions
            print("Computing Reconstructions...")

            # 1. BackProj (0)
            print("- BackProj (0)")
            recon_0 = imRecon4e(X, [0])

            # 2. BackProj (0, 45)
            print("- BackProj (0, 45)")
            recon_0_45 = imRecon4e(X, [0, 45])

            # 3. BackProj (0, 45, 90)
            print("- BackProj (0, 45, 90)")
            recon_0_45_90 = imRecon4e(X, [0, 45, 90])

            # 4. BackProj (0, 45, 90, 135)
            print("- BackProj (0, 45, 90, 135)")
            recon_0_45_90_135 = imRecon4e(X, [0, 45, 90, 135])

            # 5. BackProj (31 angles)
            print(f"- BackProj ({len(Angles)} angles)")
            recon_angles = imRecon4e(X, Angles)

            # Display
            print("Displaying results. Please close plot to continue.")

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            # 1. X
            axes[0].imshow(X, cmap="gray")
            axes[0].set_title("X")
            axes[0].axis("off")

            # 2. BackProj (0)
            axes[1].imshow(recon_0, cmap="gray")
            axes[1].set_title("BackProj (0)")
            axes[1].axis("off")

            # 3. BackProj (0, 45)
            axes[2].imshow(recon_0_45, cmap="gray")
            axes[2].set_title("BackProj (0, 45)")
            axes[2].axis("off")

            # 4. BackProj (0, 45, 90)
            axes[3].imshow(recon_0_45_90, cmap="gray")
            axes[3].set_title("BackProj (0, 45, 90)")
            axes[3].axis("off")

            # 5. BackProj (0, 45, 90, 135)
            axes[4].imshow(recon_0_45_90_135, cmap="gray")
            axes[4].set_title("BackProj (0, 45, 90, 135)")
            axes[4].axis("off")

            # 6. BackProj (31 angles) -> 32 actually
            axes[5].imshow(recon_angles, cmap="gray")
            axes[5].set_title("BackProj (31 angles)")  # Keeping MATLAB title
            axes[5].axis("off")

            plt.tight_layout()

            # Save to file
            filename = "Figure533.png"
            print(f"Saved figure to {filename}")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure534(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter05 script `Figure534.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.libdip.imRecon4e import imRecon4e
            from helpers.libdipum.data_path import dip_data

            # Parameters
            # Angles32 = 5.625 * (0 : 31);
            Angles32 = 5.625 * np.arange(32)
            # Angles64 = 2.8125 * (0 : 63);
            Angles64 = 2.8125 * np.arange(64)

            # Data
            img_name = dip_data("ellipse_and_circle.tif")
            f = imread(img_name)

            # Reconstructions
            print("Computing BackProj (90)...")
            rec90 = imRecon4e(f, 90)

            print("Computing BackProj (0, 90)...")
            rec0_90 = imRecon4e(f, [0, 90])

            print("Computing BackProj (0, 45, 90, 135)...")
            rec0_45_90_135 = imRecon4e(f, [0, 45, 90, 135])

            print("Computing BackProj (32 angles)...")
            # imRecon4e might be slow for many angles, but it implements waitbar roughly by printing (or not).
            rec32 = imRecon4e(f, Angles32)

            print("Computing BackProj (64 angles)...")
            rec64 = imRecon4e(f, Angles64)

            # Display
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            axes[0, 0].imshow(f, cmap="gray")
            axes[0, 0].set_title("f")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(rec90, cmap="gray")
            axes[0, 1].set_title("BackProj (90)")
            axes[0, 1].axis("off")

            axes[0, 2].imshow(rec0_90, cmap="gray")
            axes[0, 2].set_title("BackProj (0, 90)")
            axes[0, 2].axis("off")

            axes[1, 0].imshow(rec0_45_90_135, cmap="gray")
            axes[1, 0].set_title("BackProj (0, 45, 90, 135)")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(rec32, cmap="gray")
            axes[1, 1].set_title("BackProj (32 angles)")
            axes[1, 1].axis("off")

            axes[1, 2].imshow(rec64, cmap="gray")
            axes[1, 2].set_title("BackProj (64 angles)")
            axes[1, 2].axis("off")

            plt.tight_layout()
            print("Saved Figure534.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure538(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter05 script `Figure538.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from helpers.libgeneral.radon import radon
            from helpers.libdipum.data_path import dip_data

            # Parameters
            single_theta = 90
            # Unused in this figure (kept for parity with MATLAB script)
            theta = np.arange(0, 180)
            theta2 = [0, 90, 45, 135]
            theta3 = np.arange(0, 180, 5.625)
            debug = False

            # Data
            img_path = dip_data("wingding-circle-solid-small.tif")
            f1 = img_as_float(imread(img_path))

            # Process
            R, Rho = radon(f1, single_theta)

            # Display
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].imshow(f1, cmap="gray")
            axes[0].set_title("f1, r = 100")
            axes[0].axis("off")

            axes[1].plot(Rho, R, linewidth=1.0)
            axes[1].set_xlabel("rho")
            axes[1].set_title(f"Radon (f1), theta = {single_theta}")
            axes[1].axis("tight")
            axes[1].set_aspect("equal", adjustable="box")
            axes[1].grid(False)

            plt.tight_layout()
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure539(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter05 script `Figure539.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.data import shepp_logan_phantom
            from skimage.transform import radon, resize

            # Parameters
            NR = 256
            # Theta = 0 : 0.5 : 179.5;
            Theta = np.arange(0, 180, 0.5)

            # Data
            Rectangle = np.zeros((NR, NR))
            r_start = int(NR / 4)
            r_end = int(3 * NR / 4)
            c_center = int(NR / 2)
            c_start = c_center - 20
            c_end = c_center + 20

            Rectangle[r_start:r_end, c_start:c_end] = 1

            # SheppLogan
            base_phantom = shepp_logan_phantom()

            # Resize to NR x NR
            SheppLogan = resize(
                base_phantom, (NR, NR), anti_aliasing=True, preserve_range=True
            )

            rt_rectangle = radon(Rectangle, theta=Theta, circle=False)
            rt_shepp = radon(SheppLogan, theta=Theta, circle=False)

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            # 1. Rectangle
            axes[0, 0].imshow(Rectangle, cmap="gray")
            axes[0, 0].set_title("A rectangle")
            axes[0, 0].axis("off")

            # 2. Sinogram Rectangle
            # MATLAB: axis xy.
            # Theta on x-axis? No.
            # ylabel('theta'). xlabel('rho').
            # MATLAB radon returns [rho x theta].
            # MATLAB plotting: 'XData' [xp(1) xp(end)] -> rho range. 'YData' [Theta(end) Theta(1)].
            # So Y is Theta?
            # If MATLAB imshow shows it with YData as Theta, then Y axis is Theta.
            # skimage radon returns (projection_size, len(theta)). Rows are rho, cols are theta.
            # So x-axis is Theta (cols), y-axis is rho (rows).
            # MATLAB code: RadonTransform.Rectangle = flipud (RadonTransform.Rectangle');
            # MATLAB Transpose: (rho x theta)' -> (theta x rho).
            # Flipud: flips theta direction?
            # So in MATLAB result image, Rows are Theta, Cols are Rho.
            # Python radon: Rows are Rho, Cols are Theta.
            # To match MATLAB display (Y=Theta, X=Rho):
            # We need (Theta, Rho) array -> Python (Rows=Theta, Cols=Rho).
            # So we should transpose `radon` output.

            sinogram_rect = rt_rectangle.T  # (theta, rho)
            # flipud? Theta 0 to 180.
            # MATLAB YData: [Theta(end), Theta(1)]. 179.5 down to 0? Or 0 down to 179.5?
            # Usually sinograms are shown with 0 at top or bottom.
            # Let's just transpose to get Theta on Y-axis.

            axes[0, 1].imshow(sinogram_rect, cmap="gray", aspect="auto", origin="lower")
            axes[0, 1].set_title("Sinogram (Rect)")
            axes[0, 1].set_xlabel("rho")
            axes[0, 1].set_ylabel("theta")

            # 3. Shepp Logan
            axes[1, 0].imshow(SheppLogan, cmap="gray")
            axes[1, 0].set_title("Shepp Logan")
            axes[1, 0].axis("off")

            # 4. Sinogram Shepp
            sinogram_shepp = rt_shepp.T
            axes[1, 1].imshow(
                sinogram_shepp, cmap="gray", aspect="auto", origin="lower"
            )
            axes[1, 1].set_title("Sinogram (Shepp)")
            axes[1, 1].set_xlabel("rho")
            axes[1, 1].set_ylabel("theta")

            plt.tight_layout()
            print("Saved Figure539.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure54(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter05 script `Figure54.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from helpers.libdipum.imnoise2New import imnoise2New
            from PIL import Image
            from helpers.libdipum.data_path import dip_data

            # ------------------------------------------------------------
            # Parameters
            # ------------------------------------------------------------
            a = [0.15, 0, 5, 10, 0, 0.05]
            b = [0.07, 0.03, 1, np.nan, 0.3, 0.05]

            NBin = 256
            Bin = np.linspace(0, 1, NBin)
            BinCenters = 0.5 * (Bin[:-1] + Bin[1:])
            bin_width = Bin[1] - Bin[0]
            # ------------------------------------------------------------
            # Data
            # ------------------------------------------------------------
            f = np.array(Image.open(dip_data("test-pattern.tif")), dtype=np.float64)

            # im2double
            if f.max() > 1.0:
                f /= 255.0

            # f(1:2:end,1:2:end)  → Python slicing
            f = f[::2, ::2]
            M, N = f.shape

            # ------------------------------------------------------------
            # Add noise
            # ------------------------------------------------------------
            fn1, r = imnoise2New(f, "gaussian", a[0], b[0])
            fn2, r = imnoise2New(f, "rayleigh", a[1], b[1])
            fn3, r = imnoise2New(f, "erlang", a[2], b[2])
            fn4, r = imnoise2New(f, "exponential", a[3], b[3])
            fn5, r = imnoise2New(f, "uniform", a[4], b[4])
            fn6, r = imnoise2New(f, "salt & pepper", a[5], b[5])

            # ------------------------------------------------------------
            # Figure 1
            # ------------------------------------------------------------
            plt.figure(figsize=(12, 8))

            plt.subplot(2, 3, 1)
            plt.imshow(fn1, cmap="gray")
            plt.title(f"Gauss, μ = {a[0]}, σ = {b[0]}")
            plt.axis("off")

            plt.subplot(2, 3, 2)
            plt.imshow(fn2, cmap="gray")
            plt.title(f"Rayleigh, μ = {a[1]}, σ = {b[1]}")
            plt.axis("off")

            plt.subplot(2, 3, 3)
            plt.imshow(fn3, cmap="gray")
            plt.title(f"Erlang, μ = {a[2]}, σ = {b[2]}")
            plt.axis("off")

            plt.subplot(2, 3, 4)
            Hist, edges = np.histogram(fn1.ravel(), bins=256, range=(0, 1))
            Hist = Hist[1:-1]
            edges = edges[1:-1]
            plt.plot(edges[:-1], Hist)

            plt.subplot(2, 3, 5)
            Hist, edges = np.histogram(fn2.ravel(), bins=256, range=(0, 1))
            Hist = Hist[1:-1]
            edges = edges[1:-1]
            plt.plot(edges[:-1], Hist)

            plt.subplot(2, 3, 6)
            Hist, edges = np.histogram(fn3.ravel(), bins=256, range=(0, 1))
            Hist = Hist[1:-1]
            edges = edges[1:-1]
            plt.plot(edges[:-1], Hist)

            plt.tight_layout()
            plt.show()
            # plt.close()

            # ------------------------------------------------------------
            # Figure 2
            # ------------------------------------------------------------
            plt.figure(figsize=(12, 8))

            plt.subplot(2, 3, 1)
            plt.imshow(fn4, cmap="gray")
            plt.title(f"Exponential, a = {a[3]}")
            plt.axis("off")

            plt.subplot(2, 3, 2)
            plt.imshow(fn5, cmap="gray")
            plt.title(f"Uniform, a = {a[4]}, b = {b[4]}")
            plt.axis("off")

            plt.subplot(2, 3, 3)
            plt.imshow(fn6, cmap="gray")
            plt.title(f"Salt & Pepper, a = {a[5]}, b = {b[5]}")
            plt.axis("off")

            plt.subplot(2, 3, 4)
            Hist, edges = np.histogram(fn4.ravel(), bins=256, range=(0, 1))
            Hist = Hist[1:-1]
            edges = edges[1:-1]
            plt.plot(edges[:-1], Hist)

            plt.subplot(2, 3, 5)
            Hist, edges = np.histogram(fn5.ravel(), bins=256, range=(0, 1))
            Hist = Hist[1:-1]
            edges = edges[1:-1]
            plt.plot(edges[:-1], Hist)

            plt.subplot(2, 3, 6)
            Hist, edges = np.histogram(fn6.ravel(), bins=256, range=(0, 1))
            Hist = Hist[1:-1]
            edges = edges[1:-1]
            plt.plot(edges[:-1], Hist)

            plt.tight_layout()
            plt.show()
            # plt.close()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure540(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter05 script `Figure540.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.data import shepp_logan_phantom
            from skimage.transform import resize
            from helpers.libdip.imRecon4e import imRecon4e

            # Parameters
            NR = 256
            # Theta = 0 : 1 : 179;
            Theta = np.arange(0, 180, 1)

            # Data
            # Rectangle
            Rectangle = np.zeros((NR, NR))
            r_start = int(NR / 4)
            r_end = int(3 * NR / 4)
            c_center = int(NR / 2)
            c_start = c_center - 20
            c_end = c_center + 20

            Rectangle[r_start:r_end, c_start:c_end] = 1

            # SheppLogan
            base_phantom = shepp_logan_phantom()
            SheppLogan = resize(
                base_phantom, (NR, NR), anti_aliasing=True, preserve_range=True
            )

            # Display
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # 1. Rectangle BackProj
            print("Computing BackProj for Rectangle...")
            rec_rect = imRecon4e(Rectangle, Theta)
            axes[0].imshow(rec_rect, cmap="gray")
            axes[0].set_title("Back Projection 180 angles (Rect)")
            axes[0].axis("off")

            # 2. SheppLogan BackProj
            print("Computing BackProj for SheppLogan...")
            rec_shepp = imRecon4e(SheppLogan, Theta)
            axes[1].imshow(rec_shepp, cmap="gray")
            axes[1].set_title("Back Projection 180 angles (Shepp)")
            axes[1].axis("off")

            plt.tight_layout()
            print("Saved Figure540.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure543(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter05 script `Figure543.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.transform import radon, iradon

            # Parameters
            NR = 256
            # Theta = 0 : 1 : 179;
            Theta = np.arange(0, 180, 1)

            # Data
            X = np.zeros((NR, NR))
            r_start = int(NR / 4)
            r_end = int(3 * NR / 4)
            c_center = int(NR / 2)
            c_start = c_center - 20
            c_end = c_center + 20
            X[r_start:r_end, c_start:c_end] = 1

            # Radon transform
            R = radon(X, theta=Theta, circle=False)

            # Inverse Radon transform
            print("Computing Inverse Radon (Ram-Lak)...")
            XHat_RamLak = iradon(R, theta=Theta, filter_name="ramp", circle=False)

            # XHat.Hamming = iradon (R, Theta, 'Hamming');
            print("Computing Inverse Radon (Hamming)...")
            XHat_Hamming = iradon(R, theta=Theta, filter_name="hamming", circle=False)

            if XHat_RamLak.shape[0] > NR:
                # Center crop
                diff = XHat_RamLak.shape[0] - NR
                start = diff // 2
                XHat_RamLak = XHat_RamLak[start : start + NR, start : start + NR]

            if XHat_Hamming.shape[0] > NR:
                diff = XHat_Hamming.shape[0] - NR
                start = diff // 2
                XHat_Hamming = XHat_Hamming[start : start + NR, start : start + NR]

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            # 1. Original
            axes[0, 0].imshow(X, cmap="gray")
            axes[0, 0].set_title("Orig")
            axes[0, 0].axis("off")

            # 2. Radon
            # Display sinogram. Transpose to match (theta (y), rho (x)) or typical MATLAB view?
            # MATLAB: imshow(R, []).
            # If R is (rho, theta) in Python (skimage default), then imshow shows rho on y-axis (rows), theta on x-axis (cols).
            axes[0, 1].imshow(R, cmap="gray", aspect="auto")
            axes[0, 1].set_title("Radon transform")
            axes[0, 1].set_xlabel("theta")
            axes[0, 1].set_ylabel("rho")

            # 3. Ram-Lak
            axes[1, 0].imshow(XHat_RamLak, cmap="gray")
            axes[1, 0].set_title("RAM-LAK")
            axes[1, 0].axis("off")

            # 4. Hamming
            axes[1, 1].imshow(XHat_Hamming, cmap="gray")
            axes[1, 1].set_title("RAM-LAK + Hamming")
            axes[1, 1].axis("off")

            plt.tight_layout()
            print("Saved Figure543.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure544(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter05 script `Figure544.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.data import shepp_logan_phantom
            from skimage.transform import radon, iradon, resize

            # Parameters
            NR = 256
            # Theta = 0 : 1 : 179;
            Theta = np.arange(0, 180, 1)

            # Data
            base_phantom = shepp_logan_phantom()
            f = resize(base_phantom, (NR, NR), anti_aliasing=True, preserve_range=True)

            # Radon transform
            R = radon(f, theta=Theta, circle=False)

            # Inverse Radon transform
            print("Computing Inverse Radon (Ram-Lak)...")
            fHat_RamLak = iradon(R, theta=Theta, filter_name="ramp", circle=False)

            # fHat.Hamming = iradon (R, Theta, 'Hamming');
            print("Computing Inverse Radon (Hamming)...")
            fHat_Hamming = iradon(R, theta=Theta, filter_name="hamming", circle=False)

            # Crop to size if necessary
            if fHat_RamLak.shape[0] > NR:
                diff = fHat_RamLak.shape[0] - NR
                start = diff // 2
                fHat_RamLak = fHat_RamLak[start : start + NR, start : start + NR]

            if fHat_Hamming.shape[0] > NR:
                diff = fHat_Hamming.shape[0] - NR
                start = diff // 2
                fHat_Hamming = fHat_Hamming[start : start + NR, start : start + NR]

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            # 1. Original
            axes[0, 0].imshow(f, cmap="gray")
            axes[0, 0].set_title("Orig")
            axes[0, 0].axis("off")

            # 2. Radon
            axes[0, 1].imshow(R, cmap="gray", aspect="auto")
            axes[0, 1].set_title("Radon transform")
            axes[0, 1].set_xlabel("theta")
            axes[0, 1].set_ylabel("rho")

            # 3. Ram-Lak
            axes[1, 0].imshow(fHat_RamLak, cmap="gray")
            axes[1, 0].set_title("RAM-LAK")
            axes[1, 0].axis("off")

            # 4. Hamming
            axes[1, 1].imshow(fHat_Hamming, cmap="gray")
            axes[1, 1].set_title("RAM-LAK + Hamming")
            axes[1, 1].axis("off")

            plt.tight_layout()
            print("Saved Figure544.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure548(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter05 script `Figure548.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.libdipum.fanbeam import fanbeam
            from helpers.libdipum.ifanbeam import ifanbeam
            from helpers.libdip.intScaling4e import intScaling4e
            from helpers.libdipum.data_path import dip_data

            # Parameters
            NR = 256
            Diag = np.sqrt(2) * NR

            # Rotation increments for the 4 cases
            FanRotInc = [1.0, 0.5, 0.25, 0.125]
            # Corresponding Sensor Spacings (based on MATLAB code usage)
            # F1: Default (likely 1, 1).
            # F2: 0.5, 0.5
            # F3: 0.25, 0.25
            # F4: 0.125, 0.125
            FanSensorSpacing = [1.0, 0.5, 0.25, 0.125]

            # Data
            img_name = dip_data("vertical_rectangle.tif")

            g = imread(img_name)

            g = g.astype(float)
            if g.max() > 1:
                g /= 255.0

            M, N = g.shape
            # D: Distance source to center
            d_source = np.sqrt(M**2 + N**2) + 10

            results = []

            for i in range(4):
                rot_inc = FanRotInc[i]
                sensor_spacing = FanSensorSpacing[i]

                print(
                    f"Case {i + 1}: FanRotInc={rot_inc}, FanSensorSpacing={sensor_spacing}..."
                )

                # Fanbeam
                # Note: D corresponds to 'd' in MATLAB code
                F, gamma, beta = fanbeam(
                    g,
                    D=d_source,
                    FanRotationIncrement=rot_inc,
                    FanSensorSpacing=sensor_spacing,
                )

                # Inverse Fanbeam
                # OutputSize=600 in MATLAB
                g_recon = ifanbeam(
                    F,
                    D=d_source,
                    FanRotationIncrement=rot_inc,
                    FanSensorSpacing=sensor_spacing,
                    filter="Hamming",
                    OutputSize=600,
                )

                # Scale
                g_scaled = intScaling4e(g_recon, "full")
                results.append(g_scaled)

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes = axes.flatten()

            for i, ax in enumerate(axes):
                ax.imshow(results[i], cmap="gray")
                ax.set_title(f"Inc: {FanRotInc[i]}")
                ax.axis("off")

            plt.tight_layout()
            print("Saved Figure548.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure549(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter05 script `Figure549.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.data import shepp_logan_phantom
            from skimage.transform import resize
            from helpers.libdipum.fanbeam import fanbeam
            from helpers.libdipum.ifanbeam import ifanbeam
            from helpers.libdip.intScaling4e import intScaling4e

            # Parameters
            # f = phantom(600);
            NR = 600
            base_phantom = shepp_logan_phantom()
            f = resize(base_phantom, (NR, NR), anti_aliasing=True, preserve_range=True)

            # d = sqrt(size(f,1)^2 + size(f,2)^2) + 10;
            M, N = f.shape
            d_source = np.sqrt(M**2 + N**2) + 10

            # Fan Parameters
            FanRotInc = [1.0, 0.5, 0.25, 0.125]
            FanSensorSpacing = [1.0, 0.5, 0.25, 0.125]

            results = []

            for i in range(4):
                rot_inc = FanRotInc[i]
                sensor_spacing = FanSensorSpacing[i]

                print(
                    f"Case {i + 1}: FanRotInc={rot_inc}, FanSensorSpacing={sensor_spacing}..."
                )

                # Fanbeam
                F, gamma, beta = fanbeam(
                    f,
                    D=d_source,
                    FanRotationIncrement=rot_inc,
                    FanSensorSpacing=sensor_spacing,
                )

                # Inverse Fanbeam
                # OutputSize=600
                g_recon = ifanbeam(
                    F,
                    D=d_source,
                    FanRotationIncrement=rot_inc,
                    FanSensorSpacing=sensor_spacing,
                    filter="Hamming",
                    OutputSize=NR,
                )

                # Scale
                g_scaled = intScaling4e(g_recon, "full")
                results.append(g_scaled)

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes = axes.flatten()

            for i, ax in enumerate(axes):
                ax.imshow(results[i], cmap="gray")
                ax.set_title(f"Inc: {FanRotInc[i]}")
                ax.axis("off")

            plt.tight_layout()
            print("Saved Figure549.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure55(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter05 script `Figure55.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from skimage.color import rgb2gray
            from helpers.libdipum.cnotch import cnotch
            from helpers.libdipum.imnoise3 import imnoise3
            from helpers.libdip.intScaling4e import intScaling4e
            from helpers.libdipum.dftfilt import dftfilt
            from helpers.libdipum.data_path import dip_data

            # Data
            img_path = dip_data("astronaut.tif")

            f_orig = imread(img_path)
            if f_orig.ndim == 3:
                f_orig = rgb2gray(f_orig)
            f = img_as_float(f_orig)

            M, N = f.shape

            # Generate sinusoidal noise pattern (r), its DFT (R), and its spectrum (S)
            r, R, S = imnoise3(M, N, [25, 25], A=[0.3])
            g = f + r
            gs = intScaling4e(g)

            # Compute spectrum of g
            G = np.fft.fftshift(np.abs(np.fft.fft2(g)))
            # Follow MATLAB: 1 + log(G). Use small epsilon to avoid log(0) warnings.
            Glog = intScaling4e(1 + np.log(G + 1e-9))

            # Create notch filters
            # MATLAB: [M/2+1+25, N/2+1+25] (1-based). For Python 0-based, use M//2 + 25, N//2 + 25.
            impulse_loc = [M // 2 + 25, N // 2 + 25]
            H = cnotch("ideal", "reject", M, N, [impulse_loc], 2)
            Hc = intScaling4e(np.fft.fftshift(H))

            # Filter image
            gf = intScaling4e(dftfilt(g, H))

            # Display results
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            axes[0, 0].imshow(gs, cmap="gray")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(Glog, cmap="gray")
            axes[0, 1].axis("off")

            axes[1, 0].imshow(Hc, cmap="gray")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(gf, cmap="gray")
            axes[1, 1].axis("off")

            plt.tight_layout()
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure57(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter05 script `Figure57.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from skimage.color import rgb2gray
            from helpers.libdipum.imnoise2 import imnoise2
            from helpers.libdipum.spfilt import spfilt
            from helpers.libdipum.data_path import dip_data

            # Parameters
            kernel_size = 3
            mean = 0
            std = 0.1

            # Data
            img_path = dip_data("circuitboard.tif")
            f_orig = imread(img_path)
            if f_orig.ndim == 3:
                f_orig = rgb2gray(f_orig)
            f = img_as_float(f_orig)

            # Noise adding
            fn, _ = imnoise2(f, "gaussian", mean, std)

            # Filtering
            fHatArithMean = spfilt(fn, "amean", kernel_size, kernel_size)
            fHatGeoMean = spfilt(fn, "gmean", kernel_size, kernel_size)

            # Display
            # show_image_window(f, "Original")
            # show_image_window(fn, "Noisy")
            # show_image_window(fHatArithMean, "Arithmetic Mean Filter")
            # show_image_window(fHatGeoMean, "Geometric Mean Filter")

            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            axes[0, 0].imshow(f, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axes[0, 0].axis("off")
            axes[0, 0].set_title("Original")

            axes[0, 1].imshow(fn, cmap="gray", interpolation="nearest", resample=False)
            axes[0, 1].axis("off")
            axes[0, 1].set_title("Gaussian noise")

            axes[1, 0].imshow(fHatArithMean, cmap="gray")
            axes[1, 0].axis("off")
            axes[1, 0].set_title("Arithmetic mean")

            axes[1, 1].imshow(fHatGeoMean, cmap="gray")
            axes[1, 1].axis("off")
            axes[1, 1].set_title("Geometric mean")

            plt.tight_layout()
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure58(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter05 script `Figure58.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from skimage.color import rgb2gray
            from helpers.libdipum.imnoise2 import imnoise2
            from helpers.libdipum.spfilt import spfilt
            from helpers.libdipum.data_path import dip_data

            # Parameters
            kernel_size = 3
            Q = [1.5, -1.5]
            p_salt = 0.05
            p_pepper = 0.05

            # Data
            img_path = dip_data("circuitboard.tif")
            f_orig = imread(img_path)
            if f_orig.ndim == 3:
                f_orig = rgb2gray(f_orig)
            f = img_as_float(f_orig)

            # Noise adding
            fnPepper, _ = imnoise2(f, "salt & pepper", 0, p_pepper)
            fnSalt, _ = imnoise2(f, "salt & pepper", p_salt, 0)

            # Filtering
            fHatContraHarmonic1 = spfilt(
                fnPepper, "chmean", kernel_size, kernel_size, Q[0]
            )
            fHatContraHarmonic2 = spfilt(
                fnSalt, "chmean", kernel_size, kernel_size, Q[1]
            )

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            imshow_kwargs = dict(cmap="gray", interpolation="nearest")

            axes[0, 0].imshow(fnPepper, **imshow_kwargs)
            axes[0, 0].axis("off")
            axes[0, 0].set_title("Pepper noise")

            axes[0, 1].imshow(fnSalt, **imshow_kwargs)
            axes[0, 1].axis("off")
            axes[0, 1].set_title("Salt noise")

            axes[1, 0].imshow(fHatContraHarmonic1, **imshow_kwargs)
            axes[1, 0].axis("off")
            axes[1, 0].set_title("Contra harmonic filter")

            axes[1, 1].imshow(fHatContraHarmonic2, **imshow_kwargs)
            axes[1, 1].axis("off")
            axes[1, 1].set_title("Contra harmonic filter")

            plt.tight_layout()
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure59(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter05 script `Figure59.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from skimage.color import rgb2gray
            from helpers.libdipum.imnoise2 import imnoise2
            from helpers.libdipum.spfilt import spfilt
            from helpers.libdipum.data_path import dip_data

            # Parameters
            kernel_size = 3
            Q = [-1.5, 1.5]  # MATLAB: Q = -[1.5, -1.5]
            p_salt = 0.05
            p_pepper = 0.05

            # Data
            img_path = dip_data("circuitboard.tif")
            f_orig = imread(img_path)
            if f_orig.ndim == 3:
                f_orig = rgb2gray(f_orig)
            f = img_as_float(f_orig)

            # Noise adding
            fnPepper, _ = imnoise2(f, "salt & pepper", 0, p_pepper)
            fnSalt, _ = imnoise2(f, "salt & pepper", p_salt, 0)

            # Filtering
            fHatContraHarmonic1 = spfilt(
                fnPepper, "chmean", kernel_size, kernel_size, Q[0]
            )
            fHatContraHarmonic2 = spfilt(
                fnSalt, "chmean", kernel_size, kernel_size, Q[1]
            )

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            imshow_kwargs = dict(cmap="gray", vmin=0, vmax=1, interpolation="nearest")

            axes[0, 0].imshow(fnPepper, **imshow_kwargs)
            axes[0, 0].axis("off")
            axes[0, 0].set_title("Pepper noise")

            axes[0, 1].imshow(fnSalt, **imshow_kwargs)
            axes[0, 1].axis("off")
            axes[0, 1].set_title("Salt noise")

            axes[1, 0].imshow(fHatContraHarmonic1, **imshow_kwargs)
            axes[1, 0].axis("off")
            axes[1, 0].set_title("Contra harmonic filter")

            axes[1, 1].imshow(fHatContraHarmonic2, **imshow_kwargs)
            axes[1, 1].axis("off")
            axes[1, 1].set_title("Contra harmonic filter")

            plt.tight_layout()
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

