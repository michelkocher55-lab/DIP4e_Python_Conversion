from pathlib import Path as _Path
import os as _os

from typing import Any


class Chapter04Mixin:
    def figure42(self, data_dir: str | None = None) -> dict[str, object]:
        """Run Chapter02 script `Figure42.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(
            data_dir=data_dir
        )
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            import numpy as np

            # %% Figure 4.2
            M = 2048
            m = np.arange(1, 1025)
            m = m[2::8]
            Cn2048 = (m ** 2) / (4 * np.log2(M))
            Cs2048 = m / (2 * np.log2(M))

            fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
            ax[0].plot(Cn2048)
            ax[0].set_xlabel('m')
            ax[0].set_title('C_n[m]')
            ax[0].axis('tight')
            ax[0].set_box_aspect(1)

            ax[1].plot(Cs2048)
            ax[1].set_xlabel('m')
            ax[1].set_title('C_s[m]')
            ax[1].axis('tight')
            ax[1].set_box_aspect(1)

            plt.tight_layout()
            plt.savefig('Figure42.png', dpi=150)
            plt.show()
            plt.tight_layout()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure44(self, data_dir: str | None = None) -> dict[str, object]:
        """Run Chapter02 script `Figure44.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(
            data_dir=data_dir
        )
        try:
            import numpy as np
            import matplotlib.pyplot as plt

            def mysinc(L, H, k, inc):
                x = np.arange(L, H + inc / 2.0, inc)
                s = np.sin(np.pi * k * x) / (np.pi * k * x + np.finfo(float).eps)
                j = np.where(x == 0)[0]
                s[j] = 1.0
                return s

            s = mysinc(-9.5, 9.5, 1.0, 0.0001)

            plt.figure(figsize=(9, 4.5))
            plt.subplot(1, 2, 1)
            plt.plot(s)
            plt.axis('tight')
            plt.gca().set_box_aspect(1)

            plt.subplot(1, 2, 2)
            plt.plot(np.abs(s))
            plt.axis('tight')
            plt.gca().set_box_aspect(1)

            plt.tight_layout()
            plt.savefig('Figure44.png', dpi=150)
            plt.show()

        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure414(self, data_dir: str | None = None) -> dict[str, object]:
        """Run Chapter02 script `Figure414.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(
            data_dir=data_dir
        )
        try:
            import numpy as np
            import matplotlib.pyplot as plt

            def example4pt5(A, S, T, M, N):
                f = np.zeros((M, N), dtype=float)
                f[:S, :T] = A
                F = np.fft.fft2(f)
                F = np.fft.fftshift(F)
                Spec = np.abs(F) + np.finfo(float).eps
                return f, Spec

            f, Spec = example4pt5(1, 6, 8, 1024, 1024)

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            Z = Spec[9::20, 9::20]
            Y, X = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
            ax.plot_wireframe(X, Y, Z, color='k', linewidth=0.6)
            ax.set_axis_off()
            plt.tight_layout()
            plt.savefig('Figure414.png', dpi=150)
            plt.show()

        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure418(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure418.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from DIP4eFigures.checkerimage import checkerimage

            print("Running Figure418 (Aliasing with Checkerboard)...")

            # Parameters
            M = 96

            # Generate images
            # g1=checkerimage(0.0625,96);
            g1 = checkerimage(0.0625, M)

            # g2=checkerimage(0.1667,96);
            g2 = checkerimage(0.1667, M)

            # g3=checkerimage(1.05,96);
            g3 = checkerimage(1.05, M)

            # g4=checkerimage(2.084,96);
            g4 = checkerimage(2.084, M)

            # Save images (using 'tif' as in MATLAB script, or 'png' for easy viewing?
            # MATLAB: imwrite(..., 'Fig4.18...tif')
            # I will save as .tif to match, but also display/save a composite png.

            # Note: Images are 0/1 binary. imsave expects appropriate type or converts.
            # We should convert to uint8 (0, 255) for standard TIFF compatibility.

            g1_u8 = (g1 * 255).astype(np.uint8)
            g2_u8 = (g2 * 255).astype(np.uint8)
            g3_u8 = (g3 * 255).astype(np.uint8)
            g4_u8 = (g4 * 255).astype(np.uint8)

            # Display composite
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes = axes.flatten()

            axes[0].imshow(g1, cmap="gray")
            axes[0].set_title("delX = 0.0625 (No aliasing)")
            axes[0].axis("off")

            axes[1].imshow(g2, cmap="gray")
            axes[1].set_title("delX = 0.1667 (No aliasing)")
            axes[1].axis("off")

            axes[2].imshow(g3, cmap="gray")
            axes[2].set_title("delX = 1.05 (Aliasing)")
            axes[2].axis("off")

            axes[3].imshow(g4, cmap="gray")
            axes[3].set_title("delX = 2.084 (Severe Aliasing)")
            axes[3].axis("off")

            plt.tight_layout()
            plt.savefig("Figure418.png")
            print("Saved Figure418.png and component TIFs.")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure419(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure419.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from skimage.transform import resize
            from scipy.ndimage import correlate
            from helpers.data_path import dip_data

            print("Running Figure419 (Aliasing and Lowpass Filtering)...")

            # Parameters
            HSize = 5

            # Data Loading
            img_path = dip_data("barbara.tif")

            f_orig = imread(img_path)
            if f_orig.ndim == 3:
                f_orig = f_orig[:, :, 0]
            f = img_as_float(f_orig)

            # Decimation
            # fd = f (1:3:end, 1:3:end);
            fd = f[::3, ::3]

            # Interpolation nearest neighbor
            # fdi = imresize(fd, 3, 'nearest');
            # skimage resize takes output_shape.
            # Output shape should match f if divisible by 3, or be 3x dims of fd.
            # fd dimensions are ceil(M/3), ceil(N/3) roughly or floor depending on slicing.
            # MATLAB slicing 1:3:end includes the first element.
            # Python [::3] includes 0, 3, 6...
            # Resulting shape:
            target_shape = (fd.shape[0] * 3, fd.shape[1] * 3)
            # Note: original f might handle boundaries differently.
            # If f is 512x512, f[::3] len is 171. 171*3 = 513.
            # So fdi might be slightly larger or smaller than f.
            # I'll resize to f.shape if close, or just 3*fd.shape.
            # MATLAB imresize(fd, 3) scales by factor 3.

            fdi = resize(fd, target_shape, order=0, mode="edge", anti_aliasing=False)

            # Low pass filtering
            # h = fspecial ('average', HSize);
            h = np.ones((HSize, HSize)) / (HSize * HSize)

            # flp = imfilter (f, h, 'symmetric', 'same');
            # 'symmetric' -> 'reflect'
            flp = correlate(f, h, mode="reflect")

            # Decimation of the low pass filtered image
            # flpd = flp (1:3:end, 1:3:end);
            flpd = flp[::3, ::3]

            # Interpolation nearest neighbor
            # flpdi = imresize(flpd, 3, 'nearest');
            flpdi = resize(
                flpd, target_shape, order=0, mode="edge", anti_aliasing=False
            )

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            axes[0, 0].imshow(f, cmap="gray")
            axes[0, 0].set_title("Original")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(fdi, cmap="gray")
            axes[0, 1].set_title("Decimated, Interpolated NN")
            axes[0, 1].axis("off")

            axes[1, 0].imshow(flp, cmap="gray")
            axes[1, 0].set_title("Low Pass Filtered")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(flpdi, cmap="gray")
            axes[1, 1].set_title("Smoothed, Decimated, Interpolated NN")
            axes[1, 1].axis("off")

            plt.tight_layout()
            plt.savefig("Figure419.png")
            print("Saved Figure419.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure423(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure423.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from DIP4eFigures.intScaling4e import intScaling4e

            print("Running Figure423 (Spectra of Centered Rectangle)...")

            # Data Centered rectangle
            f = np.zeros((512, 512))
            f[207:304, 247:264] = 1

            # DFT
            # F=fft2(f);
            F = np.fft.fft2(f)

            # S=abs(F);
            S = np.abs(F)

            # Spectrum = intScaling4e(S);
            Spectrum = intScaling4e(S)

            # Sc=fftshift(S);
            Sc = np.fft.fftshift(S)

            # SpectrumCentered = intScaling4e(Sc);
            SpectrumCentered = intScaling4e(Sc)

            # ScLog=log10(1 + abs(Sc));
            ScLog = np.log10(1 + np.abs(Sc))

            # SpectrumCenteredLog = intScaling4e(ScLog);
            SpectrumCenteredLog = intScaling4e(ScLog)

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            axes[0, 0].imshow(f, cmap="gray")
            axes[0, 0].set_title("Original Image")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(Spectrum, cmap="gray")
            axes[0, 1].set_title("Spectrum (Uncentered)")
            axes[0, 1].axis("off")

            axes[1, 0].imshow(SpectrumCentered, cmap="gray")
            axes[1, 0].set_title("Centered Spectrum")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(SpectrumCenteredLog, cmap="gray")
            axes[1, 1].set_title("Centered Log Spectrum")
            axes[1, 1].axis("off")

            plt.tight_layout()
            plt.savefig("Figure423.png")
            print("Saved Figure423.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure424_425(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure424_425.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.transform import rotate

            # Centered rectangle
            f = np.zeros((512, 512), dtype=float)
            f[207:304, 247:264] = 1  # MATLAB 208:304, 248:264

            # Displaced rectangle
            g = np.zeros((512, 512), dtype=float)
            g[107:204, 347:364] = 1  # MATLAB 108:204, 348:364

            # Rotated rectangle (bilinear, crop)
            r = rotate(
                f,
                angle=-45,
                resize=False,
                order=1,
                mode="constant",
                cval=0.0,
                preserve_range=True,
            )

            # Fourier transform
            F = np.fft.fft2(f)

            G = np.fft.fft2(g)
            SG = np.abs(G)
            SG = np.fft.fftshift(SG)
            SG = np.log10(1 + np.abs(SG))
            SG = SG - SG.min()
            SG = SG / SG.max()

            R = np.fft.fft2(r)
            SR = np.abs(R)
            SR = np.fft.fftshift(SR)
            SR = np.log10(1 + np.abs(SR))
            SR = SR - SR.min()
            SR = SR / SR.max()

            # Phase angles
            fphi = np.angle(F)
            gphi = np.angle(G)
            rphi = np.angle(R)

            # Reconstruct
            Cf = 1j * fphi
            Cg = 1j * gphi

            Freconst = np.abs(F) * np.exp(Cf)
            frec = np.real(np.fft.ifft2(Freconst))

            Greconst = np.abs(G) * np.exp(Cg)
            grec = np.real(np.fft.ifft2(Greconst))

            # Display figure 1
            fig1, axes1 = plt.subplots(2, 2, figsize=(10, 10))
            axes1[0, 0].imshow(f, cmap="gray")
            axes1[0, 0].set_title("f")
            axes1[0, 0].axis("off")

            axes1[0, 1].imshow(frec, cmap="gray")
            axes1[0, 1].set_title("f_rec")
            axes1[0, 1].axis("off")

            axes1[1, 0].imshow(g, cmap="gray")
            axes1[1, 0].set_title("g")
            axes1[1, 0].axis("off")

            axes1[1, 1].imshow(grec, cmap="gray")
            axes1[1, 1].set_title("g_rec")
            axes1[1, 1].axis("off")

            plt.tight_layout()
            plt.savefig("Figure424.png")

            # Display figure 2
            fig2, axes2 = plt.subplots(2, 2, figsize=(10, 10))
            axes2[0, 0].imshow(g, cmap="gray")
            axes2[0, 0].set_title("f translated")
            axes2[0, 0].axis("off")

            axes2[0, 1].imshow(SG, cmap="gray")
            axes2[0, 1].set_title("Fourier (f translated)")
            axes2[0, 1].axis("off")

            axes2[1, 0].imshow(r, cmap="gray")
            axes2[1, 0].set_title("f rotated")
            axes2[1, 0].axis("off")

            axes2[1, 1].imshow(SR, cmap="gray")
            axes2[1, 1].set_title("Fourier (f rotated)")
            axes2[1, 1].axis("off")

            plt.tight_layout()
            plt.savefig("Figure424Bis.png")

            # Display figure 3
            fig3, axes3 = plt.subplots(1, 3, figsize=(12, 4))
            axes3[0].imshow(fphi, cmap="gray")
            axes3[0].axis("off")

            axes3[1].imshow(gphi, cmap="gray")
            axes3[1].axis("off")

            axes3[2].imshow(rphi, cmap="gray")
            axes3[2].axis("off")

            plt.tight_layout()
            plt.savefig("Figure425.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure426(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure426.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.transform import resize
            from skimage.util import img_as_float
            from helpers.data_path import dip_data

            # Data
            img_path = dip_data("boy.tif")
            f_orig = imread(img_path)

            # MATLAB imresize preserves range for double; use preserve_range and then convert to float
            f_resized = resize(
                f_orig,
                (1774, 1546),
                order=1,
                mode="reflect",
                anti_aliasing=True,
                preserve_range=True,
            )
            f = img_as_float(f_resized)

            # Rectangle
            g = np.zeros((1774, 1546), dtype=float)
            # MATLAB 722:1052, 749:797 -> Python 0-based
            g[721:1052, 748:797] = 1

            # Fourier transform
            F = np.fft.fft2(f)
            G = np.fft.fft2(g)

            # Reconstruct using mixed magnitude/phase
            RecGModulusFAngle = np.real(
                np.fft.ifft2(np.abs(G) * np.exp(1j * np.angle(F)))
            )
            RecFModulusGAngle = np.real(
                np.fft.ifft2(np.abs(F) * np.exp(1j * np.angle(G)))
            )

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            axes[0, 0].imshow(f, cmap="gray")
            axes[0, 0].set_title("f")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(RecFModulusGAngle, cmap="gray")
            axes[0, 1].set_title("Rec by using |F| and arg (G)")
            axes[0, 1].axis("off")

            axes[1, 0].imshow(g, cmap="gray")
            axes[1, 0].set_title("g")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(RecGModulusFAngle, cmap="gray")
            axes[1, 1].set_title("Rec by using |G| and arg (F)")
            axes[1, 1].axis("off")

            plt.tight_layout()
            plt.savefig("Figure426.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure428(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure428.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from helpers.data_path import dip_data

            print("Running Figure428 (Blown IC and Spectrum)...")

            # Image loading
            img_path = dip_data("blown_ic.tif")
            f = imread(img_path)
            if f.ndim == 3:
                f = f[:, :, 0]

            f = img_as_float(f)

            # Fourier transform
            # F = fft2(f);
            F = np.fft.fft2(f)

            # S = abs(F);
            S = np.abs(F)

            # S = fftshift(S);
            S = np.fft.fftshift(S)

            # S = log10(1+S);
            S_log = np.log10(1 + S)

            # Scaling for display
            # S = S - min(S(:));
            # S = S/max(S(:));
            S_disp = S_log - np.min(S_log)
            if np.max(S_disp) > 0:
                S_disp = S_disp / np.max(S_disp)

            # Display
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            axes[0].imshow(f, cmap="gray")
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            axes[1].imshow(S_disp, cmap="gray")
            axes[1].set_title("Fourier Spectrum")
            axes[1].axis("off")

            plt.tight_layout()
            plt.savefig("Figure428.png")
            print("Saved Figure428.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure429(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure429.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from DIP4eFigures.lpFilterTF4e import lpFilterTF4e
            from DIP4eFigures.dftFiltering4e import dftFiltering4e
            from helpers.data_path import dip_data

            # Image loading
            img_path = dip_data("blown_ic.tif")
            f = imread(img_path)
            if f.ndim == 3:
                f = f[:, :, 0]  # if grayscale
            f = img_as_float(f)
            M, N = f.shape

            # Low pass filter creation (Centered)
            # HNoPad = lpFilterTF4e('ideal',M,N,0.5);
            # HNoPad = imcomplement(HNoPad); -> 1 - H
            HNoPad_low = lpFilterTF4e("ideal", M, N, 0.5)
            HNoPad = 1.0 - HNoPad_low

            # With Padding
            # PQ = paddedsize(size(f)); -> dftFiltering4e pads to 2*M, 2*N internally if padmode != 'none'
            # Wait, dftFiltering4e documentation says:
            # "Unless padmode = 'none', function DFTFILTERING4E pads the input image to size P-by-Q, with P = 2*M and Q = 2*N"
            # So we must generate H of size 2M, 2N.

            P, Q = 2 * M, 2 * N
            HPad_low = lpFilterTF4e("ideal", P, Q, 0.5)
            HPad = 1.0 - HPad_low

            # Filtering

            # NO PADDING
            # gNoPad=dftfilt(f,HNoPad); -> dftFiltering4e(f, HNoPad, 'none')
            gNoPad = dftFiltering4e(f, HNoPad, "none")

            # WITH PADDING
            # gPad=dftfilt(f,HPad); -> dftFiltering4e(f, HPad, 'zeros' or 'replicate'?)
            # Input script Figure429.m uses `lpFilterTF4e('ideal',PQ(1),PQ(2),0.5)` where PQ=paddedsize(size(f)).
            # And calls gPad=dftfilt(f,HPad). dftfilt uses 'zeros' (constant) padding by default.
            # dftFiltering4e defaults to 'replicate'.
            # We should use 'zeros' to match Figure429.m logic?
            # "gPad=dftfilt(f,HPad)" in Figure429.m calls dftfilt.m.
            # dftfilt.m default padMethod is 'zeros' (0).
            # So I should use 'zeros' in dftFiltering4e to replicate behavior.

            gPad = dftFiltering4e(f, HPad, "zeros")

            # Stats
            print(
                f"No Pad: Min={gNoPad.min():.4f}, Max={gNoPad.max():.4f}, Mean={gNoPad.mean():.4e}"
            )
            print(
                f"With Pad: Min={gPad.min():.4f}, Max={gPad.max():.4f}, Mean={gPad.mean():.4f}"
            )

            # Display
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(f, cmap="gray", vmin=0, vmax=1)
            axes[0].set_title("Original")
            axes[0].axis("off")

            axes[1].imshow(gNoPad, cmap="gray", vmin=0, vmax=1)
            axes[1].set_title("HighPass No Padding")
            axes[1].axis("off")

            axes[2].imshow(gPad, cmap="gray", vmin=0, vmax=1)
            axes[2].set_title("HighPass With Padding")
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig("Figure429.png")
            print("Saved Figure429.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure430(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure430.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from DIP4eFigures.lpFilterTF4e import lpFilterTF4e
            from DIP4eFigures.hpFilterTF4e import hpFilterTF4e
            from DIP4eFigures.dftFiltering4e import dftFiltering4e
            from helpers.data_path import dip_data

            def mesh_plot(ax: Any, H: Any):
                """mesh_plot."""
                # Helper to plot mesh similar to MATLAB
                # Subsample to match MATLAB 1:10:500
                step = 10
                H_sub = H[:500:step, :500:step]
                x = np.arange(0, 500, step)
                y = np.arange(0, 500, step)
                X, Y = np.meshgrid(x, y, indexing="ij")

                # Wireframe only (no filled faces, no shading)
                ax.plot_wireframe(X, Y, H_sub, color="black", linewidth=0.4)
                ax.view_init(elev=45, azim=45)  # Adjust view
                ax.axis("off")

            print("Running Figure430 (Highpass Filtering)...")

            # Image loading
            img_name = dip_data("blown_ic_crop.tif")
            f = imread(img_name)
            if f.ndim == 3:
                f = f[:, :, 0]

            f = img_as_float(f)
            M, N = f.shape

            # Mesh Lowpass for display (500x500, D0=50)
            # HLPDisp = lpFilterTF4e('gaussian', 500, 500, 50)
            HLPDisp = lpFilterTF4e("gaussian", 500, 500, 50)

            # Mesh Highpass
            HHPDisp = hpFilterTF4e("gaussian", 500, 500, 50)

            # Filtering
            # Lowpass: D0=10, 2M x 2N
            # HLP = lpFilterTF4e('gaussian', 2*M, 2*N, 10)
            HLP = lpFilterTF4e("gaussian", 2 * M, 2 * N, 10)
            glow = dftFiltering4e(f, HLP)

            # Highpass: D0=10
            # HHP = hpFilterTF4e('gaussian', 2*M, 2*N, 10)
            HHP = hpFilterTF4e("gaussian", 2 * M, 2 * N, 10)
            ghigh = dftFiltering4e(f, HHP)

            # High Frequency Emphasis
            # Hemphasis = 0.85 + hpFilterTF4e('gaussian', 2*M, 2*N, 10)
            Hemphasis = 0.85 + hpFilterTF4e("gaussian", 2 * M, 2 * N, 10)
            gemph = dftFiltering4e(f, Hemphasis)

            # Display
            fig = plt.figure(figsize=(15, 10))

            # 1. HLP Mesh
            ax1 = fig.add_subplot(2, 3, 1, projection="3d")
            mesh_plot(ax1, HLPDisp)
            ax1.set_title("Gaussian LP Transfer Function")

            # 2. HHP Mesh
            ax2 = fig.add_subplot(2, 3, 2, projection="3d")
            mesh_plot(ax2, HHPDisp)
            ax2.set_title("Gaussian HP Transfer Function")

            # 3. Duplicate HHP Mesh? Figure430.m has duplicate? "subplot(2,3,3) mesh(HHPDisp)"
            # Ah, maybe it was meant for Emphasis filter or just repeated?
            # Original code: subplot(2,3,3) mesh(HHPDisp).
            # I will replicate.
            ax3 = fig.add_subplot(2, 3, 3, projection="3d")
            mesh_plot(ax3, HHPDisp)
            ax3.set_title("Gaussian HP (Repeated)")

            # 4. Lowpass result
            ax4 = fig.add_subplot(2, 3, 4)
            ax4.imshow(glow, cmap="gray", vmin=0, vmax=1)
            ax4.set_title("Lowpass Filtered")
            ax4.axis("off")

            # 5. Highpass result
            ax5 = fig.add_subplot(2, 3, 5)
            ax5.imshow(ghigh, cmap="gray", vmin=0, vmax=1)
            ax5.set_title("Highpass Filtered")
            ax5.axis("off")

            # 6. Emphasis result
            ax6 = fig.add_subplot(2, 3, 6)
            ax6.imshow(gemph, cmap="gray", vmin=0, vmax=1)
            ax6.set_title("High Freq Emphasis")
            ax6.axis("off")

            plt.tight_layout()
            plt.savefig("Figure430.png")
            print("Saved Figure430.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure433(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure433.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt

            # %% Figure 4.33
            # 1-D example of ringing as a result of spatial padding.

            # Data
            HIdeal = np.zeros(256, dtype=float)
            # MATLAB: HIdeal(125:131)=1; (1-based inclusive)
            HIdeal[124:131] = 1.0

            # Compute impulse response
            Hm = np.fft.fftshift(HIdeal)

            h = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(HIdeal)))
            H = np.fft.fft(h)

            # Embed h in zeros
            hpad = np.zeros(512, dtype=float)
            # MATLAB: hpad(129:384) = real(h(1:256));
            hpad[128:384] = np.real(h[:256])
            Hpad = np.fft.fft(hpad)

            # Display
            plt.figure(figsize=(10, 8))

            plt.subplot(2, 2, 1)
            plt.plot(HIdeal)
            plt.title("H_ideal")
            plt.xlim(0, len(HIdeal) - 1)

            plt.subplot(2, 2, 2)
            plt.plot(hpad)
            plt.xlim(0, len(hpad) - 1)

            plt.subplot(2, 2, 3)
            plt.plot(np.real(h))
            plt.xlim(0, len(h) - 1)

            plt.subplot(2, 2, 4)
            plt.plot(np.abs(np.fft.fftshift(Hpad)))
            plt.xlim(0, len(Hpad) - 1)

            plt.tight_layout()
            plt.savefig("Figure433.png")
            print("Saved Figure433.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure434(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure434.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from helpers.data_path import dip_data

            print("Running Figure434 (Phase Manipulation)...")

            # Image loading
            img_path = dip_data("integrated-ckt-damaged.tif")

            f = imread(img_path)
            if f.ndim == 3:
                f = f[:, :, 0]

            f = img_as_float(f)

            # Process
            # F = fft2(f);
            F = np.fft.fft2(f)

            # a = angle(F);
            a = np.angle(F)

            # Alter phase angle
            # g1 = real (ifft2(abs(F).*exp(-i*a))); % Negative of the phase.
            # Python complex 'j'.
            g1 = np.real(np.fft.ifft2(np.abs(F) * np.exp(-1j * a)))

            # g2 = real (ifft2(abs(F).*exp(i*(0.25)*a))); % Phase times a constant.
            g2 = np.real(np.fft.ifft2(np.abs(F) * np.exp(1j * 0.25 * a)))

            # Display
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Original
            axes[0].imshow(f, cmap="gray", vmin=0, vmax=1)
            axes[0].set_title("Original")
            axes[0].axis("off")

            # g1 (Negative Phase)
            axes[1].imshow(g1, cmap="gray", vmin=0, vmax=1)
            axes[1].set_title("Phase Angle * -1")
            axes[1].axis("off")

            # g2 (Phase * 0.25)
            axes[2].imshow(g2, cmap="gray", vmin=0, vmax=1)
            axes[2].set_title("Phase Angle * 0.25")
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig("Figure434.png")
            print("Saved Figure434.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure435(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure435.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float

            from helpers.lpfilter import lpfilter
            from helpers.dftfilt import dftfilt
            from helpers.paddedsize import paddedsize
            from helpers.data_path import dip_data

            print("Running Figure435 (Gaussian Lowpass Filtering Steps)...")

            # Parameters
            D0 = 25

            # Data Loading
            img_path = dip_data("blown_ic_crop.tif")

            f = imread(img_path)
            if f.ndim == 3:
                f = f[:, :, 0]
            f = img_as_float(f)
            M, N = f.shape

            # Padding
            # PQ = paddedsize(size(f))
            PQ = paddedsize(f.shape)
            # fp = padarray(f, [PQ(1)-M, PQ(2)-N], 'post') - defaults to zero padding
            pad_h = PQ[0] - M
            pad_w = PQ[1] - N
            fp = np.pad(f, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)

            # Fourier Transform of padded image
            # FP = fft2(fp)
            FP = np.fft.fft2(fp)

            # Filter Generation
            # H = lpfilter('gaussian', PQ(1), PQ(2), D0)
            # lpfilter returns centered or uncentered?
            # My implementation of lpfilter uses dftuv which uses uncentered coordinates (0 to M-1 wrapped).
            # So lpfilter returns UNCENTERED H (corners are DC).
            H = lpfilter("gaussian", PQ[0], PQ[1], D0)

            # Filtering
            # G = H .* FP
            G = H * FP

            # gp = real(ifft2(G))
            gp = np.real(np.fft.ifft2(G))

            # Crop
            # g = gp(1:M, 1:N)
            g = gp[:M, :N]

            # Compare with dftfilt
            # g1 = dftfilt(f, H, 'symmetric')
            # Wait, MATLAB script says `dftfilt(f, H, 'symmetric')`.
            # `dftfilt` in MATLAB takes H. H is generated with PQ size (padded size).
            # `dftfilt` logic: checks if H is 2M/2N or similar?
            # Actually dftfilt pads f to size(H).
            # H is PQ (padded size). f is M,N.
            # dftfilt will pad f to PQ using 'symmetric' (reflection).
            # But wait, step `fp` above used Zero padding ('post').
            # So `g` comes from Zero padded input.
            # `g1` comes from Symmetric padded input.
            # So `g` and `g1` will differ at boundaries.
            # I will replicate this exactly.

            g1 = dftfilt(
                f, H, "symmetric"
            )  # 'symmetric' -> mode='reflect' in my implementation

            # Display
            # Preparing spectra for display (Log magnitude, shifted)
            FP_shifted = np.fft.fftshift(FP)
            S_FP = np.log(1 + np.abs(FP_shifted))

            H_shifted = np.fft.fftshift(H)

            G_shifted = np.fft.fftshift(G)
            S_G = np.log(1 + np.abs(G_shifted))

            fig1, axes1 = plt.subplots(2, 2, figsize=(10, 10))

            axes1[0, 0].imshow(f, cmap="gray")
            axes1[0, 0].set_title(f"f, Size = {f.shape}")
            axes1[0, 0].axis("off")

            axes1[0, 1].imshow(fp, cmap="gray")
            axes1[0, 1].set_title(f"fp (padded), Size = {fp.shape}")
            axes1[0, 1].axis("off")

            axes1[1, 0].imshow(S_FP, cmap="gray")
            axes1[1, 0].set_title(f"DFT(fp), Size = {FP.shape}")
            axes1[1, 0].axis("off")

            axes1[1, 1].imshow(H_shifted, cmap="gray")
            axes1[1, 1].set_title(f"H (Gaussian), Size = {H.shape}, D0={D0}")
            axes1[1, 1].axis("off")

            plt.tight_layout()
            plt.savefig("Figure435_1.png")

            fig2, axes2 = plt.subplots(2, 2, figsize=(10, 10))

            axes2[0, 0].imshow(S_G, cmap="gray")
            axes2[0, 0].set_title(f"DFT(fp) * H, Size = {G.shape}")
            axes2[0, 0].axis("off")

            axes2[0, 1].imshow(gp, cmap="gray")
            axes2[0, 1].set_title(f"gp (padded result), Size = {gp.shape}")
            axes2[0, 1].axis("off")

            axes2[1, 0].imshow(g, cmap="gray")
            axes2[1, 0].set_title(f"g (cropped), Size = {g.shape}")
            axes2[1, 0].axis("off")

            axes2[1, 1].imshow(g1, cmap="gray")
            axes2[1, 1].set_title("g using dftfilt (symmetric pad)")
            axes2[1, 1].axis("off")

            plt.tight_layout()
            plt.savefig("Figure435_2.png")

            print("Saved Figure435_1.png and Figure435_2.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure437(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure437.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.data_path import dip_data

            # Data
            img_path = dip_data("building-600by600.tif")
            f = imread(img_path)

            # Fourier transform
            F = np.fft.fft2(f)

            # Display
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(f, cmap="gray")
            axes[0].set_title("f")
            axes[0].axis("off")

            spec = np.fft.fftshift(np.log(1 + np.abs(F)))
            spec = spec - spec.min()
            if spec.max() > 0:
                spec = spec / spec.max()
            axes[1].imshow(spec, cmap="gray", vmin=0, vmax=1)
            axes[1].set_title(f"DFT(f), Size = {F.shape}")
            axes[1].axis("off")

            plt.tight_layout()
            plt.savefig("Figure437.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure438(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure438.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            from skimage.io import imread
            from skimage.util import img_as_float
            from scipy.ndimage import correlate
            from helpers.paddedsize import paddedsize
            from helpers.dftfilt import dftfilt
            from helpers.freqz2_equal import freqz2_equal
            from helpers.data_path import dip_data

            # Data
            img_path = dip_data("building-600by600.tif")
            f = img_as_float(imread(img_path))
            NR, NC = f.shape

            # Padding
            PQ = paddedsize(f.shape)

            # Fourier transform
            F = np.fft.fft2(f)

            # Impulse response (Sobel) and frequency response
            h = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float)

            # MATLAB freqz2 (equal spacing) implementation
            H = freqz2_equal(h, PQ[0], PQ[1])
            H_disp = H

            # Filtering in the spatial domain (imfilter default is correlation)
            gs = correlate(f, h, mode="constant", cval=0.0)

            # Filtering in the frequency domain
            H1 = np.fft.ifftshift(H_disp)
            gf = dftfilt(f, H1)

            # Display helpers (MATLAB imshow(..., []) autoscale)
            def autoscale(img: Any):
                """autoscale."""
                img = np.asarray(img, dtype=float)
                img = img - img.min()
                maxv = img.max()
                if maxv > 0:
                    img = img / maxv
                return img

            fig = plt.figure(figsize=(10, 8))

            # 3D surface of imag(H) subsampled
            ax1 = fig.add_subplot(2, 2, 1, projection="3d")
            step = 25
            H_sub = np.imag(H_disp)[::step, ::step]
            x = np.arange(0, H.shape[1], step)
            y = np.arange(0, H.shape[0], step)
            X, Y = np.meshgrid(x, y, indexing="ij")
            ax1.plot_surface(X, Y, H_sub, cmap="gray", linewidth=0, antialiased=False)
            ax1.view_init(elev=24, azim=146)
            ax1.set_axis_off()

            # imag(H)
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.imshow(autoscale(np.imag(H_disp)), cmap="gray")
            ax2.set_title("H(f_x, f_y)")
            ax2.axis("off")

            # Spatial domain
            ax3 = fig.add_subplot(2, 2, 3)
            ax3.imshow(autoscale(gs), cmap="gray")
            ax3.set_title("filtering in the spatial domain")
            ax3.axis("off")

            # Frequency domain
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.imshow(autoscale(gf), cmap="gray")
            ax4.set_title("filtering in the frequency domain")
            ax4.axis("off")

            plt.tight_layout()
            plt.savefig("Figure438.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure439(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure439.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            from DIP4eFigures.lpFilterTF4e import lpFilterTF4e

            # Generate transfer function
            meshILPF = lpFilterTF4e("ideal", 40, 40, 6)

            # Display
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(1, 1, 1, projection="3d")

            # MATLAB mesh: wireframe only
            X, Y = np.meshgrid(
                np.arange(meshILPF.shape[1]), np.arange(meshILPF.shape[0])
            )
            ax.plot_wireframe(X, Y, meshILPF, color="black", linewidth=0.4)
            ax.set_axis_off()

            plt.tight_layout()
            plt.savefig("Figure439.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure440(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure440.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from DIP4eFigures.intScaling4e import intScaling4e
            from DIP4eFigures.lpFilterTF4e import lpFilterTF4e
            from helpers.data_path import dip_data

            # Data
            img_path = dip_data("characterTestPattern688.tif")
            f = img_as_float(imread(img_path))
            M, N = f.shape

            # Compute fft
            P = 2 * M
            Q = 2 * N
            F = np.fft.fft2(f, s=(P, Q))
            Power = np.abs(F) ** 2
            Power_shift = np.fft.fftshift(Power)
            PT = np.sum(Power)

            # Energy enclosed by radii
            radii = [10, 30, 60, 160, 460]
            E = []
            for k in radii:
                H = lpFilterTF4e("ideal", P, Q, k)
                prod = H * Power_shift
                E.append(np.sum(prod) / PT)
            E = np.array(E)

            # Build circle mask
            Y, X = np.indices((P, Q))
            cy = P // 2
            cx = Q // 2
            C = np.zeros((P, Q), dtype=bool)
            for k in radii:
                dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
                C |= np.abs(dist - k) <= 1  # thickness ~2

            # Display spectrum with circles
            S = np.fft.fftshift(np.log(1 + np.abs(F)))
            S = intScaling4e(S)
            S[C] = 1

            # Reduce size 50%
            S = S[::2, ::2]

            # Display
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(f, cmap="gray")
            axes[0].axis("off")

            energy_str = np.array2string(E, precision=2, floatmode="fixed")
            axes[1].imshow(S, cmap="gray", vmin=0, vmax=1)
            axes[1].set_title(f"Energy = {energy_str}")
            axes[1].axis("off")

            plt.tight_layout()
            plt.savefig("Figure440.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure441(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure441.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from DIP4eFigures.lpFilterTF4e import lpFilterTF4e
            from DIP4eFigures.dftFiltering4e import dftFiltering4e
            from helpers.data_path import dip_data

            # Parameters
            D0 = [10, 30, 60, 160, 460]

            # Data
            img_path = dip_data("characterTestPattern688.tif")
            f = img_as_float(imread(img_path))
            M, N = f.shape
            P = 2 * M
            Q = 2 * N

            # Process
            results = []
            for d0 in D0:
                # lpFilterTF4e returns a centered filter; dftFiltering4e expects centered H
                H = lpFilterTF4e("ideal", P, Q, d0)
                results.append(dftFiltering4e(f, H))

            # Display
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            axes[0, 0].imshow(f, cmap="gray")
            axes[0, 0].set_title("Original")
            axes[0, 0].axis("off")

            for idx, d0 in enumerate(D0):
                ax = axes.flat[idx + 1]
                ax.imshow(results[idx], cmap="gray")
                ax.set_title(f"D0 = {d0}")
                ax.axis("off")

            plt.tight_layout()
            plt.savefig("Figure441.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure442(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure442.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from helpers.lpfilter import lpfilter

            # Transfer function
            H = lpfilter("ideal", 1000, 1000, 30)
            M, N = H.shape

            # Impulse response
            h = np.fft.fftshift(np.fft.ifft2(H))
            h = np.real(h)
            profile = h[M // 2, :]

            # Display
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(np.fft.fftshift(H), cmap="gray")
            axes[0].set_title("H")
            axes[0].axis("off")

            axes[1].imshow(h, cmap="gray")
            axes[1].set_title("h")
            axes[1].axis("off")

            axes[2].plot(profile)
            axes[2].set_title("profile (h)")
            # axes[2].set_aspect('equal', adjustable='box')
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig("Figure442.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure443(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure443.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            from helpers.lpfilter import lpfilter

            # Parameters
            D0 = [10, 20, 40, 60]

            # Generate transfer functions
            GLPF40 = np.fft.fftshift(lpfilter("gaussian", 600, 600, 40))
            meshGLPF40 = np.fft.fftshift(lpfilter("gaussian", 40, 40, 4))

            # Profile extraction (improfile along x=301:600, y=300)
            Profile = []
            for d0 in D0:
                H = np.fft.fftshift(lpfilter("gaussian", 600, 600, d0))
                # MATLAB indices 301:600 (1-based) -> Python 300:600 (0-based)
                line = H[299, 300:600]
                Profile.append(line)
            Profile = np.vstack(Profile).T  # columns correspond to D0 values

            # Display
            fig = plt.figure(figsize=(12, 4))

            # Mesh
            ax1 = fig.add_subplot(1, 3, 1, projection="3d")
            X, Y = np.meshgrid(
                np.arange(meshGLPF40.shape[1]), np.arange(meshGLPF40.shape[0])
            )
            ax1.plot_wireframe(X, Y, meshGLPF40, color="black", linewidth=0.4)
            ax1.set_axis_off()
            ax1.set_box_aspect((1, 1, 0.5))

            # GLPF40 image
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.imshow(GLPF40, cmap="gray")
            ax2.axis("off")

            # Profile plot
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.plot(Profile)
            # ax3.set_aspect('equal', adjustable='box')
            ax3.legend([str(d0) for d0 in D0])

            plt.tight_layout()
            plt.savefig("Figure443.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure444(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure444.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from helpers.lpfilter import lpfilter
            from DIP4eFigures.dftFiltering4e import dftFiltering4e
            from helpers.data_path import dip_data

            # Parameters
            D0 = [10, 30, 60, 160, 460]

            # Data
            img_path = dip_data("characterTestPattern688.tif")
            f = img_as_float(imread(img_path))
            M, N = f.shape
            P = 2 * M
            Q = 2 * N

            # Process
            results = []
            for d0 in D0:
                H = np.fft.fftshift(lpfilter("gaussian", P, Q, d0))
                results.append(dftFiltering4e(f, H))

            # Display
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            axes[0, 0].imshow(f, cmap="gray")
            axes[0, 0].set_title("Original")
            axes[0, 0].axis("off")

            for idx, d0 in enumerate(D0):
                ax = axes.flat[idx + 1]
                ax.imshow(results[idx], cmap="gray")
                ax.set_title(f"D0 = {d0}")
                ax.axis("off")

            plt.tight_layout()
            plt.savefig("Figure444.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure445(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure445.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            from helpers.lpfilter import lpfilter

            # Parameters
            D0 = [10, 20, 40, 60]

            # Generate transfer functions
            BWLP = np.fft.fftshift(lpfilter("butterworth", 40, 40, 5, 2))
            BWLPimage = np.fft.fftshift(lpfilter("butterworth", 1024, 1024, 128, 1))

            # Profile extraction (improfile along x=301:600, y=300)
            Profile = []
            for d0 in D0:
                H = np.fft.fftshift(lpfilter("butterworth", 600, 600, d0))
                line = H[299, 300:600]
                Profile.append(line)
            Profile = np.vstack(Profile).T

            # Display
            fig = plt.figure(figsize=(12, 4))

            # Mesh
            ax1 = fig.add_subplot(1, 3, 1, projection="3d")
            X, Y = np.meshgrid(np.arange(BWLP.shape[1]), np.arange(BWLP.shape[0]))
            ax1.plot_wireframe(X, Y, BWLP, color="black", linewidth=0.4)
            ax1.set_axis_off()
            ax1.set_box_aspect((1, 1, 0.5))

            # BWLP image
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.imshow(BWLPimage, cmap="gray")
            ax2.axis("off")

            # Profile plot
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.plot(Profile)
            # ax3.set_aspect('equal', adjustable='box')
            ax3.legend([str(d0) for d0 in D0])

            plt.tight_layout()
            plt.savefig("Figure445.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure446(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure446.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from helpers.lpfilter import lpfilter
            from DIP4eFigures.dftFiltering4e import dftFiltering4e
            from helpers.data_path import dip_data

            # Parameters
            D0 = [10, 30, 60, 160, 460]

            # Data
            img_path = dip_data("characterTestPattern688.tif")
            f = img_as_float(imread(img_path))
            M, N = f.shape
            P = 2 * M
            Q = 2 * N

            # Process
            results = []
            for d0 in D0:
                H = np.fft.fftshift(lpfilter("butterworth", P, Q, d0))
                results.append(dftFiltering4e(f, H))

            # Display
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            axes[0, 0].imshow(f, cmap="gray")
            axes[0, 0].set_title("Original")
            axes[0, 0].axis("off")

            for idx, d0 in enumerate(D0):
                ax = axes.flat[idx + 1]
                ax.imshow(results[idx], cmap="gray")
                ax.set_title(f"D0 = {d0}")
                ax.axis("off")

            plt.tight_layout()
            plt.savefig("Figure446.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure447(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure447.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from helpers.lpfilter import lpfilter
            from DIP4eFigures.intScaling4e import intScaling4e

            # Parameters
            D0 = 5
            n_vals = [1, 2, 5, 20]

            hs1 = []
            profiles = []

            for n in n_vals:
                H = lpfilter("butterworth", 1000, 1000, D0, n)
                M, N = H.shape
                h1 = np.real(np.fft.fftshift(np.fft.ifft2(H)))
                hs1.append(intScaling4e(h1))
                profiles.append(h1[M // 2, :])

            profiles = np.vstack(profiles).T

            # Display
            fig = plt.figure(figsize=(12, 6))

            for idx, n in enumerate(n_vals):
                ax = fig.add_subplot(2, 4, idx + 1)
                ax.imshow(hs1[idx], cmap="gray")
                ax.axis("off")

                axp = fig.add_subplot(2, 4, idx + 5)
                axp.plot(profiles[:, idx])
                # axp.set_aspect('equal', adjustable='box')
                axp.axis("off")

            plt.tight_layout()
            plt.savefig("Figure447.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure448(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure448.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from helpers.lpfilter import lpfilter
            from DIP4eFigures.dftFiltering4e import dftFiltering4e
            from DIP4eFigures.intScaling4e import intScaling4e
            from helpers.data_path import dip_data

            # Data
            img_path = dip_data("text_gaps_of_1_and_2_pixels.tif")
            f = img_as_float(imread(img_path))
            M, N = f.shape
            P = 2 * M
            Q = 2 * N

            # Filter design in the frequency domain
            H = np.fft.fftshift(lpfilter("gaussian", P, Q, 120))

            # Filtering in the frequency domain
            g = dftFiltering4e(f, H)
            gs = intScaling4e(g)

            # Display
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(f, cmap="gray")
            axes[0].axis("off")

            axes[1].imshow(gs, cmap="gray")
            axes[1].axis("off")

            plt.tight_layout()
            plt.savefig("Figure448.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure449(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure449.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from helpers.lpfilter import lpfilter
            from DIP4eFigures.dftFiltering4e import dftFiltering4e
            from helpers.data_path import dip_data

            # Data
            img_path = dip_data("woman512x512.tif")
            f = img_as_float(imread(img_path))
            M, N = f.shape
            P = 2 * M
            Q = 2 * N

            # MATLAB imcrop([256,217,107,73]) is [x, y, width, height] with 1-based
            # Python slices: rows y:(y+height), cols x:(x+width)
            x, y, w, h = 256, 217, 107, 73
            fCrop = f[y : y + h, x : x + w]

            # Filter design
            H150 = np.fft.fftshift(lpfilter("gaussian", P, Q, 150))
            H130 = np.fft.fftshift(lpfilter("gaussian", P, Q, 130))

            # Filtering
            g150 = dftFiltering4e(f, H150)
            g150Crop = g150[y : y + h, x : x + w]

            g130 = dftFiltering4e(f, H130)
            g130Crop = g130[y : y + h, x : x + w]

            # Display
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            axes[0, 0].imshow(f, cmap="gray")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(g150, cmap="gray")
            axes[0, 1].axis("off")

            axes[0, 2].imshow(g130, cmap="gray")
            axes[0, 2].axis("off")

            axes[1, 0].imshow(fCrop, cmap="gray")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(g150Crop, cmap="gray")
            axes[1, 1].axis("off")

            axes[1, 2].imshow(g130Crop, cmap="gray")
            axes[1, 2].axis("off")

            plt.tight_layout()
            plt.savefig("Figure449.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure450(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure450.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from helpers.lpfilter import lpfilter
            from DIP4eFigures.dftFiltering4e import dftFiltering4e
            from helpers.data_path import dip_data

            # Data
            img_path = dip_data("satellite_original.tif")
            f = img_as_float(imread(img_path))
            M, N = f.shape
            P = 2 * M
            Q = 2 * N

            # Filter design
            H1 = np.fft.ifftshift(lpfilter("gaussian", P, Q, 50))
            H2 = np.fft.ifftshift(lpfilter("gaussian", P, Q, 20))

            # Filtering
            g50 = dftFiltering4e(f, H1)
            g20 = dftFiltering4e(f, H2)

            # Display
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(f, cmap="gray")
            axes[0].axis("off")

            axes[1].imshow(g50, cmap="gray")
            axes[1].axis("off")

            axes[2].imshow(g20, cmap="gray")
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig("Figure450.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure451(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure451.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            from helpers.hpfilter import hpfilter

            # IDEAL HIGHPASS
            meshIHPF = np.fft.fftshift(hpfilter("ideal", 40, 40, 6))
            IHPF = np.fft.fftshift(hpfilter("ideal", 600, 600, 100))

            # GAUSSIAN HIGHPASS
            meshGHPF = np.fft.fftshift(hpfilter("gaussian", 40, 40, 4))
            GHPF = np.fft.fftshift(hpfilter("gaussian", 600, 600, 100))

            # BUTTERWORTH HIGHPASS
            meshBHPF = np.fft.fftshift(hpfilter("butterworth", 40, 40, 4, 2))
            BHPF = np.fft.fftshift(hpfilter("butterworth", 600, 600, 100, 2))

            # Display
            fig = plt.figure(figsize=(12, 10))

            # Mesh plots
            ax1 = fig.add_subplot(3, 3, 1, projection="3d")
            X, Y = np.meshgrid(
                np.arange(meshIHPF.shape[1]), np.arange(meshIHPF.shape[0])
            )
            ax1.plot_wireframe(X, Y, meshIHPF, color="black", linewidth=0.4)
            ax1.set_axis_off()

            ax4 = fig.add_subplot(3, 3, 4, projection="3d")
            X, Y = np.meshgrid(
                np.arange(meshGHPF.shape[1]), np.arange(meshGHPF.shape[0])
            )
            ax4.plot_wireframe(X, Y, meshGHPF, color="black", linewidth=0.4)
            ax4.set_axis_off()

            ax7 = fig.add_subplot(3, 3, 7, projection="3d")
            X, Y = np.meshgrid(
                np.arange(meshBHPF.shape[1]), np.arange(meshBHPF.shape[0])
            )
            ax7.plot_wireframe(X, Y, meshBHPF, color="black", linewidth=0.4)
            ax7.set_axis_off()

            # Images
            ax2 = fig.add_subplot(3, 3, 2)
            ax2.imshow(IHPF, cmap="gray")
            ax2.axis("off")

            ax5 = fig.add_subplot(3, 3, 5)
            ax5.imshow(GHPF, cmap="gray")
            ax5.axis("off")

            ax8 = fig.add_subplot(3, 3, 8)
            ax8.imshow(BHPF, cmap="gray")
            ax8.axis("off")

            # Profiles
            ax3 = fig.add_subplot(3, 3, 3)
            ax3.plot(IHPF[299, 300:])

            ax6 = fig.add_subplot(3, 3, 6)
            ax6.plot(GHPF[299, 300:])

            ax9 = fig.add_subplot(3, 3, 9)
            ax9.plot(BHPF[299, 300:])

            plt.tight_layout()
            plt.savefig("Figure451.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure452(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure452.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from helpers.lpfilter import lpfilter
            from DIP4eFigures.intScaling4e import intScaling4e

            # Parameters
            M = 1000
            N = 1000
            D0 = 5
            n_vals = [1, 2, 5, 20]

            # Constant in the frequency domain. Its inverse will be an impulse.
            G = np.ones((M, N))
            imp = np.real(np.fft.fftshift(np.fft.ifft2(G)))

            hs1 = []
            profiles = []

            for n in n_vals:
                HLP = lpfilter("butterworth", M, N, D0, n)
                hLP = np.real(np.fft.fftshift(np.fft.ifft2(HLP)))
                hHP = imp - hLP / np.max(hLP)

                hs1.append(intScaling4e(hHP))
                profiles.append(hHP[M // 2, :])

            profiles = np.vstack(profiles).T

            # Display
            fig = plt.figure(figsize=(12, 6))

            for idx, n in enumerate(n_vals):
                ax = fig.add_subplot(2, 4, idx + 1)
                ax.imshow(hs1[idx], cmap="gray")
                ax.axis("off")

                axp = fig.add_subplot(2, 4, idx + 5)
                axp.plot(profiles[:, idx])
                # axp.set_aspect('equal', adjustable='box')
                axp.axis("off")

            plt.tight_layout()
            plt.savefig("Figure452.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure453(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure453.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from helpers.hpfilter import hpfilter
            from DIP4eFigures.dftFiltering4e import dftFiltering4e
            from DIP4eFigures.intScaling4e import intScaling4e
            from helpers.data_path import dip_data

            # Data
            img_path = dip_data("characterTestPattern688.tif")
            f = img_as_float(imread(img_path))
            M, N = f.shape
            P = 2 * M
            Q = 2 * N

            # Ideal High Pass
            H = np.fft.ifftshift(hpfilter("ideal", P, Q, 60))
            gI60 = dftFiltering4e(f, H)
            H = np.fft.ifftshift(hpfilter("ideal", P, Q, 160))
            gI160 = dftFiltering4e(f, H)
            gI160e = intScaling4e(gI160)

            # Gaussian High Pass
            H = np.fft.ifftshift(hpfilter("gaussian", P, Q, 60))
            gG60 = dftFiltering4e(f, H)
            H = np.fft.ifftshift(hpfilter("gaussian", P, Q, 160))
            gG160 = dftFiltering4e(f, H)
            gG160e = intScaling4e(gG160)

            # Butterworth High Pass
            H = np.fft.ifftshift(hpfilter("butterworth", P, Q, 60, 2))
            gB60 = dftFiltering4e(f, H)
            H = np.fft.ifftshift(hpfilter("butterworth", P, Q, 160, 2))
            gB160 = dftFiltering4e(f, H)
            gB160e = intScaling4e(gB160)

            # Display figure 1
            fig1, axes1 = plt.subplots(2, 3, figsize=(12, 8))
            axes1[0, 0].imshow(gI60, cmap="gray", vmin=0, vmax=1)
            axes1[0, 0].axis("off")

            axes1[0, 1].imshow(gG60, cmap="gray", vmin=0, vmax=1)
            axes1[0, 1].axis("off")

            axes1[0, 2].imshow(gB60, cmap="gray", vmin=0, vmax=1)
            axes1[0, 2].axis("off")

            axes1[1, 0].imshow(gI160, cmap="gray", vmin=0, vmax=1)
            axes1[1, 0].axis("off")

            axes1[1, 1].imshow(gG160, cmap="gray", vmin=0, vmax=1)
            axes1[1, 1].axis("off")

            axes1[1, 2].imshow(gB160, cmap="gray", vmin=0, vmax=1)
            axes1[1, 2].axis("off")

            plt.tight_layout()
            plt.savefig("Figure453.png")

            # Display figure 2
            fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
            axes2[0].imshow(gI160e, cmap="gray")
            axes2[0].axis("off")

            axes2[1].imshow(gG160e, cmap="gray")
            axes2[1].axis("off")

            axes2[2].imshow(gB160e, cmap="gray")
            axes2[2].axis("off")

            plt.tight_layout()
            plt.savefig("Figure454.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure455(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure455.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from helpers.hpfilter import hpfilter
            from DIP4eFigures.dftFiltering4e import dftFiltering4e
            from helpers.data_path import dip_data

            # Data
            img_path = dip_data("thumb-print.tif")
            f = img_as_float(imread(img_path))
            M, N = f.shape
            P = 2 * M
            Q = 2 * N

            # Filter design
            H = np.fft.ifftshift(hpfilter("butterworth", P, Q, 50, 4))

            # Filtering
            g = dftFiltering4e(f, H)

            # Thresholding
            gp = g >= 0

            # Display
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(f, cmap="gray")
            axes[0].axis("off")

            axes[1].imshow(g, cmap="gray")
            axes[1].axis("off")

            axes[2].imshow(gp, cmap="gray")
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig("Figure455.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure456(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure456.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from helpers.dftuv import dftuv
            from DIP4eFigures.dftFiltering4e import dftFiltering4e
            from helpers.data_path import dip_data

            # Data
            img_path = dip_data("blurry-moon.tif")
            f = img_as_float(imread(img_path))
            M, N = f.shape
            P = 2 * M
            Q = 2 * N

            # Transfer function
            U, V = dftuv(P, Q)
            H = -4 * (np.pi**2) * (U**2 + V**2)
            H = np.fft.fftshift(H)

            # Filtering
            glap = dftFiltering4e(f, H)

            # Scale Laplacian response
            glaps = glap / np.max(glap)

            # Sharpened image
            g = f - glaps

            # Display
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(f, cmap="gray")
            axes[0].axis("off")

            axes[1].imshow(g, cmap="gray")
            axes[1].axis("off")

            plt.tight_layout()
            plt.savefig("Figure456.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure457(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure457.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from skimage.exposure import equalize_hist
            from helpers.hpfilter import hpfilter
            from DIP4eFigures.dftFiltering4e import dftFiltering4e
            from DIP4eFigures.intScaling4e import intScaling4e
            from helpers.data_path import dip_data

            # Data
            img_path = dip_data("chestXray.tif")
            f = img_as_float(imread(img_path))
            M, N = f.shape
            P = 2 * M
            Q = 2 * N

            # High pass filter design
            H = np.fft.ifftshift(hpfilter("gaussian", P, Q, 70))

            # High frequency emphasis filter design
            Hemp = 0.5 + 0.75 * H

            # High pass filtering
            ghp = dftFiltering4e(f, H)
            ghps = intScaling4e(ghp)

            # High frequency emphasis filtering
            gemp = dftFiltering4e(f, Hemp)
            gemps = intScaling4e(gemp)

            # Histogram equalization (256 bins)
            geq = equalize_hist(gemp, nbins=256)

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes[0, 0].imshow(f, cmap="gray")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(ghps, cmap="gray")
            axes[0, 1].axis("off")

            axes[1, 0].imshow(gemps, cmap="gray")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(geq, cmap="gray")
            axes[1, 1].axis("off")

            plt.tight_layout()
            plt.savefig("Figure457.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure460(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure460.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import os
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from helpers.homomorphictf import homomorphictf
            from DIP4eFigures.dftFiltering4e import dftFiltering4e
            from DIP4eFigures.intScaling4e import intScaling4e
            from helpers.data_path import dip_data

            # Parameters
            GammaH = 3
            GammaL = 0.4
            c = 5
            D0 = 20

            # Data
            img_path = dip_data("PET-scan.tif")
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")

            f = img_as_float(imread(img_path))
            M, N = f.shape
            P = 2 * M
            Q = 2 * N

            # Homomorphic filter transfer function.
            H = homomorphictf(P, Q, GammaL, GammaH, c, D0)

            # H is not centered. Center it.
            H = np.fft.fftshift(H)

            # Use H to filter f.
            g = dftFiltering4e(f, H)
            g = intScaling4e(g)

            # Display
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(f, cmap="gray")
            axes[0].axis("off")

            axes[1].imshow(g, cmap="gray")
            axes[1].axis("off")

            plt.tight_layout()
            plt.savefig("Figure460.png", dpi=300, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure460tunnel(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure460Tunnel.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from helpers.paddedsize import paddedsize
            from helpers.dftuv import dftuv
            from helpers.lpfilter import lpfilter
            from helpers.data_path import dip_data

            # Parameters
            GammaH = 1.2
            GammaL = 0.1
            D0 = 5
            MaxDisp = 0.1

            # Data
            img_path = dip_data("tun.jpg")
            f = img_as_float(imread(img_path))
            if f.ndim == 3:
                f = f[:, :, 0]
            NR, NC = f.shape

            # Padding (post zeros)
            PQ = paddedsize(f.shape)
            pad_rows = PQ[0] - NR
            pad_cols = PQ[1] - NC
            fp = np.pad(f, ((0, pad_rows), (0, pad_cols)), mode="constant")

            # Filter design in the frequency domain
            U, V = dftuv(PQ[0], PQ[1])
            D = np.hypot(U, V)
            HLP = lpfilter("gaussian", PQ[0], PQ[1], D0)
            HHP = 1 - HLP
            H = (GammaH - GammaL) * HHP + GammaL
            Hc = np.fft.fftshift(H)

            # Filtering in the frequency domain
            fl = np.log(1 + fp)
            F = np.fft.fft2(fl)
            G = H * F
            gp = np.fft.ifft2(G)

            gp = np.real(np.exp(gp) - 1)

            # Crop to original size
            g = gp[:NR, :NC]

            # Display figure 1
            fig1, axes1 = plt.subplots(2, 2, figsize=(10, 8))
            axes1[0, 0].imshow(f, cmap="gray", vmin=0, vmax=1)
            axes1[0, 0].set_title("f")
            axes1[0, 0].axis("off")

            vmin = np.min(g)
            axes1[0, 1].imshow(g, cmap="gray", vmin=vmin, vmax=MaxDisp)
            axes1[0, 1].set_title(
                f"g = Homorphic filter (f), Dyn = [{vmin:.6g}, {MaxDisp}]"
            )
            axes1[0, 1].axis("off")

            axes1[1, 0].hist(f.ravel(), bins=256)
            axes1[1, 0].set_title("Hist(f)")

            axes1[1, 1].hist(g.ravel(), bins=256)
            axes1[1, 1].set_title("Hist(g)")
            axes1[1, 1].axvline(MaxDisp, color="black")

            plt.tight_layout()

            # Display figure 2
            fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))

            logF = np.log10(np.abs(F))
            finite = np.isfinite(logF)
            Min = np.min(logF[finite])
            Max = np.max(logF[finite])

            cmap = plt.cm.gray.copy()
            cmap.set_bad(color="black")

            axes2[0, 0].imshow(np.fft.fftshift(logF), cmap=cmap, vmin=Min, vmax=Max)
            axes2[0, 0].set_title("|F(u, v)|")
            axes2[0, 0].axis("off")

            axes2[0, 1].imshow(Hc, cmap="gray")
            axes2[0, 1].set_title("H(u, v)")
            axes2[0, 1].axis("off")

            logG = np.log10(np.abs(G))
            axes2[1, 0].imshow(np.fft.fftshift(logG), cmap=cmap, vmin=Min, vmax=Max)
            axes2[1, 0].set_title("|F(u, v)*H(u, v)|")
            axes2[1, 0].axis("off")

            axes2[1, 1].plot(Hc[Hc.shape[0] // 2, :])
            axes2[1, 1].axis("tight")

            plt.tight_layout()

            fig1.savefig("Figure460Tunnel_1.png")
            fig2.savefig("Figure460Tunnel_2.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure461(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure461.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt

            # Parameters
            M = 800
            W = 120
            C0 = 200

            # Process
            D = np.arange(0, M)

            # lowpass 1
            HL1 = np.exp(-(D**2 / (W**2)))

            # lowpass 2
            HL2 = np.exp(-(D**2 / (4 * W**2)))

            # highpass from lowpass
            Hhigh = 1 - HL2

            # Bandreject formed by sum of lowpass and highpass Gaussian filters
            H = HL1 + Hhigh

            # highpass with shifted 0-point to C0
            HhighS = 1 - np.exp(-((D - C0) ** 2) / (W**2))

            # Formula in book (avoid divide by zero at D=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                Hbook = 1 - np.exp(-(((D**2 - C0**2) / (D * W)) ** 2))

            # Display
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].plot(H)
            axes[0].set_box_aspect(1)
            axes[0].set_title("exp(-(D.^2/(W^2))) + (1 - exp(-(D.^2/(4*W^2))))")
            axes[0].axvline(C0, color="black")

            axes[1].plot(HhighS)
            axes[1].set_box_aspect(1)
            axes[1].set_title("(1 - exp(-(D - C0).^2/W^2))")
            axes[1].axvline(C0, color="black")

            axes[2].plot(Hbook)
            axes[2].set_box_aspect(1)
            axes[2].set_title("1 - exp(-((D.^2 - C0^2)./(D*W)).^2)")
            axes[2].axvline(C0, color="black")

            plt.tight_layout()
            plt.savefig("Figure461.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure462(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure462.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            from helpers.bandfilter import bandfilter

            # Parameters
            r = 12

            # Ideal
            H = np.fft.fftshift(bandfilter("ideal", "reject", 512, 512, 128, 60))
            H1 = H[::r, ::r]

            # Gaussian
            H = np.fft.fftshift(bandfilter("gaussian", "reject", 512, 512, 128, 60))
            H2 = H[::r, ::r]

            # Butterworth
            H = np.fft.fftshift(
                bandfilter("butterworth", "reject", 512, 512, 128, 60, 1)
            )
            H3 = H[::r, ::r]

            # Display figure 1 (mesh)
            fig1 = plt.figure(figsize=(12, 4))

            ax1 = fig1.add_subplot(1, 3, 1, projection="3d")
            X, Y = np.meshgrid(np.arange(H1.shape[1]), np.arange(H1.shape[0]))
            ax1.plot_wireframe(X, Y, H1, color="black", linewidth=0.4)
            ax1.set_axis_off()
            ax1.set_box_aspect((1, 1, 1))

            ax2 = fig1.add_subplot(1, 3, 2, projection="3d")
            X, Y = np.meshgrid(np.arange(H2.shape[1]), np.arange(H2.shape[0]))
            ax2.plot_wireframe(X, Y, H2, color="black", linewidth=0.4)
            ax2.set_axis_off()
            ax2.set_box_aspect((1, 1, 1))

            ax3 = fig1.add_subplot(1, 3, 3, projection="3d")
            X, Y = np.meshgrid(np.arange(H3.shape[1]), np.arange(H3.shape[0]))
            ax3.plot_wireframe(X, Y, H3, color="black", linewidth=0.4)
            ax3.set_axis_off()
            ax3.set_box_aspect((1, 1, 1))

            plt.tight_layout()
            plt.savefig("Figure462.png")

            # Display figure 2 (images)
            fig2, axes = plt.subplots(1, 3, figsize=(12, 4))

            # autoscale like imshow(..., [])
            def autoscale(img: Any):
                """autoscale."""
                img = np.asarray(img, dtype=float)
                img = img - img.min()
                maxv = img.max()
                if maxv > 0:
                    img = img / maxv
                return img

            axes[0].imshow(autoscale(H1), cmap="gray")
            axes[0].axis("off")

            axes[1].imshow(autoscale(H2), cmap="gray")
            axes[1].axis("off")

            axes[2].imshow(autoscale(H3), cmap="gray")
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig("Figure463.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure464(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure464.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from DIP4eFigures.intScaling4e import intScaling4e
            from helpers.cnotch import cnotch
            from helpers.dftfilt import dftfilt
            from helpers.data_path import dip_data

            # Data
            img_path = dip_data("car-moire-pattern.tif")
            f = img_as_float(imread(img_path))
            M, N = f.shape

            # DFT
            F = np.fft.fft2(f)
            S = intScaling4e(np.log10(1 + np.abs(np.fft.fftshift(F))))

            # Notch locations (from impixelinfo)
            C = np.array([[44, 54], [85, 56], [40, 112], [82, 112]])

            # Notch filter (uncentered)
            H = cnotch("butterworth", "reject", M, N, C, 9, 4)

            # Filtering
            P = intScaling4e(np.fft.fftshift(H) * img_as_float(S))
            g = np.real(dftfilt(f, H))
            g = img_as_float(g)

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            axes[0, 0].imshow(f, cmap="gray")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(S, cmap="gray")
            axes[0, 1].axis("off")

            axes[1, 0].imshow(P, cmap="gray")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(g, cmap="gray")
            axes[1, 1].axis("off")

            plt.tight_layout()
            plt.savefig("Figure464.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure465(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure465.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from DIP4eFigures.intScaling4e import intScaling4e
            from helpers.recnotch import recnotch
            from DIP4eFigures.dftFiltering4e import dftFiltering4e
            from helpers.data_path import dip_data

            # Data
            img_path = dip_data("Saturnringe.tif")
            f = img_as_float(imread(img_path))
            M, N = f.shape

            # Fourier transform
            F = np.fft.fft2(f)
            S = intScaling4e(np.log10(1 + np.abs(np.fft.fftshift(F))))

            # Filter design (centered)
            H = np.fft.fftshift(recnotch("reject", "vertical", M, N, 5, 15))

            # Filtering (no padding)
            g = dftFiltering4e(f, H, padmode="none")

            # Display (slightly gray)
            H_disp = 0.98 * H

            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            axes[0, 0].imshow(f, cmap="gray")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(S, cmap="gray")
            axes[0, 1].axis("off")

            axes[1, 0].imshow(H_disp, cmap="gray")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(g, cmap="gray")
            axes[1, 1].axis("off")

            plt.tight_layout()
            plt.savefig("Figure465.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure466(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter04 script `Figure466.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from DIP4eFigures.intScaling4e import intScaling4e
            from helpers.recnotch import recnotch
            from helpers.data_path import dip_data

            # Data
            img_path = dip_data("cassini-interference.tif")
            f = img_as_float(imread(img_path))
            M, N = f.shape

            # Fourier transform
            F = np.fft.fft2(f)

            # Filter design (uncentered)
            Hpass = recnotch("pass", "vertical", M, N, 5, 15)

            # Apply filter (uncentered)
            P = Hpass * F
            pattern = intScaling4e(np.real(np.fft.fftshift(np.fft.ifft2(P))))

            SP = intScaling4e(np.log10(1 + np.abs(np.fft.fftshift(P))))

            # Display
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].imshow(SP, cmap="gray")
            axes[0].axis("off")

            axes[1].imshow(pattern, cmap="gray")
            axes[1].axis("off")

            plt.tight_layout()
            plt.savefig("Figure466.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

