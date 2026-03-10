from pathlib import Path as _Path
import os as _os

from typing import Any


class Chapter06Mixin:
    def figure610(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter06 script `Figure610.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from libDIP.basisImage4e import basisImage4e
            from libDIP.tmat4e import tmat4e

            def MyDisp(
                Te: Any, Matrix: Any, N: Any, plots: Any, position: Any, Title: Any
            ):
                """MyDisp."""
                t = np.arange(0, N + Te, Te)

                for i in range(1, N + 1):
                    if i == 1:
                        f = np.sqrt(1.0 / N) * np.cos(
                            (2 * t + 1) * np.pi * (i - 1) / (2 * N)
                        )
                    else:
                        f = np.sqrt(2.0 / N) * np.cos(
                            (2 * t + 1) * np.pi * (i - 1) / (2 * N)
                        )

                    ax = plt.subplot(N, plots, (plots * i) - (plots - position))
                    markerline, stemlines, baseline = ax.stem(
                        np.arange(0, N), Matrix[i - 1, :], markerfmt="o", basefmt=" "
                    )

                    markerline.set_markeredgecolor("none")
                    markerline.set_markerfacecolor((0, 105 / 255, 166 / 255))
                    markerline.set_markersize(2.25 * 4 / 1.5)

                    # MATLAB sets stem line color to none; emulate by hiding stem lines.
                    stemlines.set_color("none")
                    stemlines.set_linewidth(0.5 * 0.5 / 0.75)

                    ax.plot(t, f)

                    ax.set_frame_on(False)
                    ax.axis("off")
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    if i == 1:
                        ax.set_title(Title)

            # Parameters
            N = 8
            P = 1
            position = 1
            plots = 2
            Te = 1e-3

            # Process
            S_COMPOSITE, S_DISPLAY = basisImage4e("DCT", 8, 1)
            DCT = tmat4e("DCT", N)

            # Display
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            MyDisp(Te, DCT, N, plots, position, "DCT")

            plt.subplot(1, 2, 2)
            plt.imshow(
                S_DISPLAY, cmap="gray", vmin=S_DISPLAY.min(), vmax=S_DISPLAY.max()
            )
            plt.axis("off")

            # Print to file
            plt.savefig("Figure610.png")
            print("Saved Figure610.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure612(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter06 script `Figure612.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from libDIP.tmat4e import tmat4e

            # %% Figure612

            # Parameters
            N = 4
            Te = 1e-3  # Sampling period

            # Data
            f = (np.array([0, 1, 2, 3], dtype=float) ** 2).reshape(-1, 1)

            # DCT Transform (MATLAB dctmtx)
            DCT = tmat4e("DCT", N)

            # Reconstruction
            t = np.arange(0, 4 * N + Te, Te)
            Rec = np.zeros((N, t.size), dtype=float)

            Theta = DCT @ f

            for i in range(1, N + 1):
                if i == 1:
                    Phi = np.sqrt(1.0 / N) * np.cos(
                        (2 * t + 1) * np.pi * (i - 1) / (2 * N)
                    )
                    Rec[i - 1, :] = Phi * Theta[i - 1, 0]
                else:
                    Phi = np.sqrt(2.0 / N) * np.cos(
                        (2 * t + 1) * np.pi * (i - 1) / (2 * N)
                    )
                    Rec[i - 1, :] = Rec[i - 2, :] + Phi * Theta[i - 1, 0]

            # Original digital signal
            tn = np.arange(0, N)

            # Display
            plt.figure(figsize=(8, 8))
            for iter_idx in range(1, N + 1):
                ax = plt.subplot(N, 1, iter_idx)
                ax.plot(t, Rec[iter_idx - 1, :])
                ax.stem(tn, f.ravel(), basefmt=" ")
                ax.set_xlim(t[0], t[-1])

            plt.tight_layout()
            plt.savefig("Figure612.png")
            print("Saved Figure612.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure613(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter06 script `Figure613.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from libDIP.basisImage4e import basisImage4e
            from libDIP.tmat4e import tmat4e

            def MyDisp(Matrix: Any, N: Any, plots: Any, position: Any, Title: Any):
                """MyDisp."""
                Factor = 1.0 / Matrix[0, 0]
                t = np.arange(0, N + 1 / 1000.0, 1 / 1000.0)

                for i in range(1, N + 1):
                    f = np.sqrt(2.0 / (N + 1)) * np.sin((t + 1.0) * i * np.pi / (N + 1))

                    ax = plt.subplot(N, plots, (plots * i) - (plots - position))
                    markerline, stemlines, baseline = ax.stem(
                        np.arange(0, N), Matrix[i - 1, :], markerfmt="o", basefmt=" "
                    )

                    markerline.set_markeredgecolor("none")
                    markerline.set_markerfacecolor((0, 105 / 255, 166 / 255))
                    markerline.set_markersize(2.25 * 4 / 1.5)

                    # MATLAB sets stem line color to none; emulate by hiding stem lines.
                    stemlines.set_color("none")
                    stemlines.set_linewidth(0.5 * 0.5 / 0.75)

                    ax.plot(t, f)

                    ax.set_frame_on(False)
                    ax.axis("off")
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    if i == 1:
                        ax.set_title(Title)

            # %% Figure 613

            # Parameters
            N = 8
            P = 1
            position = 1
            plots = 2

            # Process
            S_COMPOSITE, S_DISPLAY = basisImage4e("DST", 8, 1)
            DST = tmat4e("DST", N)

            # Display
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            MyDisp(DST, N, plots, position, "DST")

            plt.subplot(1, 2, 2)
            plt.imshow(
                S_DISPLAY, cmap="gray", vmin=S_DISPLAY.min(), vmax=S_DISPLAY.max()
            )
            plt.axis("off")

            # Print to file
            plt.savefig("Figure613.png")
            print("Saved Figure613.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure614(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter06 script `Figure614.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from libDIP.tmat4e import tmat4e

            # %% Figure614

            # Parameters
            N = 4
            Te = 1e-3  # Sampling period

            # Data
            f = (np.array([0, 1, 2, 3], dtype=float) ** 2).reshape(-1, 1)

            # DST Transform (MATLAB dstmtx)
            DST = tmat4e("DST", N)

            # Reconstruction
            t = np.arange(0, 4 * N + Te, Te)
            Rec = np.zeros((N, t.size), dtype=float)

            Theta = DST @ f

            for i in range(1, N + 1):
                Phi = np.sqrt(2.0 / (N + 1)) * np.sin((t + 1.0) * i * np.pi / (N + 1))
                if i == 1:
                    Rec[i - 1, :] = Phi * Theta[i - 1, 0]
                else:
                    Rec[i - 1, :] = Rec[i - 2, :] + Phi * Theta[i - 1, 0]

            Factor = f[-1, 0] / Rec[N - 1, 3000]

            # Original digital signal
            tn = np.arange(0, N)

            # Display
            plt.figure(figsize=(8, 8))
            for iter_idx in range(1, N + 1):
                ax = plt.subplot(N, 1, iter_idx)
                ax.plot(t, Factor * Rec[iter_idx - 1, :])
                ax.stem(tn, f.ravel(), basefmt=" ")
                ax.set_xlim(t[0], t[-1])

            plt.tight_layout()
            plt.savefig("Figure614.png")
            print("Saved Figure614.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure615(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter06 script `Figure615.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from scipy.fftpack import dct
            from libDIP.tmat4e import tmat4e
            from helpers.lpfilter import lpfilter
            from helpers.data_path import dip_data

            def dct2(a: Any):
                """dct2."""
                return dct(dct(a.T, norm="ortho").T, norm="ortho")

            # Parameters
            D0 = 60

            # Data
            f = img_as_float(imread(dip_data("characterTestPattern688.tif")))
            M, N = f.shape
            P = 2 * M
            Q = 2 * N

            # Filter design in the Fourier domain
            H_noncenter = lpfilter("ideal", M, N, D0).astype(float)
            H_center = np.fft.fftshift(H_noncenter)

            # Fourier transform
            F_Fourier = np.fft.fft2(f)

            # By using matrix multiplication
            A_DFT = tmat4e("DFT", M)
            F1_Fourier = M * A_DFT @ f @ A_DFT.T
            Temp = (1 / M) * A_DFT.conj().T @ F1_Fourier @ A_DFT.conj()

            # Cosine transform
            F_DCT = dct2(f)
            A_DCT = tmat4e("DCT", M)
            F1_DCT = A_DCT @ f @ A_DCT.T

            # Sine transform
            A_DST = tmat4e("DST", M)
            F1_DST = A_DST @ f @ A_DST.T

            # Hartley transform
            A_DHT = tmat4e("DHT", M)
            F1_DHT = A_DHT @ f @ A_DHT.T

            # Filtering in the Fourier domain
            G_Fourier = np.fft.fftshift(F_Fourier) * H_center
            g_Fourier = np.real(np.fft.ifft2(np.fft.ifftshift(G_Fourier)))

            # Filtering in the Hartley domain
            G_Hartley = F1_DHT * H_noncenter
            g_Hartley = A_DHT.conj().T @ G_Hartley @ A_DHT.conj()

            # Filtering in the Cosine domain
            G_Cosine = F1_DCT * H_noncenter
            g_Cosine = A_DCT.conj().T @ G_Cosine @ A_DCT.conj()

            # Filtering in the Sine domain
            G_Sine = F1_DST * H_noncenter
            g_Sine = A_DST.conj().T @ G_Sine @ A_DST.conj()

            # Display
            fig, axes = plt.subplots(3, 3, figsize=(10, 10))
            axes[0, 0].imshow(f, cmap="gray")
            axes[0, 0].set_title("Original")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(np.log10(np.abs(np.fft.fftshift(F_Fourier))), cmap="gray")
            axes[0, 1].set_title("|F.Fourier|")
            axes[0, 1].axis("off")

            axes[0, 2].imshow(g_Fourier, cmap="gray")
            axes[0, 2].set_title("g_Fourier")
            axes[0, 2].axis("off")

            axes[1, 0].imshow(np.log10(np.abs(np.fft.fftshift(F1_DHT))), cmap="gray")
            axes[1, 0].set_title("|F.Hartley|")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(np.log10(np.abs(F1_DCT)), cmap="gray")
            axes[1, 1].set_title("|F.Cosine|")
            axes[1, 1].axis("off")

            axes[1, 2].imshow(np.log10(np.abs(F1_DST)), cmap="gray")
            axes[1, 2].set_title("|F.Sine|")
            axes[1, 2].axis("off")

            axes[2, 0].imshow(g_Hartley, cmap="gray")
            axes[2, 0].set_title("g_Hartley")
            axes[2, 0].axis("off")

            axes[2, 1].imshow(g_Cosine, cmap="gray")
            axes[2, 1].set_title("g_Cosine")
            axes[2, 1].axis("off")

            axes[2, 2].imshow(g_Sine, cmap="gray")
            axes[2, 2].set_title("g_Sine")
            axes[2, 2].axis("off")

            plt.tight_layout()
            plt.savefig("Figure615.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure616(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter06 script `Figure616.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy.linalg import hadamard
            from scipy.interpolate import interp1d

            from libDIP.basisImage4e import basisImage4e

            # Parameters
            N = 8
            P = 1
            position = 1
            plots = 2
            Te = 1e-3
            t = np.arange(1, N + Te, Te)

            # Walsh Hadamard Matrix (sequency ordered, like MATLAB whtmtx)
            HAD = hadamard(N)
            HadIdx = np.arange(N)
            M = int(np.log2(N)) + 1
            binHadIdx = np.array(
                [list(np.binary_repr(i, width=M)) for i in HadIdx], dtype=int
            )
            binHadIdx = np.fliplr(binHadIdx)
            binSeqIdx = np.zeros((N, M - 1), dtype=int)
            for k in range(M - 1, 0, -1):
                binSeqIdx[:, k - 1] = np.bitwise_xor(
                    binHadIdx[:, k], binHadIdx[:, k - 1]
                )
            SeqIdx = binSeqIdx.dot(2 ** np.arange(M - 2, -1, -1))
            WHT = HAD[SeqIdx, :]

            S_COMPOSITE, S_DISPLAY = basisImage4e("WHT", N, P)

            # Display
            plt.figure()
            plt.subplot(1, 2, 1)
            for i in range(N):
                ax = plt.subplot(N, plots, (plots * (i + 1)) - (plots - position))
                markerline, stemlines, baseline = ax.stem(
                    np.arange(1, N + 1), WHT[i, :], markerfmt="o", basefmt=" "
                )
                markerline.set_markeredgecolor("none")
                markerline.set_markerfacecolor((0, 105 / 255, 166 / 255))
                markerline.set_markersize(2.25 * 4 / 1.5)
                stemlines.set_linewidth(0.5 * 0.5 / 0.75)

                Temp = interp1d(
                    np.arange(1, N + 1),
                    WHT[i, :],
                    kind="previous",
                    bounds_error=False,
                    fill_value=(WHT[i, 0], WHT[i, -1]),
                )(t)
                ax.plot(t, Temp)

                ax.set_frame_on(False)
                ax.axis("off")
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                if i == 0:
                    ax.set_title("DCT")

            plt.subplot(1, 2, 2)
            plt.imshow(
                S_DISPLAY, cmap="gray", vmin=S_DISPLAY.min(), vmax=S_DISPLAY.max()
            )
            plt.axis("off")

            # Print to file
            plt.savefig("Figure616.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure617(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter06 script `Figure617.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy.interpolate import interp1d
            from libDIP.basisImage4e import basisImage4e

            # Parameters
            N = 8
            P = 1
            position = 1
            plots = 2
            LN = int(np.log2(N))
            Te = 1e-3
            t = np.arange(1, N + Te, Te)

            # Slant transform matrix
            a = 3 / np.sqrt(5)
            b = 1 / np.sqrt(5)
            sp = np.array(
                [[1, 1, 1, 1], [a, b, -b, -a], [1, -1, -1, 1], [b, -a, a, -b]],
                dtype=float,
            )

            for i in range(3, LN + 1):
                NN = 2**i
                aN = np.sqrt((3 * NN**2) / (4 * (NN**2 - 1)))
                bN = np.sqrt((NN**2 - 4) / (4 * (NN**2 - 1)))

                sr1 = np.array([[1, 0], [aN, bN]], dtype=float)
                sr2 = np.array([[1, 0], [-aN, bN]], dtype=float)
                sz = np.zeros((2, (NN - 4) // 2))
                sn1 = np.hstack([sr1, sz, sr2, sz])

                q = (NN // 2) - 2
                ir = np.eye(q)
                iz = np.zeros((q, 2))
                sn2 = np.hstack([iz, ir, iz, ir])
                sn4 = np.hstack([iz, ir, iz, -ir])

                sr1 = np.array([[0, 1], [-bN, aN]], dtype=float)
                sr2 = np.array([[0, -1], [bN, aN]], dtype=float)
                sn3 = np.hstack([sr1, sz, sr2, sz])

                sn = np.vstack([sn1, sn2, sn3, sn4])

                m2 = np.block([[sp, np.zeros_like(sp)], [np.zeros_like(sp), sp]])

                sp = sn @ m2

                SLANT = np.zeros_like(sp)
                for k in range(NN):
                    if k < 2:
                        seq = k
                    elif k <= NN // 2 - 1:
                        if k % 2 == 0:
                            seq = 2 * k
                        else:
                            seq = 2 * k + 1
                    elif k == NN // 2:
                        seq = 2
                    elif k == NN // 2 + 1:
                        seq = 3
                    else:
                        if k % 2 == 0:
                            seq = 2 * (k - NN // 2) + 1
                        else:
                            seq = 2 * (k - NN // 2)
                    SLANT[seq, :] = sp[k, :]
                sp = SLANT

            sp = sp / np.sqrt(N)
            SLANT = sp.copy()

            S_COMPOSITE, S_DISPLAY = basisImage4e("SLT", N, P)

            # Display
            plt.figure()
            plt.subplot(1, 2, 1)
            for i in range(N):
                ax = plt.subplot(N, plots, (plots * (i + 1)) - (plots - position))
                markerline, stemlines, baseline = ax.stem(
                    np.arange(1, N + 1), SLANT[i, :], markerfmt="o", basefmt=" "
                )
                markerline.set_markeredgecolor("none")
                markerline.set_markerfacecolor((0, 105 / 255, 166 / 255))
                markerline.set_markersize(2.25 * 4 / 1.5)
                stemlines.set_linewidth(0.5 * 0.5 / 0.75)

                Temp = interp1d(
                    np.arange(1, N + 1),
                    SLANT[i, :],
                    kind="previous",
                    bounds_error=False,
                    fill_value=(SLANT[i, 0], SLANT[i, -1]),
                )(t)
                ax.plot(t, Temp)

                ax.set_frame_on(False)
                ax.axis("off")
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                if i == 0:
                    ax.set_title("SLANT")

            plt.subplot(1, 2, 2)
            plt.imshow(
                S_DISPLAY, cmap="gray", vmin=S_DISPLAY.min(), vmax=S_DISPLAY.max()
            )
            plt.axis("off")

            plt.savefig("Figure617.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure618(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter06 script `Figure618.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy.interpolate import interp1d
            from helpers.wavedec import wavedec
            from libDIP.basisImage4e import basisImage4e

            # Parameters
            N = 8
            P = 1
            position = 1
            plots = 2
            LN = int(np.log2(N))
            I = np.eye(N)
            Te = 1e-3
            t = np.arange(1, N + Te, Te)

            # Compute Haar matrix
            HAAR = np.zeros((N, N))
            for i in range(N):
                HAAR[:, i], _ = wavedec(I[:, i], LN, "haar")

            S_COMPOSITE, S_DISPLAY = basisImage4e("HAAR", N, P)

            # Display
            plt.figure()
            plt.subplot(1, 2, 1)
            for i in range(N):
                ax = plt.subplot(N, plots, (plots * (i + 1)) - (plots - position))
                markerline, stemlines, baseline = ax.stem(
                    np.arange(1, N + 1), HAAR[i, :], markerfmt="o", basefmt=" "
                )
                markerline.set_markeredgecolor("none")
                markerline.set_markerfacecolor((0, 105 / 255, 166 / 255))
                markerline.set_markersize(2.25 * 4 / 1.5)
                stemlines.set_linewidth(0.5 * 0.5 / 0.75)

                Temp = interp1d(
                    np.arange(1, N + 1),
                    HAAR[i, :],
                    kind="previous",
                    bounds_error=False,
                    fill_value=(HAAR[i, 0], HAAR[i, -1]),
                )(t)
                ax.plot(t, Temp)

                ax.set_frame_on(False)
                ax.axis("off")
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                if i == 0:
                    ax.set_title("Haar")

            plt.subplot(1, 2, 2)
            plt.imshow(
                S_DISPLAY, cmap="gray", vmin=S_DISPLAY.min(), vmax=S_DISPLAY.max()
            )
            plt.axis("off")

            plt.savefig("Figure618.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure630(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter06 script `Figure630.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from helpers.wavefast import wavefast
            from helpers.wavedisplay import wavedisplay
            from helpers.data_path import dip_data

            # Data
            f = img_as_float(imread(dip_data("Vase.tif")))

            # Fast wavelet transform
            c, s = wavefast(f, 1, "haar")
            c2, s2 = wavefast(f, 2, "haar")
            c8, s8 = wavefast(f, 8, "haar")

            # Display
            plt.figure()
            plt.imshow(f, cmap="gray")
            plt.axis("off")
            plt.savefig("Figure630.png")

            plt.figure()
            w = wavedisplay(c, s)
            plt.imshow(w, cmap="gray")
            plt.axis("off")
            plt.savefig("Figure630Bis.png")

            plt.figure()
            w2 = wavedisplay(c2, s2)
            plt.imshow(w2, cmap="gray")
            plt.axis("off")
            plt.savefig("Figure630Ter.png")

            plt.figure()
            Temp = wavedisplay(c8, s8, 4)
            plt.imshow(Temp, cmap="gray")
            plt.axis("off")
            plt.savefig("Figure630Quart.png")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure631(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter06 script `Figure631.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt

            from libDIP.basisImage4e import basisImage4e
            from helpers.haarDWTbasisImage import haarDWTbasisImage

            # Parameters
            N = 8
            P = 1

            # Process
            S_COMPOSITE, S_DISPLAY = basisImage4e("HAAR", N, P)

            # Display
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(
                S_DISPLAY, cmap="gray", vmin=S_DISPLAY.min(), vmax=S_DISPLAY.max()
            )
            plt.title("HAAR basis functions")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            haarDWTbasisImage(3)
            plt.title("Basis images 3 scale 8x8")
            plt.axis("off")

            # Print to file
            plt.savefig("Figure631.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure632(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter06 script `Figure632.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from helpers.wavefast import wavefast
            from helpers.wavecut import wavecut
            from helpers.waveback import waveback
            from helpers.wavedisplay import wavedisplay
            from helpers.data_path import dip_data

            # Data
            f = img_as_float(imread(dip_data("sinePulses.tif")))

            # Wavelet transform
            c, s = wavefast(f, 2, "sym4")

            # Zeroing approximation coefficients
            nc, y = wavecut("a", c, s)

            # Back to image domain
            f1 = waveback(nc, s, "sym4")

            # Zeroing approximation and horizontal details coefficients
            nc1 = c.copy()
            Ix = np.where(nc == 0)[0]
            nc1[: 2 * len(Ix)] = 0

            # Back to image domain
            f2 = waveback(nc1, s, "sym4")

            # Display
            fig, axes = plt.subplots(2, 3, figsize=(10, 7))
            axes[0, 0].imshow(f, cmap="gray")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(wavedisplay(c, s, 3), cmap="gray")
            axes[0, 1].axis("off")

            axes[0, 2].imshow(wavedisplay(nc, s, 3), cmap="gray")
            axes[0, 2].axis("off")

            axes[1, 0].imshow(f1, cmap="gray")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(wavedisplay(nc1, s, 3), cmap="gray")
            axes[1, 1].axis("off")

            axes[1, 2].imshow(f2, cmap="gray")
            axes[1, 2].axis("off")

            plt.tight_layout()
            plt.savefig("Figure632.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure63bior31(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter06 script `Figure63BIOR31.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt

            from helpers.wavedec import wavedec
            from helpers.MyDisp import MyDisp

            # Parameters
            NR = 16
            NC = 2
            LN = int(np.log2(NR))
            I = np.eye(NR)

            # Biorthogonal wavelets
            BIOR31 = np.zeros((NR, NR))
            RBIO31 = np.zeros((NR, NR))

            for i in range(NR):
                BIOR31[:, i], _ = wavedec(I[:, i], LN, "bior3.1")
                RBIO31[:, i], _ = wavedec(I[:, i], LN, "rbio3.1")

            # Display
            plt.figure()
            position = 1
            Error_BIOR31 = MyDisp(BIOR31, NR, NC, position, "DFT real")
            position = 2
            Error_RBIO31 = MyDisp(RBIO31, NR, NC, position, "DFT real")

            # Print to file
            plt.savefig("Figure63BIOR31.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure63db4(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter06 script `Figure63DB4.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt

            from helpers.wavedec import wavedec
            from helpers.MyDisp import MyDisp

            # Parameters
            NR = 16
            NC = 1
            LN = int(np.log2(NR))
            I = np.eye(NR)

            # DWT with Daubechies 4 wavelets.
            DB4 = np.zeros((NR, NR))
            for i in range(NR):
                DB4[:, i], _ = wavedec(I[:, i], LN, "db4")

            # Display
            plt.figure()
            position = 1
            Error = MyDisp(DB4, NR, NC, position, "DFT real")

            # Print to file
            plt.savefig("Figure63DB4.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure63dct(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter06 script `Figure63DCT.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt

            from helpers.MyDisp import MyDisp

            def dctmtx(N: Any):
                """dctmtx."""
                k = np.arange(N).reshape(-1, 1)
                n = np.arange(N).reshape(1, -1)
                D = np.cos(np.pi * (2 * n + 1) * k / (2 * N))
                D[0, :] = D[0, :] / np.sqrt(2)
                return np.sqrt(2 / N) * D

            # Parameters
            NR = 16
            NC = 1

            # DCT matrix
            DCT = dctmtx(NR)

            # Display
            plt.figure()
            position = 1
            Error = MyDisp(DCT, NR, NC, position, "DCT")

            # Print to file
            plt.savefig("Figure63DCT.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure63dft(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter06 script `Figure63DFT.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt

            from helpers.MyDisp import MyDisp

            def dftmtx(N: Any):
                """dftmtx."""
                n = np.arange(N)
                k = n.reshape(-1, 1)
                W = np.exp(-2j * np.pi * k * n / N)
                return W

            # Parameters
            NR = 16
            NC = 2

            # DFT matrix (normalized)
            DFT = dftmtx(NR) / np.sqrt(NR)

            # Display
            plt.figure()
            position = 1
            Error_RealDFT = MyDisp(np.real(DFT), NR, NC, position, "DFT real")
            position = 2
            Error_ImagDFT = MyDisp(np.imag(DFT), NR, NC, position, "DFT real")

            # Print to file
            plt.savefig("Figure63DFT.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure63haar(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter06 script `Figure63HAAR.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt

            from helpers.wavedec import wavedec
            from helpers.MyDisp import MyDisp

            # Parameters
            NR = 16
            NC = 1
            LN = int(np.log2(NR))

            # DWT with Haar wavelets.
            I = np.eye(NR)
            HAAR = np.zeros((NR, NR))
            for i in range(NR):
                HAAR[:, i], _ = wavedec(I[:, i], LN, "haar")

            # Display
            plt.figure()
            position = 1
            Error = MyDisp(HAAR, NR, NC, position, "Haar")

            # Print to file
            plt.savefig("Figure63HAAR.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure63slt(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter06 script `Figure63SLT.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt

            from helpers.MyDisp import MyDisp

            # Parameters
            N = 16
            NC = 1
            LN = int(np.log2(N))

            # Slant transform
            a = 3 / np.sqrt(5)
            b = 1 / np.sqrt(5)
            sp = np.array(
                [[1, 1, 1, 1], [a, b, -b, -a], [1, -1, -1, 1], [b, -a, a, -b]],
                dtype=float,
            )

            for i in range(3, LN + 1):
                NN = 2**i
                aN = np.sqrt((3 * NN**2) / (4 * (NN**2 - 1)))
                bN = np.sqrt((NN**2 - 4) / (4 * (NN**2 - 1)))

                sr1 = np.array([[1, 0], [aN, bN]], dtype=float)
                sr2 = np.array([[1, 0], [-aN, bN]], dtype=float)
                sz = np.zeros((2, (NN - 4) // 2))
                sn1 = np.hstack([sr1, sz, sr2, sz])

                q = (NN // 2) - 2
                ir = np.eye(q)
                iz = np.zeros((q, 2))
                sn2 = np.hstack([iz, ir, iz, ir])
                sn4 = np.hstack([iz, ir, iz, -ir])

                sr1 = np.array([[0, 1], [-bN, aN]], dtype=float)
                sr2 = np.array([[0, -1], [bN, aN]], dtype=float)
                sn3 = np.hstack([sr1, sz, sr2, sz])

                sn = np.vstack([sn1, sn2, sn3, sn4])

                m2 = np.block([[sp, np.zeros_like(sp)], [np.zeros_like(sp), sp]])

                sp = sn @ m2

                SLANT = np.zeros_like(sp)
                for k in range(NN):
                    if k < 2:
                        seq = k
                    elif k <= NN // 2 - 1:
                        if k % 2 == 0:
                            seq = 2 * k
                        else:
                            seq = 2 * k + 1
                    elif k == NN // 2:
                        seq = 2
                    elif k == NN // 2 + 1:
                        seq = 3
                    else:
                        if k % 2 == 0:
                            seq = 2 * (k - NN // 2) + 1
                        else:
                            seq = 2 * (k - NN // 2)
                    SLANT[seq, :] = sp[k, :]
                sp = SLANT

            sp = sp / np.sqrt(N)
            SLANT = sp.copy()

            # Display
            plt.figure()
            position = 1
            Error = MyDisp(SLANT, N, NC, position, "Slant")

            # Print to file
            plt.savefig("Figure63SLT.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure63std(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter06 script `Figure63STD.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt

            from helpers.MyDisp import MyDisp

            # Parameters
            NR = 16
            NC = 1

            I = np.eye(NR)

            # Display
            plt.figure()
            position = 1
            Error = MyDisp(I, NR, NC, position, "Canonical Basis")

            # Print to file
            plt.savefig("Figure63STD.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure63wht(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter06 script `Figure63WHT.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy.linalg import hadamard

            from helpers.MyDisp import MyDisp

            # Parameters
            NR = 16
            NC = 1

            # Walsh Hadamard transform (normalized)
            HAD = hadamard(NR) / np.sqrt(NR)

            # Walsh sequency ordering
            HadIdx = np.arange(NR)
            M = int(np.log2(NR)) + 1

            # Bit reverse
            binHadIdx = np.array(
                [list(np.binary_repr(i, width=M)) for i in HadIdx], dtype=int
            )
            binHadIdx = np.fliplr(binHadIdx)

            binSeqIdx = np.zeros((NR, M - 1), dtype=int)
            for k in range(M - 1, 0, -1):
                binSeqIdx[:, k - 1] = np.bitwise_xor(
                    binHadIdx[:, k], binHadIdx[:, k - 1]
                )

            SeqIdx = binSeqIdx.dot(2 ** np.arange(M - 2, -1, -1))
            WHT = HAD[SeqIdx, :]

            # Display
            plt.figure()
            position = 1
            Error = MyDisp(WHT, NR, NC, position, "Walsh Hadamard")

            # Print to file
            plt.savefig("Figure63WHT.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure67(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter06 script `Figure67.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt

            from libDIP.basisImage4e import basisImage4e

            # Process
            S_COMPOSITEr, S_DISPLAYr = basisImage4e("DFTr", 8, 1)
            S_COMPOSITEi, S_DISPLAYi = basisImage4e("DFTi", 8, 1)

            # Display
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(
                S_DISPLAYr, cmap="gray", vmin=S_DISPLAYr.min(), vmax=S_DISPLAYr.max()
            )
            axes[0].axis("off")
            axes[1].imshow(
                S_DISPLAYi, cmap="gray", vmin=S_DISPLAYi.min(), vmax=S_DISPLAYi.max()
            )
            axes[1].axis("off")

            plt.tight_layout()
            plt.savefig("Figure67.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure68(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter06 script `Figure68.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt

            from libDIP.basisImage4e import basisImage4e
            from libDIP.tmat4e import tmat4e

            def MyDisp(Matrix: Any, N: Any, plots: Any, position: Any, Title: Any):
                """MyDisp."""
                Factor = 1.0 / Matrix[0, 0]
                t = np.arange(0, N + 1 / 1000, 1 / 1000.0)

                for i in range(N):
                    f = np.cos(2 * np.pi * t * i / N) + np.sin(2 * np.pi * t * i / N)

                    ax = plt.subplot(N, plots, (plots * (i + 1)) - (plots - position))
                    markerline, stemlines, baseline = ax.stem(
                        np.arange(0, N), Matrix[i, :], markerfmt="o", basefmt=" "
                    )
                    markerline.set_markeredgecolor("none")
                    markerline.set_markerfacecolor((0, 105 / 255, 166 / 255))
                    markerline.set_markersize(2.25 * 4 / 1.5)
                    stemlines.set_linewidth(0.5 * 0.5 / 0.75)

                    ax.plot(t, f / Factor)

                    ax.set_frame_on(False)
                    ax.axis("off")
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    if i == 0:
                        ax.set_title(Title)

            # Parameters
            N = 8
            P = 1
            position = 1
            plots = 2

            # Process
            S_COMPOSITE, S_DISPLAY = basisImage4e("DHT", 8, 1)
            DHT = tmat4e("DHT", N)

            # Display
            plt.figure()
            plt.subplot(1, 2, 1)
            MyDisp(DHT, N, plots, position, "DHT")
            plt.subplot(1, 2, 2)
            plt.imshow(
                S_DISPLAY, cmap="gray", vmin=S_DISPLAY.min(), vmax=S_DISPLAY.max()
            )
            plt.axis("off")

            # Print to file
            plt.savefig("Figure68.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure69(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter06 script `Figure69.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from libDIP.tmat4e import tmat4e

            # %% Figure69

            # Parameters
            N = 8

            # Data
            f = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=float)

            # DHT / DFT matrices
            DHT = tmat4e("DHT", N)

            # MATLAB dftmtx(N)
            k = np.arange(N)
            DFT = np.exp(-1j * 2 * np.pi * np.outer(k, k) / N)

            # Reconstruction buffers
            Rec = np.zeros((N, 8001), dtype=float)
            Rec1 = np.zeros((N, 8001), dtype=complex)

            t = np.arange(0, N + 1 / 1000.0, 1 / 1000.0)

            Theta = DHT @ f
            Theta1 = DFT @ f

            for i in range(N):
                Phi = np.cos(2 * np.pi * t * i / N) + np.sin(2 * np.pi * t * i / N)
                Phi1 = np.cos(2 * np.pi * t * i / N) + 1j * np.sin(
                    2 * np.pi * t * i / N
                )

                if i == 0:
                    Rec[i, :] = Phi * Theta[i]
                    Rec1[i, :] = Phi1 * Theta1[i]
                else:
                    Rec[i, :] = Rec[i - 1, :] + Phi * Theta[i]
                    Rec1[i, :] = Rec1[i - 1, :] + Phi1 * Theta1[i]

            Factor = 1.0 / Rec[7, 0]
            Factor1 = 1.0 / Rec1[7, 0]

            tn = np.arange(0, 8)

            # Display
            plt.figure(figsize=(10, 7), dpi=100)
            for iter_idx in range(N):
                ax1 = plt.subplot(N, 2, 2 * iter_idx + 1)
                ax1.plot(t, Factor * Rec[iter_idx, :])
                ax1.stem(tn, f, linefmt="C1-", markerfmt="C1o", basefmt=" ")
                ax1.set_xlim(t[0], t[-1])

                ax2 = plt.subplot(N, 2, 2 * iter_idx + 2)
                ax2.plot(t, np.real(Factor1 * Rec1[iter_idx, :]))
                ax2.stem(tn, f, linefmt="C1-", markerfmt="C1o", basefmt=" ")
                ax2.set_xlim(t[0], t[-1])

            plt.subplots_adjust(
                left=0.06, right=0.98, top=0.96, bottom=0.06, wspace=0.18, hspace=0.35
            )
            plt.savefig("Figure69.png")
            print("Saved Figure69.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

