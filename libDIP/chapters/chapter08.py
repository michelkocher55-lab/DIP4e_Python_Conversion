from pathlib import Path as _Path
import os as _os

from typing import Any


class Chapter08Mixin:
    def figure81(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter08 script `Figure81.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread

            from processing.Chapter08.fig81bc import fig81bc
            from helpers.data_path import dip_data

            # Figure81

            # Data
            fa = imread(dip_data("Fig0801(a).tif"))
            fb = fig81bc("b")
            fc = fig81bc("c")

            # Display (Figure 1)
            fig1, axes = plt.subplots(1, 3, figsize=(10, 4))
            axes[0].imshow(fa, cmap="gray", vmin=0, vmax=255)
            axes[0].axis("off")
            axes[1].imshow(fb, cmap="gray", vmin=0, vmax=255)
            axes[1].axis("off")
            axes[2].imshow(fc, cmap="gray", vmin=0, vmax=255)
            axes[2].axis("off")
            fig1.tight_layout()

            # Display (Figure 2): bar(hist(double(fb(:)), 256))
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            counts, edges = np.histogram(fb.astype(float).ravel(), bins=256)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax2.bar(centers, counts, width=(edges[1] - edges[0]))
            fig2.tight_layout()

            # Print to file
            fig1.savefig("Figure81.png", dpi=300, bbox_inches="tight")
            fig2.savefig("Figure82.png", dpi=300, bbox_inches="tight")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure810(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter08 script `Figure810.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt

            # Figure810

            def cdf(w: Any, mu: Any, beta: Any):
                """cdf."""
                if w < -5:
                    return 0.0
                if w > 5:
                    return 1.0
                return 0.5 * (
                    1.0 + (np.sign(w - mu) * (1.0 - np.exp(-abs(w - mu) / beta)))
                )

            def one_sided_geometric(select: Any):
                """one_sided_geometric."""
                rho = 0.25
                x = np.ones((10, 3), dtype=float)
                xval = np.linspace(0, 9, 10)

                for j in range(3):
                    for i in range(10):
                        x[i, j] = (1.0 - rho) * (rho**i)
                    rho = rho + 0.25

                z = np.floor(1000 * x[:, select - 1]).astype(int)
                zsum = int(np.sum(z))
                zz = np.zeros(zsum, dtype=int)

                symbol = 0
                index = 0
                for i in range(len(z)):
                    for _ in range(z[i]):
                        zz[index] = symbol
                        index += 1
                    symbol += 1

                return x, xval, zz

            def two_sided_exp():
                """two_sided_exp."""
                mu = 0.0
                beta = np.sqrt(0.5)
                xval = np.linspace(-5, 5, 11)
                y = np.zeros(11, dtype=float)
                psum = 0.0

                for i in range(11):
                    y[i] = cdf(xval[i] + 0.5, mu, beta) - cdf(xval[i] - 0.5, mu, beta)
                    psum = psum + y[i]

                return xval, y, psum

            def two_sided_exp_reordered(y: Any):
                """two_sided_exp_reordered."""
                xval = np.linspace(0, 9, 10)
                z = xval.copy()
                v = np.array([6, 5, 7, 4, 8, 3, 9, 2, 10, 1], dtype=int)
                for i in range(10):
                    z[i] = y[v[i] - 1]
                return z

            # Parameters
            select = 3

            # Process
            x, xval, zz = one_sided_geometric(select)
            xval1, y, psum = two_sided_exp()
            print(psum)
            z = two_sided_exp_reordered(y)

            # Display
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))

            axes[0].plot(
                xval, x[:, 0], "k-s", xval, x[:, 1], "k--o", xval, x[:, 2], "k:d"
            )
            axes[0].set_xlim(0, 9)
            axes[0].set_ylim(0, 1)
            axes[0].set_xticks(np.arange(0, 10, 1))
            axes[0].set_box_aspect(1)
            axes[0].set_xlabel("n")
            axes[0].set_ylabel("Probability")
            axes[0].legend(["0.25", "0.5", "0.75"], frameon=False)
            axes[0].set_title("Geometric Distributions")

            axes[1].plot(xval1, y, "k-s")
            axes[1].set_xlim(-5, 5)
            axes[1].set_ylim(0, 1)
            axes[1].set_xticks(np.arange(-5, 6, 1))
            axes[1].set_box_aspect(1)
            axes[1].set_xlabel("x")
            axes[1].text(0, 0.9, "mean 0, variance 1")
            axes[1].set_title("Laplacian Distribution")

            axes[2].plot(xval, z, "k-s")
            axes[2].set_xlim(0, 9)
            axes[2].set_ylim(0, 1)
            axes[2].set_xticks(np.arange(0, 10, 1))
            axes[2].set_box_aspect(1)
            axes[2].set_xlabel("n")
            axes[2].set_title("Interleaved Laplacian Distribution")

            plt.tight_layout()
            fig.savefig("Figure810.png", dpi=300, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure811(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter08 script `Figure811.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.golomb import golomb
            from helpers.mat2huff import mat2huff
            from helpers.imratio import imratio
            from helpers.data_path import dip_data

            # Figure 811

            # Parameters
            m = 5

            # Data
            # Read image and subtract the average value.
            i = imread(dip_data("Fig81c.tif"))
            x = i.astype(float) - 128

            # Compute histogram between min and max with bin size 1.
            xmin = int(np.min(x))
            xmax = int(np.max(x))
            x = x.ravel()

            edges = np.arange(xmin, xmax + 2)
            h, _ = np.histogram(x, bins=edges)
            hx = np.linspace(xmin, xmax, xmax - xmin + 1)

            # Golomb coding
            h1, x1, cr = golomb(x, m)
            print(cr)

            # Display
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            axes[0].plot(hx, h / np.sum(h), "k-s")
            axes[0].set_xticks(np.arange(xmin, xmax + 1, 1))
            axes[0].set_box_aspect(1)

            x2 = np.linspace(0, xmax - xmin + 1, xmax - xmin + 2)
            axes[1].plot(x2, h1 / np.sum(h1), "k-s")
            axes[1].set_xticks(np.arange(0, xmax - xmin + 3, 1))
            axes[1].set_box_aspect(1)

            plt.tight_layout()

            # To compute the Huffman alternative ratio
            c = mat2huff(i)
            cr1 = imratio(i, c)
            print(cr1)

            # Print to file
            fig.savefig("Figure811.png", dpi=300, bbox_inches="tight")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure819(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter08 script `Figure819.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            # Figure819.py

            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.data_path import dip_data

            # Data
            f = imread(dip_data("Fig0819(a).tif")).astype(np.uint8)

            # Convert to Gray code
            g = np.bitwise_xor(f, np.right_shift(f, 1))

            # Get all bit planes (bit 0 to bit 7)
            planef = [np.bitwise_and(np.right_shift(f, k), 1) for k in range(8)]
            planeg = [np.bitwise_and(np.right_shift(g, k), 1) for k in range(8)]

            # Figure 1
            fig1 = plt.figure(1, figsize=(10, 5))

            plt.subplot(2, 4, 1)
            plt.imshow(planef[7], cmap="gray", vmin=0, vmax=1)
            plt.title("Original, bit 7")
            plt.axis("off")
            plt.subplot(2, 4, 2)
            plt.imshow(planef[6], cmap="gray", vmin=0, vmax=1)
            plt.title("Original, bit 6")
            plt.axis("off")
            plt.subplot(2, 4, 3)
            plt.imshow(planef[5], cmap="gray", vmin=0, vmax=1)
            plt.title("Original, bit 5")
            plt.axis("off")
            plt.subplot(2, 4, 4)
            plt.imshow(planef[4], cmap="gray", vmin=0, vmax=1)
            plt.title("Original, bit 4")
            plt.axis("off")

            plt.subplot(2, 4, 5)
            plt.imshow(planeg[7], cmap="gray", vmin=0, vmax=1)
            plt.title("Gray coded, bit 7")
            plt.axis("off")
            plt.subplot(2, 4, 6)
            plt.imshow(planeg[6], cmap="gray", vmin=0, vmax=1)
            plt.title("Gray coded, bit 6")
            plt.axis("off")  # fixed
            plt.subplot(2, 4, 7)
            plt.imshow(planeg[5], cmap="gray", vmin=0, vmax=1)
            plt.title("Gray coded, bit 5")
            plt.axis("off")
            plt.subplot(2, 4, 8)
            plt.imshow(planeg[4], cmap="gray", vmin=0, vmax=1)
            plt.title("Gray coded, bit 4")
            plt.axis("off")

            fig1.tight_layout()

            # Figure 2
            fig2 = plt.figure(2, figsize=(10, 5))

            plt.subplot(2, 4, 1)
            plt.imshow(planef[3], cmap="gray", vmin=0, vmax=1)
            plt.title("Original, bit 3")
            plt.axis("off")
            plt.subplot(2, 4, 2)
            plt.imshow(planef[2], cmap="gray", vmin=0, vmax=1)
            plt.title("Original, bit 2")
            plt.axis("off")
            plt.subplot(2, 4, 3)
            plt.imshow(planef[1], cmap="gray", vmin=0, vmax=1)
            plt.title("Original, bit 1")
            plt.axis("off")
            plt.subplot(2, 4, 4)
            plt.imshow(planef[0], cmap="gray", vmin=0, vmax=1)
            plt.title("Original, bit 0")
            plt.axis("off")

            plt.subplot(2, 4, 5)
            plt.imshow(planeg[3], cmap="gray", vmin=0, vmax=1)
            plt.title("Gray coded, bit 3")
            plt.axis("off")
            plt.subplot(2, 4, 6)
            plt.imshow(planeg[2], cmap="gray", vmin=0, vmax=1)
            plt.title("Gray coded, bit 2")
            plt.axis("off")
            plt.subplot(2, 4, 7)
            plt.imshow(planeg[1], cmap="gray", vmin=0, vmax=1)
            plt.title("Gray coded, bit 1")
            plt.axis("off")
            plt.subplot(2, 4, 8)
            plt.imshow(planeg[0], cmap="gray", vmin=0, vmax=1)
            plt.title("Gray coded, bit 0")
            plt.axis("off")

            fig2.tight_layout()

            # Save figures
            fig1.savefig("Figure819.png", dpi=150, bbox_inches="tight")
            fig2.savefig("Figure820.png", dpi=150, bbox_inches="tight")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure822(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter08 script `Figure822.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from libDIP.tmat4e import tmat4e
            from helpers.data_path import dip_data

            def blkdct(x: Any, a: Any):
                """blkdct."""
                return a @ x @ a.conj().T

            def blkzero(x: Any):
                """blkzero."""
                k = 32
                y = np.zeros_like(x)
                idx = np.argsort(np.abs(x).ravel())[::-1]
                y.ravel()[idx[:k]] = x.ravel()[idx[:k]]
                return y

            def blockproc_8x8(f: Any, func: Any, pad_partial_blocks: Any = True):
                """blockproc_8x8."""
                m, n = f.shape
                if pad_partial_blocks:
                    pad_m = (8 - (m % 8)) % 8
                    pad_n = (8 - (n % 8)) % 8
                    fp = np.pad(f, ((0, pad_m), (0, pad_n)), mode="constant")
                else:
                    fp = f

                out = np.zeros_like(fp, dtype=np.complex128)
                mp, np_ = fp.shape

                for r in range(0, mp, 8):
                    for c in range(0, np_, 8):
                        out[r : r + 8, c : c + 8] = func(fp[r : r + 8, c : c + 8])

                return out

            def process(f: Any, t: Any):
                """process."""
                y = blockproc_8x8(f, lambda b: blkdct(b, t), pad_partial_blocks=True)
                y = blockproc_8x8(y, blkzero, pad_partial_blocks=True)
                y = blockproc_8x8(
                    y, lambda b: blkdct(b, t.conj().T), pad_partial_blocks=True
                )
                return np.real(y)

            # Parameters
            t = [
                tmat4e("DFT", 8),
                tmat4e("DCT", 8),
                tmat4e("DHT", 8),
            ]

            # Data
            f = imread(dip_data("lena.tif")).astype(float)
            if f.ndim == 3:
                f = f[..., 0]

            # Process
            f_hat = [process(f, t[0]), process(f, t[1]), process(f, t[2])]

            # Display
            fig = plt.figure(1, figsize=(12, 7))

            plt.subplot(2, 3, 1)
            plt.imshow(f_hat[0], cmap="gray")
            plt.axis("off")

            plt.subplot(2, 3, 2)
            plt.imshow(f_hat[1], cmap="gray")
            plt.axis("off")

            plt.subplot(2, 3, 3)
            plt.imshow(f_hat[2], cmap="gray")
            plt.axis("off")

            plt.subplot(2, 3, 4)
            plt.imshow(f - f_hat[0], cmap="gray")
            plt.axis("off")

            plt.subplot(2, 3, 5)
            plt.imshow(f - f_hat[1], cmap="gray")
            plt.axis("off")

            plt.subplot(2, 3, 6)
            plt.imshow(f - f_hat[2], cmap="gray")
            plt.axis("off")

            plt.tight_layout()
            fig.savefig("Figure822.png", dpi=150, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure823(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter08 script `Figure823.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread

            from libDIP.tmat4e import tmat4e
            from helpers.compare import compare
            from helpers.data_path import dip_data

            # Figure 823

            def blkdct(x: Any, a: Any):
                """Block transform using matrix multiplications."""
                return a @ x @ a.conj().T

            def blkzero(x: Any, keep_frac: Any):
                """Keep the largest-magnitude coefficients in a block."""
                n = x.size
                k = int(np.round(keep_frac * n))

                idx = np.argsort(np.abs(x).ravel())[::-1]
                y = np.zeros_like(x)
                if k > 0:
                    y.ravel()[idx[:k]] = x.ravel()[idx[:k]]
                return y

            def blockproc_square(f: Any, block_size: Any, func: Any):
                """Apply func to non-overlapping square blocks with zero-padding of partial blocks."""
                m, n = f.shape
                pad_m = (block_size - (m % block_size)) % block_size
                pad_n = (block_size - (n % block_size)) % block_size

                fp = np.pad(f, ((0, pad_m), (0, pad_n)), mode="constant")
                out = np.zeros_like(fp, dtype=np.complex128)

                mp, np_ = fp.shape
                for r in range(0, mp, block_size):
                    for c in range(0, np_, block_size):
                        out[r : r + block_size, c : c + block_size] = func(
                            fp[r : r + block_size, c : c + block_size]
                        )

                # Crop back to original size for fair RMS comparison.
                return out[:m, :n]

            def process(f: Any, t: Any, block_size: Any, keep_frac: Any):
                """process."""
                # Compute forward transform of block_size x block_size blocks.
                y = blockproc_square(f, block_size, lambda b: blkdct(b, t))

                # Zero coefficients based on magnitude.
                y = blockproc_square(y, block_size, lambda b: blkzero(b, keep_frac))

                # Compute inverse transform.
                y = np.real(
                    blockproc_square(y, block_size, lambda b: blkdct(b, np.conj(t.T)))
                )

                # Compute RMS error.
                return compare(f, y, 0)

            # Parameters
            BlockSize = [2, 4, 8, 16, 32, 64, 128, 256, 512]
            KeepFrac = 0.25  # 75% are zeroed

            # Data
            f = imread(dip_data("lena.tif")).astype(float)
            if f.ndim == 3:
                f = f[..., 0]

            # Process
            RMS = np.zeros((3, len(BlockSize)), dtype=float)
            transforms = [[None for _ in BlockSize] for _ in range(3)]

            for iter_idx in range(3):
                for iter1, bsz in enumerate(BlockSize):
                    if iter_idx == 0:
                        transforms[iter_idx][iter1] = tmat4e("DFT", bsz)
                    elif iter_idx == 1:
                        transforms[iter_idx][iter1] = tmat4e("DCT", bsz)
                    else:
                        transforms[iter_idx][iter1] = tmat4e("WHT", bsz)

                    RMS[iter_idx, iter1] = process(
                        f, transforms[iter_idx][iter1], bsz, KeepFrac
                    )

            # Display
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(BlockSize, RMS[0, :], "k-s", label="DFT")
            ax.plot(BlockSize, RMS[1, :], "k--o", label="DCT")
            ax.plot(BlockSize, RMS[2, :], "k:d", label="WHT")
            ax.legend()
            ax.set_xlabel("Block size")
            ax.set_ylabel("RMS error")
            ax.set_title(f"Only {KeepFrac * 100:g} % of the coefficients are kept")
            ax.autoscale(enable=True, axis="both", tight=True)

            plt.tight_layout()
            fig.savefig("Figure823.png", dpi=300, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure824(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter08 script `Figure824.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from libDIP.tmat4e import tmat4e
            from helpers.data_path import dip_data

            def blkdct(x: Any, a: Any):
                """blkdct."""
                return a @ x @ a.conj().T

            def blkzero(x: Any, keep_frac: Any):
                """blkzero."""
                nr, nc = x.shape
                k = int(np.round(keep_frac * nr * nc))
                y = np.zeros_like(x)
                idx = np.argsort(np.abs(x).ravel())[::-1]
                if k > 0:
                    y.ravel()[idx[:k]] = x.ravel()[idx[:k]]
                return y

            def blockproc_square(
                f: Any, block_size: Any, func: Any, pad_partial_blocks: Any = True
            ):
                """blockproc_square."""
                m, n = f.shape
                if pad_partial_blocks:
                    pad_m = (block_size - (m % block_size)) % block_size
                    pad_n = (block_size - (n % block_size)) % block_size
                    fp = np.pad(f, ((0, pad_m), (0, pad_n)), mode="constant")
                else:
                    fp = f
                    pad_m = 0
                    pad_n = 0

                out = np.zeros_like(fp, dtype=np.complex128)
                mp, np_ = fp.shape

                for r in range(0, mp, block_size):
                    for c in range(0, np_, block_size):
                        out[r : r + block_size, c : c + block_size] = func(
                            fp[r : r + block_size, c : c + block_size]
                        )

                if pad_partial_blocks and (pad_m or pad_n):
                    out = out[:m, :n]

                return out

            def process(f: Any, t: Any, block_size: Any, keep_frac: Any):
                """process."""
                y = blockproc_square(
                    f, block_size, lambda b: blkdct(b, t), pad_partial_blocks=True
                )
                y = blockproc_square(
                    y,
                    block_size,
                    lambda b: blkzero(b, keep_frac),
                    pad_partial_blocks=True,
                )
                y = blockproc_square(
                    y,
                    block_size,
                    lambda b: blkdct(b, t.conj().T),
                    pad_partial_blocks=True,
                )
                return np.real(y)

            def imcrop_matlab(img: Any, rect: Any):
                """imcrop_matlab."""
                # MATLAB imcrop([x, y, w, h]) includes both endpoints for integer coords.
                x, y, w, h = [int(v) for v in rect]
                return img[y : y + h + 1, x : x + w + 1]

            # Parameters
            keep_frac = 0.25
            block_sizes = [2, 4, 8]
            t = [tmat4e("DCT", bs) for bs in block_sizes]

            # Data
            f = imread(dip_data("lena.tif")).astype(float)
            if f.ndim == 3:
                f = f[..., 0]

            f = imcrop_matlab(f, [243 - 7, 249 - 7, 31, 31])

            # Process
            f_hat = [
                process(f, t[i], block_sizes[i], keep_frac)
                for i in range(len(block_sizes))
            ]

            # Display
            fig = plt.figure(1, figsize=(12, 4))

            plt.subplot(1, 4, 1)
            plt.imshow(f, cmap="gray")
            plt.title("Original")
            plt.axis("off")

            for i, bs in enumerate(block_sizes):
                plt.subplot(1, 4, i + 2)
                plt.imshow(f_hat[i], cmap="gray")
                plt.title(f"Block Size = {bs}")
                plt.axis("off")

            plt.tight_layout()
            fig.savefig("Figure824.png", dpi=150, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure825(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter08 script `Figure825.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread

            from libDIP.tmat4e import tmat4e
            from helpers.compare import compare
            from helpers.data_path import dip_data

            def blkdct(x: Any, a: Any):
                """blkdct."""
                return a @ x @ a.conj().T

            def blkzero_zonal(x: Any, mask: Any):
                """blkzero_zonal."""
                return x * mask

            def blkzero_magnitude(x: Any, keep_frac: Any):
                """blkzero_magnitude."""
                m, n = x.shape
                k = int(np.round(keep_frac * m * n))
                y = np.zeros_like(x)
                if k <= 0:
                    return y
                idx = np.argsort(np.abs(x).ravel())[::-1]
                y.ravel()[idx[:k]] = x.ravel()[idx[:k]]
                return y

            def blockproc_square(
                f: Any, block_size: Any, func: Any, pad_partial_blocks: Any = True
            ):
                """blockproc_square."""
                m, n = f.shape
                if pad_partial_blocks:
                    pad_m = (block_size - (m % block_size)) % block_size
                    pad_n = (block_size - (n % block_size)) % block_size
                    fp = np.pad(f, ((0, pad_m), (0, pad_n)), mode="constant")
                else:
                    fp = f
                    pad_m = 0
                    pad_n = 0

                out = np.zeros_like(fp, dtype=np.complex128)
                mp, np_ = fp.shape

                for r in range(0, mp, block_size):
                    for c in range(0, np_, block_size):
                        out[r : r + block_size, c : c + block_size] = func(
                            fp[r : r + block_size, c : c + block_size]
                        )

                if pad_partial_blocks and (pad_m or pad_n):
                    out = out[:m, :n]

                return out

            def compute_zonal_mask(f: Any, t: Any, block_size: Any, keep_frac: Any):
                """compute_zonal_mask."""
                # Forward transform block-by-block.
                y = blockproc_square(
                    f, block_size, lambda b: blkdct(b, t), pad_partial_blocks=True
                )

                # Build 3D stack explicitly to avoid reshape-order mismatch with MATLAB.
                m, n = y.shape
                pad_m = (block_size - (m % block_size)) % block_size
                pad_n = (block_size - (n % block_size)) % block_size
                yp = (
                    np.pad(y, ((0, pad_m), (0, pad_n)), mode="constant")
                    if (pad_m or pad_n)
                    else y
                )

                nb_r = yp.shape[0] // block_size
                nb_c = yp.shape[1] // block_size
                num_blocks = nb_r * nb_c

                p = np.zeros((block_size, block_size, num_blocks), dtype=float)
                kblk = 0
                for br in range(nb_r):
                    for bc in range(nb_c):
                        block = yp[
                            br * block_size : (br + 1) * block_size,
                            bc * block_size : (bc + 1) * block_size,
                        ]
                        p[:, :, kblk] = np.real(block)
                        kblk += 1

                v = np.var(p, axis=2, ddof=0)

                k = int(np.round(keep_frac * block_size * block_size))
                idx = np.argsort(v.ravel())[::-1]
                msk = np.zeros((block_size, block_size), dtype=float)
                if k > 0:
                    msk.ravel()[idx[:k]] = 1.0

                return v, msk

            def zonal_coding(f: Any, t: Any, block_size: Any, mask: Any):
                """zonal_coding."""
                y = blockproc_square(
                    f, block_size, lambda b: blkdct(b, t), pad_partial_blocks=True
                )
                y = blockproc_square(
                    y,
                    block_size,
                    lambda b: blkzero_zonal(b, mask),
                    pad_partial_blocks=True,
                )
                y = blockproc_square(
                    y,
                    block_size,
                    lambda b: blkdct(b, t.conj().T),
                    pad_partial_blocks=True,
                )
                return np.real(y)

            def magnitude_coding(f: Any, t: Any, block_size: Any, keep_frac: Any):
                """magnitude_coding."""
                y = blockproc_square(
                    f, block_size, lambda b: blkdct(b, t), pad_partial_blocks=True
                )
                y = blockproc_square(
                    y,
                    block_size,
                    lambda b: blkzero_magnitude(b, keep_frac),
                    pad_partial_blocks=True,
                )
                y = blockproc_square(
                    y,
                    block_size,
                    lambda b: blkdct(b, t.conj().T),
                    pad_partial_blocks=True,
                )
                return np.real(y)

            # Parameters
            keep_frac = 0.125
            block_size = 8
            t = tmat4e("DCT", block_size)

            # Data
            f = imread(dip_data("lena.tif")).astype(float)
            if f.ndim == 3:
                f = f[..., 0]

            # Magnitude coding
            f_hat_magnitude = magnitude_coding(f, t, block_size, keep_frac)
            e_magnitude = f - f_hat_magnitude
            e_rms_magnitude = compare(f, f_hat_magnitude, 0)

            # Zonal coding
            variances, mask = compute_zonal_mask(f, t, block_size, keep_frac)
            f_hat_zonal = zonal_coding(f, t, block_size, mask)
            e_zonal = f - f_hat_zonal
            e_rms_zonal = compare(f, f_hat_zonal, 0)

            # Display figure 1
            fig1 = plt.figure(1, figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(f, cmap="gray")
            plt.title("Original image")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(np.log10(1 + variances), cmap="gray")
            plt.title("log(1+variances)")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(mask, cmap="gray", vmin=0, vmax=1)
            plt.title(f"Mask, KeepFrac = {keep_frac:g}")
            plt.axis("off")

            plt.tight_layout()
            fig1.savefig("Figure825.png", dpi=150, bbox_inches="tight")

            # Display figure 2
            fig2 = plt.figure(2, figsize=(10, 8))
            plt.subplot(2, 2, 1)
            plt.imshow(f_hat_magnitude, cmap="gray")
            plt.title("Magnitude Coding")
            plt.axis("off")

            plt.subplot(2, 2, 2)
            plt.imshow(4 * e_magnitude, cmap="gray")
            plt.title(f"4 * error, RMS = {e_rms_magnitude:g}")
            plt.axis("off")

            plt.subplot(2, 2, 3)
            plt.imshow(f_hat_zonal, cmap="gray")
            plt.title("Zonal Coding")
            plt.axis("off")

            plt.subplot(2, 2, 4)
            plt.imshow(4 * e_zonal, cmap="gray")
            plt.title(f"4 * error, RMS = {e_rms_zonal:g}")
            plt.axis("off")

            plt.tight_layout()
            fig2.savefig("Figure825Bis.png", dpi=150, bbox_inches="tight")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure828(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter08 script `Figure828.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread

            from libDIP.tmat4e import tmat4e
            from helpers.compare import compare
            from helpers.data_path import dip_data

            def compute_q(q: Any):
                """compute_q."""
                base = np.array(
                    [
                        [16, 11, 10, 16, 24, 40, 51, 61],
                        [12, 12, 14, 19, 26, 58, 60, 55],
                        [14, 13, 16, 24, 40, 57, 69, 56],
                        [14, 17, 22, 29, 51, 87, 80, 62],
                        [18, 22, 37, 56, 68, 109, 103, 77],
                        [24, 35, 55, 64, 81, 104, 113, 92],
                        [49, 64, 78, 87, 103, 121, 120, 101],
                        [72, 92, 95, 98, 112, 100, 103, 99],
                    ],
                    dtype=float,
                )
                return q * base

            def blkdct(x: Any, a: Any):
                """blkdct."""
                return a @ x @ a.conj().T

            def blkidct(x: Any, a: Any, qmatrix: Any):
                """blkidct."""
                x = qmatrix * x
                return a @ x @ a.conj().T

            def threshold_coding(x: Any, qmatrix: Any):
                """threshold_coding."""
                return np.round(x / qmatrix)

            def blockproc_square(
                f: Any, block_size: Any, func: Any, pad_partial_blocks: Any = True
            ):
                """blockproc_square."""
                m, n = f.shape
                if pad_partial_blocks:
                    pad_m = (block_size - (m % block_size)) % block_size
                    pad_n = (block_size - (n % block_size)) % block_size
                    fp = np.pad(f, ((0, pad_m), (0, pad_n)), mode="constant")
                else:
                    fp = f
                    pad_m = 0
                    pad_n = 0

                out = np.zeros_like(fp, dtype=np.complex128)
                mp, np_ = fp.shape

                for r in range(0, mp, block_size):
                    for c in range(0, np_, block_size):
                        out[r : r + block_size, c : c + block_size] = func(
                            fp[r : r + block_size, c : c + block_size]
                        )

                if pad_partial_blocks and (pad_m or pad_n):
                    out = out[:m, :n]

                return out

            def process(f: Any, t: Any, block_size: Any, qmatrix: Any):
                """process."""
                y = blockproc_square(
                    f, block_size, lambda b: blkdct(b, t), pad_partial_blocks=True
                )
                y = blockproc_square(
                    y,
                    block_size,
                    lambda b: threshold_coding(b, qmatrix),
                    pad_partial_blocks=True,
                )
                y = blockproc_square(
                    y,
                    block_size,
                    lambda b: blkidct(b, t.conj().T, qmatrix),
                    pad_partial_blocks=True,
                )
                return np.real(y)

            # Parameters
            q_values = [1, 2, 4, 8, 16, 32]
            block_size = 8
            t = tmat4e("DCT", block_size)

            # Data
            f = imread(dip_data("lena.tif")).astype(float)
            if f.ndim == 3:
                f = f[..., 0]

            # Threshold coding
            f_hat_threshold = []
            error_rms_threshold = []

            for q in q_values:
                q_matrix = compute_q(q)
                f_hat = process(f, t, block_size, q_matrix)
                f_hat_threshold.append(f_hat)
                error_rms_threshold.append(compare(f, f_hat, 0))

            # Display
            fig = plt.figure(1, figsize=(12, 7))
            for i, q in enumerate(q_values):
                plt.subplot(2, 3, i + 1)
                plt.imshow(f_hat_threshold[i], cmap="gray")
                plt.title(
                    f"Thr Cod., Q = {q}, e_{{RMS}} = {error_rms_threshold[i]:.2g}"
                )
                plt.axis("off")

            plt.tight_layout()
            fig.savefig("Figure828.png", dpi=150, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure829(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter08 script `Figure829.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread

            from helpers.im2jpeg import im2jpeg
            from helpers.jpeg2im import jpeg2im
            from helpers.imratio import imratio
            from helpers.compare import compare
            from helpers.data_path import dip_data

            # Parameters
            quality = [4, 8]

            # Data
            f = imread(dip_data("lena.tif"))
            if f.ndim == 3:
                f = f[..., 0]
            f = f.astype(np.uint8)

            # Compression, decompression
            f_hat = []
            compression_ratio = []
            rmse = []
            e = []
            error_min = []
            error_max = []

            for q in quality:
                y = im2jpeg(f, q)
                fh = jpeg2im(y)
                f_hat.append(fh)
                compression_ratio.append(imratio(f, y))
                rmse.append(compare(f, fh, 0))

                err = f.astype(np.float64) - fh.astype(np.float64)
                e.append(err)
                error_min.append(np.min(err))
                error_max.append(np.max(err))

            # MATLAB imcrop([x, y, w, h]) equivalent (inclusive width/height)
            def imcrop_matlab(img: Any, rect: Any):
                """imcrop_matlab."""
                x, y, w, h = [int(v) for v in rect]
                x2 = x + w + 1
                y2 = y + h + 1
                return img[y:y2, x:x2]

            # Display
            fig = plt.figure(1, figsize=(12, 8))

            plt.subplot(2, 3, 1)
            plt.imshow(f_hat[0], cmap="gray", vmin=0, vmax=255)
            plt.title(
                f"Q = {quality[0]}, RMSE = {rmse[0]:.2g} Comp. = {compression_ratio[0]:.2g}"
            )
            plt.axis("off")

            plt.subplot(2, 3, 2)
            # Matches MATLAB script: uses ErrorMin(2), ErrorMax(2) for first error display.
            plt.imshow(e[0], cmap="gray", vmin=error_min[1], vmax=error_max[1])
            plt.axis("off")

            plt.subplot(2, 3, 3)
            plt.imshow(
                imcrop_matlab(f_hat[0], [234, 250, 60, 40]),
                cmap="gray",
                vmin=0,
                vmax=255,
            )
            plt.axis("off")

            plt.subplot(2, 3, 4)
            plt.imshow(f_hat[1], cmap="gray", vmin=0, vmax=255)
            plt.title(
                f"Q = {quality[1]}, RMSE = {rmse[1]:.2g} Comp. = {compression_ratio[1]:.2g}"
            )
            plt.axis("off")

            plt.subplot(2, 3, 5)
            plt.imshow(e[1], cmap="gray", vmin=error_min[1], vmax=error_max[1])
            plt.axis("off")

            plt.subplot(2, 3, 6)
            plt.imshow(
                imcrop_matlab(f_hat[1], [234, 250, 60, 40]),
                cmap="gray",
                vmin=0,
                vmax=255,
            )
            plt.axis("off")

            plt.tight_layout()
            fig.savefig("Figure829.png", dpi=150, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure83(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter08 script `Figure83.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt

            from processing.Chapter08.fig81bc import fig81bc
            from libDIP.histEqual4e import histEqual4e

            # Figure 8.3

            # Process
            y = fig81bc("c")
            z = histEqual4e(y)

            # Display
            fig, axes = plt.subplots(1, 2, figsize=(9, 4))

            # MATLAB equivalent: plot(hist(double(y(:)), 1:256)), axis square
            # For y in [0, 255], MATLAB hist with centers 1..256 maps value 0 to the first bin.
            centers = np.arange(1, 257)
            counts = np.bincount(y.astype(np.uint8).ravel(), minlength=256)
            axes[0].plot(centers, counts)

            axes[1].imshow(z, cmap="gray")

            plt.tight_layout()

            # Print to file
            fig.savefig("Figure83.png", dpi=300, bbox_inches="tight")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure831(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter08 script `Figure831.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread

            from helpers.compare import compare
            from helpers.lpc2mat import lpc2mat
            from helpers.mat2lpc import mat2lpc
            from helpers.ntrop import ntrop
            from helpers.data_path import dip_data

            # Data
            f_raw = imread(dip_data("nasaframe67.tif"))
            if f_raw.ndim == 3:
                f_raw = f_raw[..., 0]

            # TIFF input is video-inversed for this dataset; invert polarity.
            if np.issubdtype(f_raw.dtype, np.integer):
                f = (np.iinfo(f_raw.dtype).max - f_raw).astype(float)
            else:
                f = (np.max(f_raw) - f_raw).astype(float)

            # Predictive coding 1D
            y = mat2lpc(f, 1)
            f_hat = lpc2mat(y, 1)
            rmse = compare(f, f_hat, 0)

            # Display
            fig = plt.figure(1, figsize=(10, 8))

            plt.subplot(2, 2, 1)
            plt.imshow(f, cmap="gray")
            plt.title("Original")
            plt.axis("off")

            plt.subplot(2, 2, 2)
            hf, _ = np.histogram(f.ravel(), bins=256)
            plt.bar(np.arange(hf.size), hf)
            plt.title(f"H = {ntrop(f.ravel()):g}")

            plt.subplot(2, 2, 3)
            plt.imshow(y, cmap="gray")
            plt.title(f"RMSE = {rmse:g}")
            plt.axis("off")

            plt.subplot(2, 2, 4)
            hy, _ = np.histogram(y.ravel(), bins=256)
            plt.bar(np.arange(hy.size), hy)
            plt.title(f"H = {ntrop(y.ravel()):g}")

            plt.tight_layout()
            fig.savefig("Figure831.png", dpi=150, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure832(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter08 script `Figure832.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread

            from helpers.ntrop import ntrop
            from helpers.data_path import dip_data

            # Data
            f_raw = imread(dip_data("nasaframe78.tif"))
            f_next_raw = imread(dip_data("nasaframe79.tif"))
            if f_raw.ndim == 3:
                f_raw = f_raw[..., 0]
            if f_next_raw.ndim == 3:
                f_next_raw = f_next_raw[..., 0]

            # TIFF input is video-inversed for this dataset; invert polarity.
            if np.issubdtype(f_raw.dtype, np.integer):
                f = (np.iinfo(f_raw.dtype).max - f_raw).astype(float)
            else:
                f = (np.max(f_raw) - f_raw).astype(float)

            if np.issubdtype(f_next_raw.dtype, np.integer):
                f_next = (np.iinfo(f_next_raw.dtype).max - f_next_raw).astype(float)
            else:
                f_next = (np.max(f_next_raw) - f_next_raw).astype(float)

            # Time linear predictive coding
            delta = f - f_next

            # Display
            fig = plt.figure(1, figsize=(10, 8))

            plt.subplot(2, 2, 1)
            plt.imshow(f, cmap="gray")
            plt.title("Frame 78")
            plt.axis("off")

            plt.subplot(2, 2, 2)
            plt.imshow(f_next, cmap="gray")
            plt.title("Frame 79")
            plt.axis("off")

            plt.subplot(2, 2, 3)
            plt.imshow(delta, cmap="gray")
            plt.title("Delta")
            plt.axis("off")

            plt.subplot(2, 2, 4)
            hd, _ = np.histogram(delta.ravel(), bins=256)
            plt.bar(np.arange(hd.size), hd)
            sigma = np.std(delta, ddof=1)  # MATLAB std2 equivalent normalization (N-1)
            plt.title(rf"H = {ntrop(delta.ravel()):g}, $\sigma$ = {sigma:g}")

            plt.tight_layout()
            fig.savefig("Figure832.png", dpi=150, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure834(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter08 script `Figure834.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.motion import motion
            from helpers.data_path import dip_data

            def _invert_if_needed(img: Any):
                """_invert_if_needed."""
                # Match MATLAB visual polarity for these NASA TIFF frames.
                if np.issubdtype(img.dtype, np.integer):
                    return np.iinfo(img.dtype).max - img
                return 1.0 - img

            # %% Figure 834

            # %% Data
            j = imread(dip_data("nasa67.tif"))
            if j.ndim == 3:
                j = j[:, :, 0]
            # MATLAB: j = j(1:352, 119:470)
            j = j[0:352, 118:470]
            j = _invert_if_needed(j)

            i = imread(dip_data("nasa79.tif"))
            if i.ndim == 3:
                i = i[:, :, 0]
            # MATLAB: i = i(1:352, 119:470)
            i = i[0:352, 118:470]
            i = _invert_if_needed(i)

            # %% Motion computation
            e, a, dx, dy = motion(i, j, 16, [16, 16], 1)

            # %% Display
            plt.figure(figsize=(10, 8))

            plt.subplot(2, 2, 1)
            plt.imshow(j, cmap="gray")
            plt.title("Frame 67")
            plt.axis("off")

            plt.subplot(2, 2, 2)
            plt.imshow(i, cmap="gray")
            plt.title("Frame 79")
            plt.axis("off")

            # mat2gray(double(i) - double(j))
            d = i.astype(float) - j.astype(float)
            dmin, dmax = np.min(d), np.max(d)
            if dmax > dmin:
                dshow = (d - dmin) / (dmax - dmin)
            else:
                dshow = np.zeros_like(d)

            plt.subplot(2, 3, 4)
            plt.imshow(dshow, cmap="gray", vmin=0, vmax=1)
            plt.title("Difference")
            plt.axis("off")

            plt.subplot(2, 3, 5)
            plt.imshow(e, cmap="gray")  # MATLAB imshow(e, []) auto scaling
            plt.title("column error")
            plt.axis("off")

            plt.subplot(2, 3, 6)
            plt.imshow(a, cmap="gray", vmin=0, vmax=255)
            plt.title("Motion vectors")
            plt.axis("off")

            plt.tight_layout()
            plt.savefig("Figure834.png")
            print("Saved Figure834.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure835(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter08 script `Figure835.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.motion import motion
            from helpers.data_path import dip_data

            # Parameters
            MacroBlock = [16, 8, 8, 8]
            Delta = np.array([[16, 16], [8, 8], [8, 8], [8, 8]], dtype=int)
            SubPixel = [1, 1, 0.5, 0.25]

            def _invert_if_needed(img: Any):
                """_invert_if_needed."""
                # Match MATLAB visual polarity for these NASA TIFF frames.
                if np.issubdtype(img.dtype, np.integer):
                    return np.iinfo(img.dtype).max - img
                return 1.0 - img

            # Data
            j = imread(dip_data("nasa67.tif"))
            if j.ndim == 3:
                j = j[:, :, 0]
            j = j[0:352, 118:470]
            j = _invert_if_needed(j)

            i = imread(dip_data("nasa79.tif"))
            if i.ndim == 3:
                i = i[:, :, 0]
            i = i[0:352, 118:470]
            i = _invert_if_needed(i)

            # Motion computation
            # [e, a, dx, dy] = motion(i, j, MacroBlock(1), Delta(1, :), SubPixel(1));
            e1, a1, dx1, dy1 = motion(i, j, MacroBlock[1], Delta[1, :], SubPixel[1])
            e2, a2, dx2, dy2 = motion(i, j, MacroBlock[2], Delta[2, :], SubPixel[2])
            e3, a3, dx3, dy3 = motion(i, j, MacroBlock[3], Delta[3, :], SubPixel[3])

            # Difference (mat2gray(double(i)-double(j)))
            d = i.astype(float) - j.astype(float)
            dmin, dmax = np.min(d), np.max(d)
            if dmax > dmin:
                dshow = (d - dmin) / (dmax - dmin)
            else:
                dshow = np.zeros_like(d)

            # Display
            plt.figure(figsize=(10, 8))

            plt.subplot(2, 2, 1)
            plt.imshow(dshow, cmap="gray", vmin=0, vmax=1)
            plt.title("Difference")
            plt.axis("off")

            plt.subplot(2, 2, 2)
            plt.imshow(e1, cmap="gray")  # MATLAB imshow(e1,[])
            plt.title(
                f"e, m.block={MacroBlock[1]}, delta=[{Delta[1, 0]} {Delta[1, 1]}], SubP={SubPixel[1]}"
            )
            plt.axis("off")

            plt.subplot(2, 2, 3)
            plt.imshow(e2, cmap="gray")
            plt.title(
                f"e, m.block={MacroBlock[2]}, delta=[{Delta[2, 0]} {Delta[2, 1]}], SubP={SubPixel[2]}"
            )
            plt.axis("off")

            plt.subplot(2, 2, 4)
            plt.imshow(e3, cmap="gray")
            plt.title(
                f"e, m.block={MacroBlock[3]}, delta=[{Delta[3, 0]} {Delta[3, 1]}], SubP={SubPixel[3]}"
            )
            plt.axis("off")

            plt.tight_layout()
            plt.savefig("Figure835.png")
            print("Saved Figure835.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure84(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter08 script `Figure84.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import os
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from PIL import Image

            from helpers.compare import compare
            from helpers.data_path import dip_data

            # Figure84

            # Parameters
            Quality = [74, 10, 14]
            Name = [
                _os.path.join("output", "Figure84a.jpg"),
                _os.path.join("output", "Figure84b.jpg"),
                _os.path.join("output", "Figure84c.jpg"),
            ]

            # Data
            img_path = dip_data("Fig0801(a).tif")
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")

            f = imread(img_path)

            # Compression, decompression
            fHat = []
            RMSE = []
            for q, name in zip(Quality, Name):
                Image.fromarray(f).save(name, format="JPEG", quality=q)
                rec = imread(name)
                fHat.append(rec)
                RMSE.append(compare(f, rec, 0))

            # Display
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for i in range(len(Quality)):
                fi = np.asarray(fHat[i])
                axes[i].imshow(fi, cmap="gray", vmin=fi.min(), vmax=fi.max())
                axes[i].set_title(f"Q = {Quality[i]}, RMSE = {RMSE[i]:.4f}")
                axes[i].axis("off")

            plt.tight_layout()

            # Print to file
            fig.savefig("Figure84.png", dpi=300, bbox_inches="tight")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure840(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter08 script `Figure840.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread

            from helpers.compare import compare
            from helpers.lpc2mat2d import lpc2mat2d
            from helpers.mat2lpc2d import mat2lpc2d
            from helpers.ntrop import ntrop
            from helpers.data_path import dip_data

            # Data
            f = imread(dip_data("lena.tif"))
            if f.ndim == 3:
                f = f[..., 0]

            # Predictive coding 2D
            y1 = mat2lpc2d(f, 0.97, 0, 0)
            f_hat1 = lpc2mat2d(y1, 0.97, 0, 0)
            rmse1 = compare(f.astype(float), f_hat1, 0)

            y2 = mat2lpc2d(f, 0.5, 0.5, 0)
            f_hat2 = lpc2mat2d(y2, 0.5, 0.5, 0)
            rmse2 = compare(f.astype(float), f_hat2, 0)

            y3 = mat2lpc2d(f, 0.75, 0.75, -0.5)
            f_hat3 = lpc2mat2d(y3, 0.75, 0.75, -0.5)
            rmse3 = compare(f.astype(float), f_hat3, 0)

            # Display
            fig = plt.figure(1, figsize=(10, 7))

            plt.subplot(2, 3, 1)
            plt.imshow(y1, cmap="gray")
            plt.title(f"RMSE = {rmse1:g}")
            plt.axis("off")

            plt.subplot(2, 3, 2)
            plt.imshow(y2, cmap="gray")
            plt.title(f"RMSE = {rmse2:g}")
            plt.axis("off")

            plt.subplot(2, 3, 3)
            plt.imshow(y3, cmap="gray")
            plt.title(f"RMSE = {rmse3:g}")
            plt.axis("off")

            plt.subplot(2, 3, 4)
            hist1, _ = np.histogram(y1.ravel(), bins=256)
            plt.bar(np.arange(hist1.size), hist1)
            plt.title(f"H = {ntrop(y1.astype(np.int8)):g}")

            plt.subplot(2, 3, 5)
            hist2, _ = np.histogram(y2.ravel(), bins=256)
            plt.bar(np.arange(hist2.size), hist2)
            plt.title(f"H = {ntrop(y2.astype(np.int8)):g}")

            plt.subplot(2, 3, 6)
            hist3, _ = np.histogram(y3.ravel(), bins=256)
            plt.bar(np.arange(hist3.size), hist3)
            plt.title(f"H = {ntrop(y3.astype(np.int8)):g}")

            plt.tight_layout()
            fig.savefig("Figure840.png", dpi=150, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure843(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter08 script `Figure843.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread

            from helpers.wavefast import wavefast
            from helpers.wavedisplay import wavedisplay
            from helpers.data_path import dip_data

            # Parameters
            n_levels = 3

            # Data
            f = imread(dip_data("lena.tif"))
            if f.ndim == 3:
                f = f[..., 0]

            # Fast wavelet transform
            c1, s1 = wavefast(f, n_levels, "haar")
            c2, s2 = wavefast(f, n_levels, "db4")
            c3, s3 = wavefast(f, n_levels, "sym4")
            c4, s4 = wavefast(f, n_levels, "bior6.8")

            # Display + save
            plt.figure(1)
            plt.imshow(wavedisplay(c1, s1), cmap="gray")
            plt.axis("off")
            plt.savefig("Figure843.png", dpi=150, bbox_inches="tight")

            plt.figure(2)
            plt.imshow(wavedisplay(c2, s2), cmap="gray")
            plt.axis("off")
            plt.savefig("Figure843Bis.png", dpi=150, bbox_inches="tight")

            plt.figure(3)
            plt.imshow(wavedisplay(c3, s3), cmap="gray")
            plt.axis("off")
            plt.savefig("Figure843Ter.png", dpi=150, bbox_inches="tight")

            plt.figure(4)
            plt.imshow(wavedisplay(c4, s4), cmap="gray")
            plt.axis("off")
            plt.savefig("Figure843Quart.png", dpi=150, bbox_inches="tight")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure844(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter08 script `Figure844.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.compare import compare
            from helpers.waveback import waveback
            from helpers.wavecopy import wavecopy
            from helpers.wavefast import wavefast
            from helpers.wavepaste import wavepaste
            from helpers.data_path import dip_data

            # Figure 8.44 (dead-zone threshold sweep)

            # Parameters
            n_level = 3  # 3-scale biorthogonal wavelet
            thresholds = np.arange(0, 19)  # dead-zone threshold (width) from 0 to 18

            # Data
            f = imread(dip_data("lena.tif"))
            if f.ndim == 3:
                f = f[..., 0]
            if f.dtype != np.uint8:
                f = np.clip(np.round(f), 0, 255).astype(np.uint8)

            # Wavelet decomposition once (biorthogonal JPEG 9/7).
            c, s = wavefast(f.astype(float) - 128.0, n_level, "jpeg9.7")

            # Count total detail coefficients (h, v, d at all levels) for percentage.
            detail_total = 0
            for k in range(1, n_level + 1):
                detail_total += wavecopy("h", c, s, k).size
                detail_total += wavecopy("v", c, s, k).size
                detail_total += wavecopy("d", c, s, k).size

            # Process
            rms = np.zeros_like(thresholds, dtype=float)
            truncated_pct = np.zeros_like(thresholds, dtype=float)

            for i, T in enumerate(thresholds):
                # Interpret threshold as dead-zone width -> half-width criterion.
                half_width = float(T) / 2.0

                cq = c.copy()
                truncated_count = 0

                # Apply dead-zone truncation to detail subbands only.
                for k in range(1, n_level + 1):
                    for band in ("h", "v", "d"):
                        w = wavecopy(band, cq, s, k)
                        mask = (np.abs(w) <= half_width) & (w != 0)
                        truncated_count += np.count_nonzero(mask)
                        w[mask] = 0.0
                        cq = wavepaste(band, cq, s, k, w)

                truncated_pct[i] = 100.0 * truncated_count / detail_total

                rec = waveback(cq, s, "jpeg9.7")
                rec = np.clip(np.round(rec + 128.0), 0, 255)
                rms[i] = compare(f.astype(float), rec.astype(float), 0)

            # Display
            fig, ax1 = plt.subplots(figsize=(9, 5))
            ax1.plot(thresholds, rms, "k-s", linewidth=1.2, markersize=4)
            ax1.set_xlabel("Dead-zone threshold")
            ax1.set_ylabel("RMS error")
            ax1.set_xticks(np.arange(0, 19, 1))
            ax1.tick_params(axis="y")

            ax2 = ax1.twinx()
            ax2.plot(thresholds, truncated_pct, "k--o", linewidth=1.2, markersize=3)
            ax2.set_ylabel("Truncated detail coefficients (%)")
            ax2.tick_params(axis="y")
            ax2.set_ylim(0, 100)

            ax1.set_title(
                "3-Scale Biorthogonal Wavelet: RMS and Truncated Coefficients vs Dead-zone Threshold"
            )
            fig.tight_layout()
            fig.savefig("Figure844.png", dpi=300, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure846(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter08 script `Figure846.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread

            from helpers.compare import compare
            from helpers.im2jpeg2k import im2jpeg2k
            from helpers.imratio import imratio
            from helpers.jpeg2k2im import jpeg2k2im
            from helpers.data_path import dip_data

            def imcrop_matlab(img: Any, rect: Any):
                """imcrop_matlab."""
                # MATLAB: rect = [x, y, w, h], inclusive integer rectangle.
                x, y, w, h = [int(v) for v in rect]
                return img[y : y + h + 1, x : x + w + 1]

            # Parameters
            mu_b = [8, 8, 8, 8]
            epsilon_b = [8.5, 7, 6.5, 6.0]
            n_level = 5

            # Data
            f = imread(dip_data("lena.tif"))
            if f.ndim == 3:
                f = f[..., 0]
            if f.dtype != np.uint8:
                f = np.clip(np.round(f), 0, 255).astype(np.uint8)

            # Process
            f_hat = []
            compression_ratio = []
            rmse = []

            for i in range(len(mu_b)):
                q = [mu_b[i], epsilon_b[i]]
                y, _ = im2jpeg2k(f, n_level, q)
                compression_ratio.append(imratio(f, y))

                rec = jpeg2k2im(y)
                f_hat.append(rec)
                rmse.append(compare(f.astype(float), rec.astype(float), 0))

            # Display
            fig = plt.figure(1, figsize=(12, 14))
            for i in range(len(mu_b)):
                plt.subplot(4, 3, 1 + i * 3)
                plt.imshow(f_hat[i], cmap="gray")
                plt.title(f"RMSE = {rmse[i]:.2g} Comp. = {compression_ratio[i]:.2g}")
                plt.axis("off")

                plt.subplot(4, 3, 2 + i * 3)
                plt.imshow(f.astype(float) - f_hat[i].astype(float), cmap="gray")
                plt.title("error")
                plt.axis("off")

                plt.subplot(4, 3, 3 + i * 3)
                temp = imcrop_matlab(f_hat[i], [243 - 21, 249 - 21, 64, 64])
                plt.imshow(temp, cmap="gray")
                plt.title("zoom")
                plt.axis("off")

            plt.tight_layout()
            fig.savefig("Figure846.png", dpi=150, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure848(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter08 script `Figure848.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread

            from helpers.im2jpeg import im2jpeg
            from helpers.jpeg2im import jpeg2im
            from helpers.data_path import dip_data

            # %% Figure848

            # %% Parameters
            row = 400
            col = 200

            # %% Data
            f = imread(dip_data("lena.tif"))
            if f.ndim == 3:
                f = f[..., 0]
            f = f.astype(np.uint8)

            # %% Watermark
            w = imread(dip_data("Fig0850(a).tif"))
            if w.ndim == 3:
                w = w[:, :, 0]
            w = w.astype(np.uint8)

            # %% Code it in 2 LSB bits from 0 to 3
            w = np.bitwise_and(w, 3)

            # %% Place it into a 512 by 512 image
            temp = np.zeros_like(f, dtype=np.uint8)
            rr0 = row - 1
            cc0 = col - 1
            temp[rr0 : rr0 + 52, cc0 : cc0 + 153] = w

            # %% Fragile watermarking
            fw = ((f // 4) * 4 + temp).astype(np.uint8)

            # %% Decoding
            w_hat = np.bitwise_and(fw, 3)

            # %% Attack by using JPEG compression and decompression
            fw_attack = jpeg2im(im2jpeg(fw))
            w_attack_hat = np.bitwise_and(fw_attack.astype(np.uint8), 3)

            # %% Display
            fig = plt.figure(1, figsize=(13, 8))

            plt.subplot(2, 3, 1)
            plt.imshow(f, cmap="gray")
            plt.title("Original image")
            plt.axis("off")

            plt.subplot(2, 3, 2)
            plt.imshow(w, cmap="gray", vmin=0, vmax=3)
            plt.title("Watermark on 2 bits")
            plt.axis("off")

            plt.subplot(2, 3, 3)
            plt.imshow(fw, cmap="gray")
            plt.title("Watermarked image")
            plt.axis("off")

            plt.subplot(2, 3, 4)
            plt.imshow(w_hat, cmap="gray", vmin=0, vmax=3)
            plt.title("Watermark detected")
            plt.axis("off")

            plt.subplot(2, 3, 5)
            plt.imshow(fw_attack, cmap="gray")
            plt.title("Watermarked image attacked")
            plt.axis("off")

            plt.subplot(2, 3, 6)
            plt.imshow(w_attack_hat, cmap="gray", vmin=0, vmax=3)
            plt.title("Watermark detected after attack")
            plt.axis("off")

            plt.tight_layout()
            fig.savefig("Figure848.png", dpi=150, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure850(
        self,
        watermark_size: int = 1000,
        seed: int = 123,
        image_name: str = "lena.tif",
        data_dir: str | None = None,
    ) -> dict[str, Any]:
        """Run Chapter08 script `Figure850.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread

            from helpers.data_path import dip_data
            from libDIP.watermark4e import watermark4e

            np.random.seed(seed)

            f = imread(dip_data(image_name))
            if f.ndim == 3:
                f = f[..., 0]
            f = f.astype(np.uint8)

            m1 = np.random.randn(watermark_size)
            m2 = np.random.randn(watermark_size)

            g1, w1 = watermark4e(f, m1)
            diff1 = g1.astype(float) - f.astype(float)

            g2, w2 = watermark4e(f, m2)
            diff2 = g2.astype(float) - f.astype(float)

            r1 = self._compute_correlation(g1, w1)
            r2 = self._compute_correlation(g2, w2)

            fig = plt.figure(1, figsize=(10, 8))

            plt.subplot(2, 2, 1)
            plt.imshow(g1, cmap="gray")
            plt.title(f"Watermarked, r = {r1:.3g}")
            plt.axis("off")

            plt.subplot(2, 2, 2)
            plt.imshow(diff1, cmap="gray")
            plt.title(f"Difference (Max) = {np.max(np.abs(diff1)):.6g}")
            plt.axis("off")

            plt.subplot(2, 2, 3)
            plt.imshow(g2, cmap="gray")
            plt.title(f"Watermarked, r = {r2:.3g}")
            plt.axis("off")

            plt.subplot(2, 2, 4)
            plt.imshow(diff2, cmap="gray")
            plt.title(f"Difference (Max) = {np.max(np.abs(diff2)):.6g}")
            plt.axis("off")

            plt.tight_layout()

            return {
                "image": f,
                "g1": g1,
                "g2": g2,
                "diff1": diff1,
                "diff2": diff2,
                "r1": r1,
                "r2": r2,
                "w1": w1,
                "w2": w2,
                "figures": [fig],
            }
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)

    def figure851(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter08 script `Figure851.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            """Figure 8.51 - Watermark attacks and correlation display."""

            import os
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.watermarkMKR import watermarkMKR
            from helpers.data_path import dip_data

            print("Running Figure851 (watermark attack comparison)...")

            # Parameters
            WatermarkSize = 1000
            HSize = 21
            Sigma = 7
            Alpha = 0.1

            # Data (fixed path)
            f_path = dip_data("lena.tif")
            f = imread(f_path)
            if f.ndim == 3:
                f = f[..., 0]
            NR, NC = f.shape

            # Watermark sequence
            w = np.random.randn(WatermarkSize, 1)

            # Add watermark with attacks
            wi = {}
            d = {}
            c = {}

            wi["JPEG70"], d["JPEG70"], c["JPEG70"] = watermarkMKR(
                f, w, Alpha, "jpeg70", "same"
            )
            wi["JPEG10"], d["JPEG10"], c["JPEG10"] = watermarkMKR(
                f, w, Alpha, "jpeg10", "same"
            )
            wi["Filter"], d["Filter"], c["Filter"] = watermarkMKR(
                f, w, Alpha, "filter", "same"
            )
            wi["noise"], d["noise"], c["noise"] = watermarkMKR(
                f, w, Alpha, "noise", "same"
            )
            wi["HistEq"], d["HistEq"], c["HistEq"] = watermarkMKR(
                f, w, Alpha, "heq", "same"
            )
            wi["Rotate"], d["Rotate"], c["Rotate"] = watermarkMKR(
                f, w, Alpha, "rotate", "same"
            )

            # Display
            fig = plt.figure(1, figsize=(11, 7))

            keys = ["JPEG70", "JPEG10", "Filter", "noise", "HistEq", "Rotate"]
            for i, k in enumerate(keys, start=1):
                ax = fig.add_subplot(2, 3, i)
                ax.imshow(wi[k], cmap="gray")
                ax.set_title(f"c = {c[k]:.4f}")
                ax.axis("off")

            # Save
            out_path = os.path.join(os.path.join(str(_Path(__file__).resolve().parents[2]), "output"), "Figure851.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved {out_path}")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure89(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter08 script `Figure89.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.data_path import dip_data

            # Data
            f = imread(dip_data("lena.tif"))

            # Display
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(f, cmap="gray")
            axes[0].axis("off")

            counts, edges = np.histogram(f.ravel().astype(float), bins=256)
            centers = (edges[:-1] + edges[1:]) / 2
            axes[1].bar(centers, counts, width=edges[1] - edges[0])
            axes[1].set_aspect("auto")
            axes[1].set_box_aspect(1)
            axes[1].set_anchor("C")

            fig.subplots_adjust(wspace=0.3)
            plt.savefig("Figure89.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figuretifs2cv(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter08 script `FigureTifs2cv.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread

            from helpers.compare import compare
            from helpers.cv2tifs import cv2tifs
            from helpers.imratio import imratio
            from helpers.tifs2cv import tifs2cv
            from helpers.data_path import dip_data

            # %% FigureTifs2cv

            # %% Parameters (edit these)
            input_tif = dip_data("shuttle.tif")
            output_tif = dip_data("shuttlereconstructed_sequence.tif")
            m = 8
            d = [16, 8]
            q = 1  # q=0 lossless residual coding, q>0 lossy JPEG residual coding

            # %% Helpers

            def load_frames(path: Any):
                """load_frames."""
                arr = np.asarray(imread(path))
                if arr.ndim == 2:
                    return [arr]
                if arr.ndim == 3:
                    if arr.shape[-1] in (3, 4) and arr.shape[0] != arr.shape[1]:
                        raise ValueError(
                            "Expected a grayscale TIFF sequence, got color data."
                        )
                    return [arr[i, :, :] for i in range(arr.shape[0])]
                raise ValueError("Unsupported TIFF dimensions.")

            # %% Data
            original = load_frames(input_tif)

            # %% Process
            y = tifs2cv(input_tif, m, d, q)
            compression_ratio = imratio(input_tif, y)
            reconstructed = cv2tifs(y, output_tif)

            if len(original) != len(reconstructed):
                raise RuntimeError(
                    "Frame count mismatch between original and reconstructed."
                )

            rmse = []
            for i in range(len(original)):
                rmse.append(
                    compare(
                        original[i].astype(float), reconstructed[i].astype(float), 0
                    )
                )

            print(f"Input file: {input_tif}")
            print(f"Output file: {output_tif}")
            print(f"Frames: {len(original)}")
            print(f"m={m}, d={d}, q={q}")
            print(f"Compression ratio: {compression_ratio:.4f}")
            print(f"Mean RMSE: {np.mean(rmse):.4f}")
            print(f"Max RMSE: {np.max(rmse):.4f}")

            # %% Display
            show_n = min(3, len(original))
            fig = plt.figure(1, figsize=(12, 4 * show_n))
            for i in range(show_n):
                plt.subplot(show_n, 3, 1 + 3 * i)
                plt.imshow(original[i], cmap="gray")
                plt.title(f"Original frame {i + 1}")
                plt.axis("off")

                plt.subplot(show_n, 3, 2 + 3 * i)
                plt.imshow(reconstructed[i], cmap="gray")
                plt.title(f"Reconstructed frame {i + 1} (RMSE={rmse[i]:.3f})")
                plt.axis("off")

                plt.subplot(show_n, 3, 3 + 3 * i)
                err = original[i].astype(float) - reconstructed[i].astype(float)
                plt.imshow(err, cmap="gray")
                plt.title(f"Error frame {i + 1}")
                plt.axis("off")

            plt.tight_layout()
            fig.savefig("FigureTifs2cv.png", dpi=150, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def fig81bc(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter08 script `fig81bc.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np

            def fig81bc(part: Any, s: Any = 256, n: Any = None, p: Any = None):
                """
                Figure 8.1(b,c) generator.
                (b) s x s image with s gray-level lines (uniform distribution)
                (c) s x s image with random gray levels n at p points on medium gray field
                """
                if n is None:
                    n = np.array([125, 126, 127, 129, 130, 131], dtype=np.uint8)
                    p = np.array([1935, 5123, 9997, 7652, 4755, 1877], dtype=int)
                    jend = 6
                else:
                    n = np.asarray(n, dtype=np.uint8)
                    p = np.asarray(p, dtype=int)
                    if n.shape != p.shape:
                        raise ValueError("n and p must be the same size!")
                    jend = n.shape[0] if n.ndim == 1 else n.shape[1]

                img = np.zeros((s, s), dtype=np.uint8)

                if part == "b":
                    gl = np.arange(s, dtype=np.uint8)
                    for k in range(s):
                        r = int(np.ceil(len(gl) * np.random.rand())) - 1
                        img[k, :] = gl[r]
                        gl = np.delete(gl, r)
                else:
                    img[:, :] = 128
                    for j in range(jend):
                        for _ in range(int(p[j])):
                            img[
                                int(np.ceil(s * np.random.rand())) - 1,
                                int(np.ceil(s * np.random.rand())) - 1,
                            ] = n[j]

                    for k in range(50, 81):
                        img[:, k] = 128

                    for j in range(50, 81):
                        img[j, :] = 128

                return img

        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)
