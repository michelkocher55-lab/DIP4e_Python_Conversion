from DIP4eFigures.core import dip_data

from pathlib import Path as _Path
import os as _os


class Chapter02Mixin:
    def figureXXX(self, data_dir: str | None = None) -> dict[str, object]:
        """Run Chapter02 script `FigureXXX.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(
            data_dir=data_dir
        )
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread

            img_name = dip_data("Chronometer.tif")
            f = imread(img_name)
            original_shape = f.shape[:2]
            from helpers.libdip.xxx4e import xxx4e

            g = xxx4e(f, 11, 11)
            # g = 255 - f
            _, axes = plt.subplots(1, 2, figsize=(10, 10))
            axes = axes.flatten()

            axes[0].imshow(f, cmap="gray")
            axes[0].set_title(f"Original ({original_shape})")
            axes[0].axis("off")

            axes[1].imshow(g, cmap="gray")
            axes[1].set_title(f"aMean4e ({original_shape})")
            axes[1].axis("off")

            plt.tight_layout()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure223(self, data_dir: str | None = None) -> dict[str, object]:
        """Run Chapter02 script `Figure223.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(
            data_dir=data_dir
        )
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.transform import resize

            img_name = dip_data("Chronometer.tif")
            f = imread(img_name)
            original_shape = f.shape[:2]

            print("Reducing to 300 dpi...")
            gnn1 = resize(
                f,
                (689, 690),
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            )

            print("Reducing to 150 dpi...")
            gnn2 = resize(
                f,
                (345, 345),
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            )

            print("Reducing to 72 dpi...")
            gnn3 = resize(
                f,
                (165, 166),
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            )

            print("Resizing back to original...")
            fr300dpi = resize(
                gnn1,
                original_shape,
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            )
            fr150dpi = resize(
                gnn2,
                original_shape,
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            )
            fr72dpi = resize(
                gnn3,
                original_shape,
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            )

            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes = axes.flatten()

            axes[0].imshow(f, cmap="gray")
            axes[0].set_title("Original")
            axes[0].axis("off")

            axes[1].imshow(fr300dpi.astype("uint8"), cmap="gray")
            axes[1].set_title("Resol: 300 dpi (Nearest)")
            axes[1].axis("off")

            axes[2].imshow(fr150dpi.astype("uint8"), cmap="gray")
            axes[2].set_title("Resol: 150 dpi (Nearest)")
            axes[2].axis("off")

            axes[3].imshow(fr72dpi.astype("uint8"), cmap="gray")
            axes[3].set_title("Resol: 72 dpi (Nearest)")
            axes[3].axis("off")

            plt.tight_layout()
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure224(self, data_dir: str | None = None) -> dict[str, object]:
        """Run Chapter02 script `Figure224.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(
            data_dir=data_dir
        )
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_ubyte
            from helpers.libdipum.im2bitplanes import im2bitplanes
            from helpers.libdipum.bitplanes2im import bitplanes2im

            img_name = dip_data("drip-bottle.tif")
            f256 = imread(img_name)
            B = im2bitplanes(f256, 8)
            f128 = img_as_ubyte(bitplanes2im(B, [1, 2, 3, 4, 5, 6, 7]))
            f64 = img_as_ubyte(bitplanes2im(B, range(2, 8)))
            f32 = img_as_ubyte(bitplanes2im(B, range(3, 8)))
            f16 = img_as_ubyte(bitplanes2im(B, range(4, 8)))
            f8 = img_as_ubyte(bitplanes2im(B, range(5, 8)))
            f4 = img_as_ubyte(bitplanes2im(B, range(6, 8)))
            f2 = img_as_ubyte(bitplanes2im(B, [7]))

            fig1, axes1 = plt.subplots(2, 2, figsize=(10, 10))
            axes1 = axes1.flatten()
            axes1[0].imshow(f256, cmap="gray")
            axes1[0].set_title("Original (8 bits)")
            axes1[0].axis("off")
            axes1[1].imshow(f128, cmap="gray")
            axes1[1].set_title("7 bits")
            axes1[1].axis("off")
            axes1[2].imshow(f64, cmap="gray")
            axes1[2].set_title("6 bits")
            axes1[2].axis("off")
            axes1[3].imshow(f32, cmap="gray")
            axes1[3].set_title("5 bits")
            axes1[3].axis("off")
            plt.tight_layout()
            plt.savefig("Figure224.png")
            print("Saved Figure224.png")

            fig2, axes2 = plt.subplots(2, 2, figsize=(10, 10))
            axes2 = axes2.flatten()
            axes2[0].imshow(f16, cmap="gray")
            axes2[0].set_title("4 bits")
            axes2[0].axis("off")
            axes2[1].imshow(f8, cmap="gray")
            axes2[1].set_title("3 bits")
            axes2[1].axis("off")
            axes2[2].imshow(f4, cmap="gray")
            axes2[2].set_title("2 bits")
            axes2[2].axis("off")
            axes2[3].imshow(f2, cmap="gray")
            axes2[3].set_title("1 bit")
            axes2[3].axis("off")
            plt.tight_layout()
            plt.savefig("Figure224Bis.png")
            print("Saved Figure224Bis.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure227(self, data_dir: str | None = None) -> dict[str, object]:
        """Run Chapter02 script `Figure227.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(
            data_dir=data_dir
        )
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.transform import resize

            from helpers.libdipum.data_path import dip_data

            img_name = dip_data("Chronometer.tif")
            f = imread(img_name)
            target_shape = (165, 166)
            original_shape = f.shape[:2]

            print("Processing Nearest Neighbor...")
            g72nn = resize(
                f,
                target_shape,
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            )
            f72zoomnn = resize(
                g72nn,
                original_shape,
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            )

            print("Processing Bilinear...")
            g72bl = resize(f, target_shape, order=1, preserve_range=True)
            f72zoombl = resize(
                g72bl, original_shape, order=1, preserve_range=True
            )

            print("Processing Bicubic...")
            g72bc = resize(f, target_shape, order=3, preserve_range=True)
            f72zoombc = resize(
                g72bc, original_shape, order=3, preserve_range=True
            )

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes = axes.flatten()
            axes[0].imshow(f72zoomnn.astype("uint8"), cmap="gray")
            axes[0].set_title("Nearest Neighbor")
            axes[0].axis("off")
            axes[1].imshow(f72zoombl.astype("uint8"), cmap="gray")
            axes[1].set_title("Bilinear")
            axes[1].axis("off")
            axes[2].imshow(f72zoombc.astype("uint8"), cmap="gray")
            axes[2].set_title("Bicubic")
            axes[2].axis("off")
            plt.tight_layout()
            plt.savefig("Figure227.png")
            print("Saved Figure227.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure229(self, data_dir: str | None = None) -> dict[str, object]:
        """Run Chapter02 script `Figure229.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(
            data_dir=data_dir
        )
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from helpers.libdipum.averaging4noisereduction import (
                averaging4noisereduction,
            )
            from helpers.libdip.intScaling4e import intScaling4e
            from helpers.libdipum.data_path import dip_data

            img_path = dip_data("sombrero-galaxy-original.tif")
            forig = img_as_float(imread(img_path))
            print("Generating noisy image results...")
            noisy_pure = averaging4noisereduction(forig, 1, "gaussian", 0, 64)
            forig_noisy = intScaling4e(noisy_pure)

            ks = [10, 50, 100, 500, 1000]
            results = []
            for k in ks:
                print(f"Averaging {k} images...")
                res = averaging4noisereduction(forig, k, "gaussian", 0, 64)
                results.append(intScaling4e(res))

            fav10, fav50, fav100, fav500, fav1000 = results

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            axes[0].imshow(forig_noisy, cmap="gray")
            axes[0].set_title("Noisy Image (K=1)")
            axes[0].axis("off")
            axes[1].imshow(fav10, cmap="gray")
            axes[1].set_title("Average of 10 Images")
            axes[1].axis("off")
            axes[2].imshow(fav50, cmap="gray")
            axes[2].set_title("Average of 50 Images")
            axes[2].axis("off")
            axes[3].imshow(fav100, cmap="gray")
            axes[3].set_title("Average of 100 Images")
            axes[3].axis("off")
            axes[4].imshow(fav500, cmap="gray")
            axes[4].set_title("Average of 500 Images")
            axes[4].axis("off")
            axes[5].imshow(fav1000, cmap="gray")
            axes[5].set_title("Average of 1000 Images")
            axes[5].axis("off")
            plt.tight_layout()
            plt.savefig("Figure229.png")
            print("Saved Figure229.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure231(self, data_dir: str | None = None) -> dict[str, object]:
        """Run Chapter02 script `Figure231.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(
            data_dir=data_dir
        )
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            import ia870 as ia
            from helpers.libdipum.data_path import dip_data

            forig = imread(
                dip_data("chronometer-2136x2140-2pt3-inch-930-dpi.tif")
            )
            f300 = imread(
                dip_data("chronometer-689x690-2pt3-inch-300-dpi.tif")
            )
            f150 = imread(
                dip_data("chronometer-345x345-2pt3-inch-150-dpi.tif")
            )
            f72 = imread(dip_data("chronometer-165x166-2pt3-inch-72-dpi.tif"))

            d300 = np.clip(
                forig.astype(float) - f300.astype(float), 0, 255
            ).astype(np.uint8)
            d150 = np.clip(
                forig.astype(float) - f150.astype(float), 0, 255
            ).astype(np.uint8)
            d72 = np.clip(
                forig.astype(float) - f72.astype(float), 0, 255
            ).astype(np.uint8)

            B = ia.iasebox(2)
            gd72 = ia.iadil(d72, B)
            gd150 = ia.iadil(d150, B)
            gd300 = ia.iadil(d300, B)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes = axes.flatten()
            axes[0].imshow(gd72, cmap="gray")
            axes[0].set_title("Dilated Diff (72 dpi)")
            axes[0].axis("off")
            axes[1].imshow(gd150, cmap="gray")
            axes[1].set_title("Dilated Diff (150 dpi)")
            axes[1].axis("off")
            axes[2].imshow(gd300, cmap="gray")
            axes[2].set_title("Dilated Diff (300 dpi)")
            axes[2].axis("off")
            plt.tight_layout()
            plt.savefig("Figure231.png")
            print("Saved Figure231.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure232(self, data_dir: str | None = None) -> dict[str, object]:
        """Run Chapter02 script `Figure232.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(
            data_dir=data_dir
        )
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from helpers.libdipum.intensityTransformations import (
                intensityTransformations,
            )
            from helpers.libgeneral.mat2gray import mat2gray
            from helpers.libdipum.data_path import dip_data

            mask = img_as_float(
                imread(dip_data("angiography-mask-image.tif"))
            )
            live = img_as_float(
                imread(dip_data("angiography-live-image.tif"))
            )
            d = mat2gray(live - mask)
            m = np.mean(d)
            g = intensityTransformations(d, "stretch", m - 0.03, 10)

            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes = axes.flatten()
            axes[0].imshow(mask, cmap="gray")
            axes[0].set_title("Mask")
            axes[0].axis("off")
            axes[1].imshow(live, cmap="gray")
            axes[1].set_title("Live")
            axes[1].axis("off")
            axes[2].imshow(d, cmap="gray")
            axes[2].set_title("Difference (mat2gray)")
            axes[2].axis("off")
            axes[3].imshow(g, cmap="gray")
            axes[3].set_title("Enhanced (Stretch)")
            axes[3].axis("off")
            plt.tight_layout()
            plt.savefig("Figure232.png")
            print("Saved Figure232.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure234(self, data_dir: str | None = None) -> dict[str, object]:
        """Run Chapter02 script `Figure234.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(
            data_dir=data_dir
        )
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float

            f = img_as_float(imread(dip_data("dentalXray.tif")))
            h = img_as_float(imread(dip_data("dentalXrayMask.tif")))

            if f.shape != h.shape:
                print(
                    "Warning: Image and mask have different dimensions. Resizing mask might be needed."
                )

            g = f * h

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes = axes.flatten()
            axes[0].imshow(f, cmap="gray")
            axes[0].set_title("Original Image (f)")
            axes[0].axis("off")
            axes[1].imshow(h, cmap="gray")
            axes[1].set_title("Mask (h)")
            axes[1].axis("off")
            axes[2].imshow(g, cmap="gray")
            axes[2].set_title("Product (g = f * h)")
            axes[2].axis("off")
            plt.tight_layout()
            plt.savefig("Figure234.png")
            print("Saved Figure234.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure236(self, data_dir: str | None = None) -> dict[str, object]:
        """Run Chapter02 script `Figure236.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(
            data_dir=data_dir
        )
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float

            img_path = dip_data("skeleton.tif")
            A = img_as_float(imread(img_path))
            An = 1.0 - A
            B_val = 3 * np.mean(A)
            Union = np.maximum(A, B_val)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes = axes.flatten()
            axes[0].imshow(A, cmap="gray")
            axes[0].set_title("Original Image (A)")
            axes[0].axis("off")
            axes[1].imshow(An, cmap="gray")
            axes[1].set_title("Complement (1 - A)")
            axes[1].axis("off")
            axes[2].imshow(Union, cmap="gray", vmin=0, vmax=1)
            axes[2].set_title(
                f"Union (max(A, 3*mean(A))) \n3*mean(A)={B_val:.2f}"
            )
            axes[2].axis("off")
            plt.tight_layout()
            plt.savefig("Figure236.png")
            print("Saved Figure236.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure238(self, data_dir: str | None = None) -> dict[str, object]:
        """Run Chapter02 script `Figure238.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(
            data_dir=data_dir
        )
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import invert

            f = imread(dip_data("Chronometer.tif"))
            g = invert(f)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes = axes.flatten()
            axes[0].imshow(f, cmap="gray")
            axes[0].set_title("Original Image")
            axes[0].axis("off")
            axes[1].imshow(g, cmap="gray")
            axes[1].set_title("Complement (Negative)")
            axes[1].axis("off")
            plt.tight_layout()
            plt.savefig("Figure238.png")
            print("Saved Figure238.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure239(self, data_dir: str | None = None) -> dict[str, object]:
        """Run Chapter02 script `Figure239.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(
            data_dir=data_dir
        )
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from scipy.ndimage import uniform_filter

            f = imread(dip_data("angiogram-aortic-kidney.tif"))
            g_float = uniform_filter(f.astype(float), size=41, mode="reflect")
            g = np.clip(g_float, 0, 255).astype(f.dtype)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes = axes.flatten()
            axes[0].imshow(f, cmap="gray")
            axes[0].set_title("Original Image")
            axes[0].axis("off")
            axes[1].imshow(g, cmap="gray")
            axes[1].set_title("Smoothed Image (41x41 Average)")
            axes[1].axis("off")
            plt.tight_layout()
            plt.savefig("Figure239.png")
            print("Saved Figure239.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure240(self, data_dir: str | None = None) -> dict[str, object]:
        """Run Chapter02 script `Figure240.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(
            data_dir=data_dir
        )
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.transform import rotate
            from skimage.util import img_as_float

            f_orig = imread(dip_data("letterT.tif"))
            if f_orig.ndim == 3:
                f_orig = f_orig[:, :, 0]
            f = img_as_float(f_orig[::8, ::8])

            frn = rotate(f, -21, resize=False, order=0)
            frbl = rotate(f, -21, resize=False, order=1)
            frbc = rotate(f, -21, resize=False, order=3)

            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes[0, 0].imshow(f, cmap="gray")
            axes[0, 0].set_title("Original (Subsampled)")
            axes[0, 0].axis("off")
            axes[0, 1].imshow(frn, cmap="gray")
            axes[0, 1].set_title("Nearest Neighbor")
            axes[0, 1].axis("off")
            axes[1, 0].imshow(frbl, cmap="gray")
            axes[1, 0].set_title("Bilinear")
            axes[1, 0].axis("off")
            axes[1, 1].imshow(frbc, cmap="gray")
            axes[1, 1].set_title("Bicubic")
            axes[1, 1].axis("off")
            plt.tight_layout()
            plt.savefig("Figure240.png")
            print("Saved Figure240.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure242(self, data_dir: str | None = None) -> dict[str, object]:
        """Run Chapter02 script `Figure242.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(
            data_dir=data_dir
        )
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from skimage.transform import (
                AffineTransform,
                warp,
                estimate_transform,
            )

            f_orig = imread(dip_data("characterTestPattern688.tif"))
            if f_orig.ndim == 3:
                f_orig = f_orig[:, :, 0]
            f = img_as_float(f_orig)
            h, w = f.shape

            shear_matrix = np.array([[1, 0.05, 0], [0.4, 1, 0], [0, 0, 1]])
            corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])
            tform_shear = AffineTransform(matrix=shear_matrix)
            corners_transformed = tform_shear(corners)
            min_x = corners_transformed[:, 0].min()
            max_x = corners_transformed[:, 0].max()
            min_y = corners_transformed[:, 1].min()
            max_y = corners_transformed[:, 1].max()
            out_w = int(np.ceil(max_x - min_x))
            out_h = int(np.ceil(max_y - min_y))

            m_inv_shear = np.linalg.inv(shear_matrix)
            m_trans = np.array([[1, 0, min_x], [0, 1, min_y], [0, 0, 1]])
            map_matrix = m_inv_shear @ m_trans
            tform_warp = AffineTransform(matrix=map_matrix)
            gd = warp(f, tform_warp, output_shape=(out_h, out_w), order=1)

            base_points = np.array(
                [
                    [114.9692, 109.4923],
                    [618.2154, 75.7538],
                    [600.3538, 610.9385],
                    [75.0923, 633.4308],
                ]
            )
            input_points = np.array(
                [
                    [118.5971, 155.0846],
                    [623.2462, 322.6837],
                    [629.7279, 851.9587],
                    [104.7077, 663.9885],
                ]
            )
            tform_reg = estimate_transform(
                "affine", src=input_points, dst=base_points
            )
            g = warp(gd, tform_reg.inverse, output_shape=f.shape, order=1)
            dif_abs = np.abs(f - g)

            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes[0, 0].imshow(f, cmap="gray")
            axes[0, 0].set_title("Original Image f")
            axes[0, 0].axis("off")
            axes[0, 1].imshow(gd, cmap="gray")
            axes[0, 1].set_title("Geometrically Distorted gd")
            axes[0, 1].axis("off")
            axes[1, 0].imshow(g, cmap="gray")
            axes[1, 0].set_title("Recovered Image g")
            axes[1, 0].axis("off")
            axes[1, 1].imshow(dif_abs, cmap="gray")
            axes[1, 1].set_title("Difference |f - g|")
            axes[1, 1].axis("off")
            plt.tight_layout()
            plt.savefig("Figure242.png")
            print("Saved Figure242.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure245(self, data_dir: str | None = None) -> dict[str, object]:
        """Run Chapter02 script `Figure245.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(
            data_dir=data_dir
        )
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from scipy.fft import fft2, fftshift
            from helpers.libdipum.imnoise3 import imnoise3
            from helpers.libdipum.cnotch import cnotch
            from helpers.libdip.intScaling4e import intScaling4e
            from helpers.libdipum.dftfilt import dftfilt

            f = img_as_float(imread(dip_data("astronaut.tif")))
            M, N = f.shape
            r, R, S = imnoise3(M, N, [[25, 25]], [0.3])
            g = f + r
            gs = intScaling4e(g)
            F = fft2(g)
            G = fftshift(np.abs(F))
            Glog = intScaling4e(1 + np.log(G))
            u_impulse = M // 2 + 25
            v_impulse = N // 2 + 25
            H = cnotch("ideal", "reject", M, N, [[u_impulse, v_impulse]], 2)
            Hc = intScaling4e(fftshift(H))
            gf = intScaling4e(dftfilt(g, H))

            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes = axes.flatten()
            axes[0].imshow(gs, cmap="gray")
            axes[0].set_title("Noisy Image")
            axes[0].axis("off")
            axes[1].imshow(Glog, cmap="gray")
            axes[1].set_title("Spectrum (Log)")
            axes[1].axis("off")
            axes[2].imshow(Hc, cmap="gray")
            axes[2].set_title("Notch Filter")
            axes[2].axis("off")
            axes[3].imshow(gf, cmap="gray")
            axes[3].set_title("Restored Image")
            axes[3].axis("off")
            plt.tight_layout()
            plt.savefig("Figure245.png")
            print("Saved Figure245.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure257(self, data_dir: str | None = None) -> dict[str, object]:
        """Run Chapter02 script `Figure257.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(
            data_dir=data_dir
        )
        try:
            import numpy as np
            import matplotlib.pyplot as plt

            m = [10, 10]
            C1 = [[5, 0], [0, 5]]
            C2 = [[12, 0], [0, 5]]
            C3 = [[12, 6], [6, 5]]
            C4 = [[12, -6], [-6, 5]]
            n_samples = 1000

            r1 = np.random.multivariate_normal(m, C1, n_samples)
            r2 = np.random.multivariate_normal(m, C2, n_samples)
            r3 = np.random.multivariate_normal(m, C3, n_samples)
            r4 = np.random.multivariate_normal(m, C4, n_samples)

            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes = axes.flatten()
            limit = [0, 20, 0, 20]
            axes[0].plot(r1[:, 0], r1[:, 1], ".")
            axes[0].axis(limit)
            axes[0].set_aspect("equal", adjustable="box")
            axes[0].set_title("Covariance: [5 0; 0 5]")
            axes[1].plot(r2[:, 0], r2[:, 1], ".")
            axes[1].axis(limit)
            axes[1].set_aspect("equal", adjustable="box")
            axes[1].set_title("Covariance: [12 0; 0 5]")
            axes[2].plot(r3[:, 0], r3[:, 1], ".")
            axes[2].axis(limit)
            axes[2].set_aspect("equal", adjustable="box")
            axes[2].set_title("Covariance: [12 6; 6 5]")
            axes[3].plot(r4[:, 0], r4[:, 1], ".")
            axes[3].axis(limit)
            axes[3].set_aspect("equal", adjustable="box")
            axes[3].set_title("Covariance: [12 -6; -6 5]")
            plt.tight_layout()
            plt.savefig("Figure257.png")
            print("Saved Figure257.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)
