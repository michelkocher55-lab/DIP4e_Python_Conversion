from pathlib import Path as _Path
import os as _os

from typing import Any


class Chapter11Mixin:
    def figure1110(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter11 script `Figure1110.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.libdip.snakeMap4e import snakeMap4e
            from helpers.libdip.snakeForce4e import snakeForce4e
            from helpers.libdip.snakeIterate4e import snakeIterate4e
            from helpers.libdip.snakeReparam4e import snakeReparam4e
            from helpers.libdipum.snake_display import snake_display
            from helpers.libdip.intScaling4e import intScaling4e
            from helpers.libdipum.data_path import dip_data

            # Parameters
            NIter = 400

            # Data
            img_path = dip_data("U200.tif")
            f = imread(img_path)

            # Specify a circle
            t = np.arange(0, 2 * np.pi + 0.1, 0.1)
            xi = 100 + 80 * np.cos(t)
            yi = 100 + 80 * np.sin(t)

            # Close the snake
            xi = np.concatenate([xi, [xi[0]]])
            yi = np.concatenate([yi, [yi[0]]])

            # Edge map
            emap = snakeMap4e(f, 0.001, 3, 1, "after")

            # Scale to range [0,1]
            emap = intScaling4e(emap)

            # Snake force (GVF)
            FTx, FTy = snakeForce4e(emap, "gvf", 0.25, 80)

            # Normalize
            mag = np.sqrt(FTx**2 + FTy**2)
            FTx = FTx / (mag + np.finfo(float).eps)
            FTy = FTy / (mag + np.finfo(float).eps)

            x = xi.copy()
            y = yi.copy()

            # Iterate
            for _ in range(NIter):
                x, y = snakeIterate4e(0.05, 0.5, 5, x, y, 1, FTx, FTy)
                x, y = snakeReparam4e(x, y)

            # Redistribute the points one last time
            x, y = snakeReparam4e(x, y)

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))

            axes[0, 0].imshow(f, cmap="gray")
            axes[0, 0].axis("off")
            axes[0, 0].plot(np.append(yi, yi[0]), np.append(xi, xi[0]), "g.")

            axes[0, 1].imshow(emap, cmap="gray")
            axes[0, 1].axis("off")

            axes[1, 0].quiver(np.flipud(FTy[::2, ::2]), np.flipud(-FTx[::2, ::2]))
            axes[1, 0].axis("off")

            axes[1, 1].imshow(f, cmap="gray")
            axes[1, 1].axis("off")
            plt.sca(axes[1, 1])
            snake_display(x, y, "g.")

            plt.tight_layout()
            plt.savefig("Figure1110.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1112(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter11 script `Figure1112.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import os
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from scipy.io import loadmat, savemat
            from helpers.libdipum.snake_manual_input import snake_manual_input
            from helpers.libdip.snakeMap4e import snakeMap4e
            from helpers.libdip.snakeForce4e import snakeForce4e
            from helpers.libdip.snakeIterate4e import snakeIterate4e
            from helpers.libdip.snakeReparam4e import snakeReparam4e
            from helpers.libdipum.snake_display import snake_display
            from helpers.libdip.intScaling4e import intScaling4e
            from helpers.libdipum.data_path import dip_data

            # Parameters
            T = 0.005
            Sig = 11
            NSig = 5
            r = 10
            NIter = 100

            # Data
            img_path = dip_data("rose957by1024.tif")
            f = imread(img_path)

            if os.path.exists("Figure118.mat"):
                mat = loadmat("Figure118.mat")
                xi = mat["xi"].squeeze()
                yi = mat["yi"].squeeze()
            else:
                xi, yi = snake_manual_input(f, 150, "ro")
                savemat("Figure118.mat", {"xi": xi, "yi": yi})

            # Edge map
            emap = snakeMap4e(f, T, Sig, NSig, "both")

            # Scale to [0,1]
            emap = intScaling4e(emap)

            # Snake force (gradient)
            FTxa, FTya = snakeForce4e(emap, "gradient")

            # Normalize
            maga = np.sqrt(FTxa**2 + FTya**2)
            FTxa = FTxa / (maga + 1e-10)
            FTya = FTya / (maga + 1e-10)

            # Snake force (GVF)
            FTxb, FTyb = snakeForce4e(emap, "gvf", 0.2, 160)

            # Normalize
            magb = np.sqrt(FTxb**2 + FTyb**2)
            FTxb = FTxb / (magb + 1e-10)
            FTyb = FTyb / (magb + 1e-10)

            xa = xi.copy()
            ya = yi.copy()
            xb = xi.copy()
            yb = yi.copy()

            # Iterate
            for _ in range(150):
                xa, ya = snakeIterate4e(1, 0.5, 5, xa, ya, 1, FTxa, FTya)
                xa, ya = snakeReparam4e(xa, ya)

                xb, yb = snakeIterate4e(1, 0.5, 5, xb, yb, 1, FTxb, FTyb)
                xb, yb = snakeReparam4e(xb, yb)

            xa, ya = snakeReparam4e(xa, ya)
            xb, yb = snakeReparam4e(xb, yb)

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))

            axes[0, 0].quiver(
                np.flipud(FTya[::r, ::r]),
                np.flipud(-FTxa[::r, ::r]),
                angles="xy",
                scale_units="xy",
                scale=1,
            )
            axes[0, 0].axis("off")

            axes[1, 0].imshow(f, cmap="gray")
            axes[1, 0].axis("off")
            plt.sca(axes[1, 0])
            snake_display(xa, ya, "g.")

            axes[0, 1].quiver(
                np.flipud(FTyb[::r, ::r]),
                np.flipud(-FTxb[::r, ::r]),
                angles="xy",
                scale_units="xy",
                scale=1,
            )
            axes[0, 1].axis("off")

            axes[1, 1].imshow(f, cmap="gray")
            axes[1, 1].axis("off")
            plt.sca(axes[1, 1])
            snake_display(xb, yb, "g.")

            plt.tight_layout()
            plt.savefig("Figure1112.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1113(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter11 script `Figure1113.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.libdip.snakeMap4e import snakeMap4e
            from helpers.libdip.snakeForce4e import snakeForce4e
            from helpers.libdip.snakeIterate4e import snakeIterate4e
            from helpers.libdip.snakeReparam4e import snakeReparam4e
            from helpers.libdipum.snake_display import snake_display
            from helpers.libdip.intScaling4e import intScaling4e
            from helpers.libdipum.data_path import dip_data

            # Parameters
            NIter = [10, 20, 40, 60, 80]
            T = 0.01
            Sig = 11
            NSig = 5

            # Data
            img_path = dip_data("breast-implant.tif")
            f = imread(img_path)

            # Edge map
            emap = snakeMap4e(f, T, Sig, NSig, "both")
            # Scale to [0, 1] range
            emap = intScaling4e(emap)

            # Initial contour
            t = np.arange(0, 2 * np.pi + 0.05, 0.05)
            xi = 300 + 70 * np.cos(t)
            yi = 360 + 90 * np.sin(t)

            # Close the snake
            xi = np.concatenate([xi, [xi[0]]])
            yi = np.concatenate([yi, [yi[0]]])

            # Snake force
            FTx, FTy = snakeForce4e(emap, "gvf", 0.2, 160)

            # Normalize
            mag = np.sqrt(FTx**2 + FTy**2)
            FTx = FTx / (mag + 1e-10)
            FTy = FTy / (mag + 1e-10)

            def script_for_fig1113(xi: Any, yi: Any, FTx: Any, FTy: Any, n_iter: Any):
                """script_for_fig1113."""
                x = xi.copy()
                y = yi.copy()
                for _ in range(n_iter):
                    x, y = snakeIterate4e(0.05, 0.5, 2.5, x, y, 1, FTx, FTy)
                    x, y = snakeReparam4e(x, y)
                x, y = snakeReparam4e(x, y)
                return x, y

            # Process
            xs = []
            ys = []
            for n_iter in NIter:
                x, y = script_for_fig1113(xi, yi, FTx, FTy, n_iter)
                xs.append(x)
                ys.append(y)

            # Display
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))

            axes[0, 0].imshow(f, cmap="gray")
            axes[0, 0].axis("off")
            plt.sca(axes[0, 0])
            snake_display(xi[::2], yi[::2], "g.")

            for idx, n_iter in enumerate(NIter):
                ax = axes.flat[idx + 1]
                ax.imshow(f, cmap="gray")
                ax.axis("off")
                plt.sca(ax)
                snake_display(xs[idx][::2], ys[idx][::2], "g.")

            plt.tight_layout()
            plt.savefig("Figure1113.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1114(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter11 script `Figure1114.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            # Parameters
            low = -0.5
            high = 0.5
            np_points = 40
            K = 10
            C = 0.6
            r = 0.2

            # Function phi (centered so that the bowl intersects near the plane center)
            x, y = np.meshgrid(
                np.linspace(low, high, np_points), np.linspace(low, high, np_points)
            )
            phi = K * (x**2 + y**2 - r**2)

            # Display
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(1, 1, 1, projection="3d")

            # Mesh-style surface (lines only, like MATLAB mesh)
            ax.plot_wireframe(
                x, y, phi, rstride=1, cstride=1, color="black", linewidth=0.3
            )

            # Plane
            plane = 0 * x + C
            ax.plot_surface(
                x,
                y,
                plane,
                rstride=1,
                cstride=1,
                facecolor=(0.5, 0.5, 0.5),
                edgecolor="none",
                linewidth=0.0,
                antialiased=False,
                shade=False,
                alpha=0.9,
            )

            # Intersection of phi and plane as a 3D curve at z = C:
            # K*(x^2 + y^2 - r^2) = C  ->  x^2 + y^2 = r^2 + C/K
            radius_sq = r**2 + C / K
            if radius_sq > 0:
                th = np.linspace(0, 2 * np.pi, 600)
                rr = np.sqrt(radius_sq)
                xc = rr * np.cos(th)
                yc = rr * np.sin(th)
                zc = np.full_like(th, C)
                ax.plot(xc, yc, zc, color="black", linewidth=2.0)

            ax.grid(False)
            ax.view_init(elev=8, azim=35)
            plt.savefig("Figure1114.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1115(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter11 script `Figure1115.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from skimage.measure import find_contours
            from helpers.libdip.levelSetFunction4e import levelSetFunction4e
            from helpers.libdip.levelSetForce4e import levelSetForce4e
            from helpers.libdip.levelSetIterate4e import levelSetIterate4e
            from helpers.libdipum.data_path import dip_data

            # Parameters
            r = 1
            times = [50, 100, 300, 700, 800]

            # Data
            img_path = dip_data("gray-lakes.tif")
            lakes = img_as_float(imread(img_path))
            M, N = lakes.shape

            # Initial level set function
            x0 = int(round(M / 2))
            y0 = int(round(N / 2))
            phi = [levelSetFunction4e("circular", M, N, x0, y0, r)]

            # Force field
            f = (lakes > 0.9).astype(float)
            F = levelSetForce4e("binary", [f, 1, 0])

            # Process
            contours = [None]  # 1-based indexing style
            K = len(times)
            for i in range(K):
                niter = times[i]
                phi_next = phi[0].copy()
                for _ in range(niter):
                    phi_next = levelSetIterate4e(phi_next, F)
                phi.append(phi_next)
                # contourc(phi, [0 0]) equivalent: contours at level 0
                contours.append(find_contours(phi_next, level=0))

            # Display
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))

            for idx in range(K + 1):
                ax = axes.flat[idx]
                ax.imshow(lakes, cmap="gray")
                ax.axis("off")
                # Plot contours (skip index 0 since it is placeholder)
                if idx > 0:
                    for c in contours[idx]:
                        # c is (row, col); plot x=col, y=row
                        ax.plot(c[:, 1], c[:, 0], "g.")

            plt.tight_layout()
            plt.savefig("Figure1115.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1118(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter11 script `Figure1118.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from skimage.measure import find_contours
            from helpers.libdip.levelSetFunction4e import levelSetFunction4e
            from helpers.libdip.levelSetForce4e import levelSetForce4e
            from helpers.libdip.levelSetIterate4e import levelSetIterate4e
            from helpers.libdipum.data_path import dip_data

            # Parameters
            x0 = [100, 100]
            y0 = [350, 350]
            r = [60, 150]
            a = [1, 1]
            b = [0, -1]
            iterations = [100, 400, 900]

            # Data
            img_path = dip_data("letterA-distorted.tif")
            f = img_as_float(imread(img_path))
            M, N = f.shape

            # Binarize
            fbin = (f < 0.6).astype(float)  # Letter should be white

            # Helpers
            def contourc_zero(phi: Any):
                """contourc_zero."""
                return find_contours(phi, level=0)

            # Initial level set function located outside character.
            phi0 = [None, None]
            phi0[0] = levelSetFunction4e("circular", M, N, x0[0], y0[0], r[0])

            c = [[], []]
            c[0].append(contourc_zero(phi0[0]))

            # Force definition for top row
            F = [None, None]
            F[0] = levelSetForce4e("binary", [fbin, a[0], b[0]])

            # Show stages of curve evolution
            for niter in iterations:
                phi = phi0[0].copy()
                for _ in range(niter):
                    phi = levelSetIterate4e(phi, F[0])
                c[0].append(contourc_zero(phi))

            # Initial level set function located inside character.
            phi0[1] = levelSetFunction4e("circular", M, N, x0[1], y0[1], r[1])

            c[1].append(contourc_zero(phi0[1]))

            # Force definition for bottom row
            F[1] = levelSetForce4e("binary", [fbin, a[1], b[1]])

            # Show stages of curve evolution
            for niter in iterations:
                phi = phi0[1].copy()
                for _ in range(niter):
                    phi = levelSetIterate4e(phi, F[1])
                c[1].append(contourc_zero(phi))

            # Display top row
            fig1, axes1 = plt.subplots(2, 2, figsize=(10, 8))
            for idx in range(len(iterations) + 1):
                ax = axes1.flat[idx]
                ax.imshow(f, cmap="gray")
                ax.axis("off")
                for cont in c[0][idx]:
                    # cont is (row, col)
                    ax.plot(cont[:, 1], cont[:, 0], "g.")

            plt.tight_layout()
            plt.savefig("Figure1118.png")

            # Display bottom row
            fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))
            for idx in range(len(iterations) + 1):
                ax = axes2.flat[idx]
                ax.imshow(f, cmap="gray")
                ax.axis("off")
                for cont in c[1][idx]:
                    ax.plot(cont[:, 1], cont[:, 0], "g.")

            plt.tight_layout()
            plt.savefig("Figure1118Bis.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1119(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter11 script `Figure1119.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from skimage.measure import find_contours
            from helpers.libdip.levelSetFunction4e import levelSetFunction4e
            from helpers.libdip.levelSetForce4e import levelSetForce4e
            from helpers.libdip.levelSetIterate4e import levelSetIterate4e
            from helpers.libdipum.data_path import dip_data

            # Parameters
            x0 = 100
            y0 = 350
            r = 150

            a = 1
            b = -1

            iterations = [50, 100, 120, 125]

            # Data
            img_path = dip_data("letterA-distorted.tif")
            f = img_as_float(imread(img_path))
            M, N = f.shape

            # Binarize
            fbin = (f < 0.6).astype(float)

            # Process
            phi0 = levelSetFunction4e("circular", M, N, x0, y0, r)
            contours_list = []

            F = levelSetForce4e("binary", [fbin, a, b])

            for niter in iterations:
                phi = phi0.copy()
                for _ in range(niter):
                    phi = levelSetIterate4e(phi, F)
                contours_list.append(find_contours(phi, level=0))

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            for idx in range(len(contours_list)):
                ax = axes.flat[idx]
                ax.imshow(f, cmap="gray")
                ax.axis("off")
                for cont in contours_list[idx]:
                    ax.plot(cont[:, 1], cont[:, 0], "r.")

            plt.tight_layout()
            plt.savefig("Figure1119.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure112(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter11 script `Figure112.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import os
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from scipy.io import loadmat, savemat
            from helpers.libdipum.snake_manual_input import snake_manual_input
            from helpers.libdip.snakeMap4e import snakeMap4e
            from helpers.libdip.snakeForce4e import snakeForce4e
            from helpers.libdip.snakeIterate4e import snakeIterate4e
            from helpers.libdip.snakeReparam4e import snakeReparam4e
            from helpers.libdipum.snake_display import snake_display
            from helpers.libdipum.data_path import dip_data

            # Parameters
            Alpha = 0.5
            Beta = 2.0
            Gamma = 5.0
            NPoints = 150
            T = 0.001
            Sig = 15
            NSig = 3
            Order = "both"
            NIter = 300

            # Data
            img_path = dip_data("noisy-elliptical-object.tif")
            g = imread(img_path)

            # Display image and get initial snake manually.
            if os.path.exists("Figure112.mat"):
                mat = loadmat("Figure112.mat")
                xi = mat["xi"].squeeze()
                yi = mat["yi"].squeeze()
            else:
                xi, yi = snake_manual_input(g, NPoints, "go")
                savemat("Figure112.mat", {"xi": xi, "yi": yi})

            # Close the snake
            xi = np.concatenate([xi, [xi[0]]])
            yi = np.concatenate([yi, [yi[0]]])

            # Edge map using filtering before and after edge map is computed.
            emap = snakeMap4e(g, T, Sig, NSig, Order)

            # Snake force using plain gradient.
            FTx, FTy = snakeForce4e(emap, "gradient")

            # Normalize the forces.
            mag = np.sqrt(FTx**2 + FTy**2)
            FTx = FTx / (mag + 1e-10)
            FTy = FTy / (mag + 1e-10)

            x = xi.copy()
            y = yi.copy()

            # Iterate
            for _ in range(NIter):
                x, y = snakeIterate4e(Alpha, Beta, Gamma, x, y, 1, FTx, FTy)
                x, y = snakeReparam4e(x, y)

            # Display figure 1
            fig1, axes1 = plt.subplots(2, 2, figsize=(10, 8))
            axes1[0, 0].imshow(g, cmap="gray")
            axes1[0, 0].set_title("Original image")
            axes1[0, 0].axis("off")

            axes1[0, 1].imshow(emap, cmap="gray")
            axes1[0, 1].set_title("edge map")
            axes1[0, 1].axis("off")

            axes1[1, 0].imshow(FTx, cmap="gray")
            axes1[1, 0].set_title("FT_x")
            axes1[1, 0].axis("off")

            axes1[1, 1].imshow(FTy, cmap="gray")
            axes1[1, 1].set_title("FT_y")
            axes1[1, 1].axis("off")

            plt.tight_layout()
            plt.savefig("Figure112.png")

            # Display figure 2
            fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
            r = 10
            ax2.quiver(np.flipud(FTy[::r, ::r]), np.flipud(-FTx[::r, ::r]))
            ax2.set_title("Vector snake force")
            plt.tight_layout()
            plt.savefig("Figure112Bis.png")

            # Display figure 3
            fig3, ax3 = plt.subplots(1, 1, figsize=(6, 6))
            ax3.imshow(g, cmap="gray")
            ax3.plot(xi, yi, "or")
            ax3.set_title("Initial contour")
            ax3.axis("off")
            plt.tight_layout()
            plt.savefig("Figure112Ter.png")

            # Display figure 4
            fig4, ax4 = plt.subplots(1, 1, figsize=(6, 6))
            ax4.imshow(g, cmap="gray")
            snake_display(x, y, "go")
            plt.tight_layout()
            plt.savefig("Figure112Quart.png")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1120(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter11 script `Figure1120.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from skimage.measure import find_contours
            from helpers.libdip.levelSetFunction4e import levelSetFunction4e
            from helpers.libdip.levelSetForce4e import levelSetForce4e
            from helpers.libdip.levelSetIterate4e import levelSetIterate4e
            from helpers.libdipum.data_path import dip_data

            # Parameters
            x0 = 250
            y0 = 225
            r = 120

            a = -1
            b = 1

            iterations = [30, 160, 170, 226, 250]

            # Data
            img_path = dip_data("multiple-regions.tif")
            f = img_as_float(imread(img_path))
            M, N = f.shape

            # Binarize
            fbin = (f > 0.7).astype(float)

            # Process
            phi0 = levelSetFunction4e("circular", M, N, x0, y0, r)
            contours_list = [find_contours(phi0, level=0)]

            F = levelSetForce4e("binary", [fbin, a, b])

            for niter in iterations:
                phi = phi0.copy()
                for _ in range(niter):
                    phi = levelSetIterate4e(phi, F)
                contours_list.append(find_contours(phi, level=0))

            # Display
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            for idx in range(len(contours_list)):
                ax = axes.flat[idx]
                ax.imshow(f, cmap="gray")
                ax.axis("off")
                for cont in contours_list[idx]:
                    ax.plot(cont[:, 1], cont[:, 0], "g.")

            plt.tight_layout()
            plt.savefig("Figure1120.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1121(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter11 script `Figure1121.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            from skimage.io import imread
            from skimage.util import img_as_float
            from skimage.measure import find_contours
            from helpers.libdip.levelSetFunction4e import levelSetFunction4e
            from helpers.libdip.levelSetForce4e import levelSetForce4e
            from helpers.libdip.levelSetIterate4e import levelSetIterate4e
            from helpers.libdipum.data_path import dip_data

            # Parameters
            x0 = 250
            y0 = 225
            r = 120

            a = -1
            b = 1

            iterations = [30, 250]

            # Data
            img_path = dip_data("multiple-regions.tif")
            f = img_as_float(imread(img_path))
            M, N = f.shape

            # Binarize
            fbin = (f > 0.7).astype(float)

            # Process
            phi0 = levelSetFunction4e("circular", M, N, x0, y0, r)
            contours_list = [find_contours(phi0, level=0)]

            F = levelSetForce4e("binary", [fbin, a, b])

            phi_list = []
            for niter in iterations:
                phi = phi0.copy()
                for _ in range(niter):
                    phi = levelSetIterate4e(phi, F)
                phi_list.append(phi)
                contours_list.append(find_contours(phi, level=0))

            # Display
            fig = plt.figure(figsize=(12, 8))

            # Top row: contours on image
            for i in range(3):
                ax = fig.add_subplot(2, 3, i + 1)
                ax.imshow(f, cmap="gray")
                ax.axis("off")
                for cont in contours_list[i]:
                    ax.plot(cont[:, 1], cont[:, 0], "g.")

            # Bottom row: surface plots
            for i in range(3):
                ax = fig.add_subplot(2, 3, i + 4, projection="3d")
                if i == 0:
                    phi = phi0
                else:
                    phi = phi_list[i - 1]
                phi_sub = phi[::8, ::8]
                X, Y = np.meshgrid(
                    np.arange(phi_sub.shape[1]), np.arange(phi_sub.shape[0])
                )
                ax.plot_surface(
                    X,
                    Y,
                    phi_sub,
                    facecolor=(0.9, 0.9, 0.9),
                    edgecolor="black",
                    linewidth=0.3,
                    shade=False,
                    alpha=1.0,
                )

                plane = np.zeros((round(N / 8), round(M / 8)))
                Xp, Yp = np.meshgrid(
                    np.arange(plane.shape[1]), np.arange(plane.shape[0])
                )
                ax.plot_surface(
                    Xp,
                    Yp,
                    plane,
                    facecolor=(0.5, 0.5, 0.5),
                    edgecolor="none",
                    alpha=1.0,
                )

                ax.view_init(elev=10, azim=232)
                ax.set_box_aspect((1, 1, 0.5))
                ax.set_axis_off()

            plt.tight_layout()
            plt.savefig("Figure1121.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1122(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter11 script `Figure1122.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            """Figure 11.22 - Level set segmentation of breast implant using Eq. (11-96)."""

            import os
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from skimage.measure import find_contours
            from scipy.ndimage import convolve

            from helpers.libdip.levelSetFunction4e import levelSetFunction4e
            from helpers.libdip.levelSetForce4e import levelSetForce4e
            from helpers.libdip.levelSetIterate4e import levelSetIterate4e
            from helpers.libdip.gaussKernel4e import gaussKernel4e
            from helpers.libdipum.data_path import dip_data

            print("Running Figure1122...")

            # Parameters
            n = 21
            sig = 5
            x0 = 370
            y0 = 350
            r = 18
            iterations = [50, 100, 200, 400]

            # Data
            img_path = dip_data("breast-implant.tif")
            f = img_as_float(imread(img_path))
            M, N = f.shape

            # Smooth image
            G = gaussKernel4e(n, sig)
            fsmooth = convolve(f, G, mode="nearest")  # replicate-like boundary

            # Initial level set and contour
            phi0 = levelSetFunction4e("circular", M, N, x0, y0, r)
            contours_list = [find_contours(phi0, level=0)]

            # Force field + thresholded binary force
            F = levelSetForce4e("gradient", [fsmooth, 1, 50])
            T = (float(np.min(F)) + float(np.max(F))) / 2.0
            FBin = F > T

            # Iterate
            phi = phi0.copy()
            for niter in iterations:
                phi = phi0.copy()
                for _ in range(niter):
                    phi = levelSetIterate4e(phi, FBin)
                contours_list.append(find_contours(phi, level=0))

            # Final mask from last iterate
            X = phi <= 0

            # Display Figure 1122
            fig1, axes = plt.subplots(2, 3, figsize=(12, 8))

            axes[0, 0].imshow(f, cmap="gray")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(F, cmap="gray")
            axes[0, 1].axis("off")

            for idx in range(len(iterations)):
                ax = axes.flat[idx + 2]
                ax.imshow(f, cmap="gray")
                ax.axis("off")
                for cont in contours_list[idx + 1]:
                    ax.plot(cont[:, 1], cont[:, 0], "w.")

            # Display Figure 1123
            fig2 = plt.figure(2)
            plt.imshow(X, cmap="gray")
            plt.axis("off")

            # Save
            out_dir = _os.environ.get("DIP4E_OUTPUT_DIR", str(_Path(__file__).resolve().parents[2] / "output"))
            out1 = os.path.join(out_dir, "Figure1122.png")
            out2 = os.path.join(out_dir, "Figure1123.png")
            fig1.savefig(out1, dpi=150, bbox_inches="tight")
            fig2.savefig(out2, dpi=150, bbox_inches="tight")
            print(f"Saved {out1}")
            print(f"Saved {out2}")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1124(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter11 script `Figure1124.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import os
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from skimage.measure import find_contours
            from scipy.io import loadmat, savemat
            from scipy.ndimage import convolve
            from helpers.libdipum.curve_manual_input import curve_manual_input
            from helpers.libdipum.coord2mask import coord2mask
            from helpers.libdip.levelSetFunction4e import levelSetFunction4e
            from helpers.libdip.levelSetForce4e import levelSetForce4e
            from helpers.libdip.levelSetIterate4e import levelSetIterate4e
            from helpers.libdip.levelSetReInit4e import levelSetReInit4e
            from helpers.libdip.gaussKernel4e import gaussKernel4e
            from helpers.libdipum.data_path import dip_data

            # Parameters
            n = 21
            sig = 5
            iterations = [200, 400, 600, 1000, 2000]

            # Data
            img_path = dip_data("breast-implant.tif")
            f = img_as_float(imread(img_path))
            M, N = f.shape

            # Input initial phi manually
            if os.path.exists("Figure1124.mat"):
                mat = loadmat("Figure1124.mat")
                x = mat["y"].squeeze()  # MKR
                y = mat["x"].squeeze()  # MKR
            else:
                x, y, vx, vy = curve_manual_input(f, 200, "g.")
                savemat("Figure1124.mat", {"x": x, "y": y})

            # Create mask for generating initial level set function
            binmask = coord2mask(M, N, x, y)

            # Create initial level set function
            phi0 = levelSetFunction4e("mask", binmask)

            # Obtain zero-level set contour
            contours_list = [find_contours(phi0, level=0)]

            # Smooth image
            G = gaussKernel4e(n, sig)
            fsmooth = convolve(f, G, mode="nearest")

            # Compute edge-marking function
            W = levelSetForce4e("gradient", [fsmooth, 1, 50])
            T = (np.max(W) + np.min(W)) / 2.0
            WBin = W > T

            # Iterate for specified iterations
            for niter in iterations:
                phi = phi0.copy()
                C = 0.5
                for i in range(1, niter + 1):
                    F = levelSetForce4e("geodesic", [phi, C, WBin])
                    phi = levelSetIterate4e(phi, F)
                    if i % 5 == 0:
                        phi = levelSetReInit4e(phi, 5, 0.5)
                contours_list.append(find_contours(phi, level=0))

            # Save final phi
            phi1500 = phi

            # Display
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            for idx in range(len(contours_list)):
                ax = axes.flat[idx]
                ax.imshow(f, cmap="gray")
                ax.axis("off")
                for cont in contours_list[idx]:
                    ax.plot(cont[:, 1], cont[:, 0], "y.")

            plt.tight_layout()
            plt.savefig("Figure1124.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1125(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter11 script `Figure1125.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from scipy.ndimage import convolve

            from helpers.libdip.levelSetForce4e import levelSetForce4e
            from helpers.libdip.gaussKernel4e import gaussKernel4e
            from helpers.libdipum.data_path import dip_data

            # Parameters
            n = 21
            sig = 5

            # Data
            img_path = dip_data("breast-implant.tif")
            f = img_as_float(imread(img_path))
            M, N = f.shape

            # Smooth image (fspecial('gaussian') + imfilter('replicate'))
            G = gaussKernel4e(n, sig)
            fsmooth = convolve(f, G, mode="nearest")

            # Compute edge-marking function
            W = levelSetForce4e("gradient", [fsmooth, 1, 50])
            T = (np.max(W) + np.min(W)) / 2.0
            WBin = W > T

            # Display
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(fsmooth, cmap="gray")
            axes[0].axis("off")
            axes[1].imshow(W, cmap="gray")
            axes[1].axis("off")
            axes[2].imshow(WBin, cmap="gray")
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig("Figure1125.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1126(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter11 script `Figure1126.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import os
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from scipy.ndimage import convolve
            import scipy.io as sio

            from helpers.libdip.levelSetFunction4e import levelSetFunction4e
            from helpers.libdip.levelSetForce4e import levelSetForce4e
            from helpers.libdip.levelSetIterate4e import levelSetIterate4e
            from helpers.libdip.gaussKernel4e import gaussKernel4e
            from helpers.libdipum.curve_manual_input import curve_manual_input
            from helpers.libdipum.coord2mask import coord2mask
            from helpers.libdipum.curve_display import curve_display
            from helpers.libdipum.contourc import contourc
            from helpers.libdipum.data_path import dip_data

            # Input initial phi manually.
            if os.path.exists("Figure1124.mat"):
                mat = sio.loadmat("Figure1124.mat")
                x = mat["y"].squeeze()
                y = mat["x"].squeeze()
            else:
                # Data is needed for manual input display
                img_path = dip_data("breast-implant.tif")
                f_tmp = img_as_float(imread(img_path))
                x, y, vx, vy = curve_manual_input(f_tmp, 200, "g.")
                sio.savemat("Figure1124.mat", {"x": x, "y": y, "vx": vx, "vy": vy})

            # Parameters
            n = 21
            sig = 5
            iterations = [1500]

            # Data
            img_path = dip_data("breast-implant.tif")
            f = img_as_float(imread(img_path))
            M, N = f.shape

            # Create mask for generating initial level set function.
            binmask = coord2mask(M, N, x, y)

            # Create initial level set function.
            phi0 = levelSetFunction4e("mask", binmask)

            # Obtain the zero-level set contour.
            c_list = [contourc(phi0, [0, 0])]

            # Smooth image.
            G = gaussKernel4e(n, sig)
            fsmooth = convolve(f, G, mode="nearest")

            # Compute edge-marking function.
            W = levelSetForce4e("gradient", [fsmooth, 1, 50])
            T = (np.max(W) + np.min(W)) / 2.0
            WBin = W > T

            # Iterate.
            for niter in iterations:
                phi = phi0.copy()
                C = 0.5
                for _ in range(niter):
                    F = levelSetForce4e("geodesic", [phi, C, WBin])
                    phi = levelSetIterate4e(phi, F)
                c_list.append(contourc(phi, [0, 0]))

            # Display
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            for idx, cc in enumerate(c_list):
                ax = axes[idx]
                ax.imshow(f, cmap="gray")
                ax.axis("off")
                curve_display(cc[1, :], cc[0, :], "y.", ax=ax)

            plt.tight_layout()
            plt.savefig("Figure1126.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1127(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter11 script `Figure1127.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import os
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from scipy.ndimage import convolve
            import scipy.io as sio

            from helpers.libdip.levelSetFunction4e import levelSetFunction4e
            from helpers.libdip.levelSetForce4e import levelSetForce4e
            from helpers.libdip.levelSetIterate4e import levelSetIterate4e
            from helpers.libdip.levelSetReInit4e import levelSetReInit4e
            from helpers.libdip.gaussKernel4e import gaussKernel4e
            from helpers.libdipum.curve_manual_input import curve_manual_input
            from helpers.libdipum.coord2mask import coord2mask
            from helpers.libdipum.data_path import dip_data

            # Parameters
            n = 21
            sig = 5
            NIter = 1500
            C = 0.5

            # Data
            img_path = dip_data("breast-implant.tif")
            f = img_as_float(imread(img_path))
            M, N = f.shape

            # Input initial phi manually.
            if os.path.exists("Figure1124.mat"):
                mat = sio.loadmat("Figure1124.mat")
                x = mat["x"].squeeze()
                y = mat["y"].squeeze()
            else:
                x, y, vx, vy = curve_manual_input(f, 200, "g.")
                sio.savemat("Figure1124.mat", {"x": x, "y": y, "vx": vx, "vy": vy})

            # Create mask for generating initial level set function.
            binmask = coord2mask(M, N, x, y)

            # Create initial level set function.
            phi0 = levelSetFunction4e("mask", binmask)

            # Smooth image.
            G = gaussKernel4e(n, sig)
            fsmooth = convolve(f, G, mode="nearest")

            # Compute edge-marking function.
            W = levelSetForce4e("gradient", [fsmooth, 1, 50])
            T = (np.max(W) + np.min(W)) / 2.0
            WBin = W > T

            # With reinitialization
            phi1 = phi0.copy()
            for i in range(1, NIter + 1):
                F = levelSetForce4e("geodesic", [phi1, C, WBin])
                phi1 = levelSetIterate4e(phi1, F)
                if i % 5 == 0:
                    phi1 = levelSetReInit4e(phi1, 5, 0.5)

            # Without reinitialization
            phi2 = phi0.copy()
            for _ in range(NIter):
                F = levelSetForce4e("geodesic", [phi2, C, WBin])
                phi2 = levelSetIterate4e(phi2, F)

            # Display
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(phi0, cmap="gray")
            axes[0].axis("off")
            axes[1].imshow(phi1, cmap="gray")
            axes[1].axis("off")
            axes[2].imshow(phi2, cmap="gray")
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig("Figure1127.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1128(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter11 script `Figure1128.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from helpers.libdip.levelSetForce4e import levelSetForce4e
            from helpers.libdipum.data_path import dip_data

            # Data
            img_path = dip_data("rose479by512.tif")
            f = img_as_float(imread(img_path))

            # Compute edge-marking function.
            W = levelSetForce4e("gradient", [f, 1, 50])

            # Display
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(f, cmap="gray")
            axes[0].axis("off")
            axes[1].imshow(W, cmap="gray")
            axes[1].axis("off")

            plt.tight_layout()
            plt.savefig("Figure1128.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1129(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter11 script `Figure1129.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float

            from helpers.libdip.levelSetFunction4e import levelSetFunction4e
            from helpers.libdip.levelSetForce4e import levelSetForce4e
            from helpers.libdip.levelSetIterate4e import levelSetIterate4e
            from helpers.libdip.levelSetReInit4e import levelSetReInit4e
            from helpers.libdipum.contourc import contourc
            from helpers.libdipum.curve_display import curve_display
            from helpers.libdipum.data_path import dip_data

            # Parameters
            iterations = [200, 300, 400, 600, 800]

            # Data
            img_path = dip_data("rose479by512.tif")
            f = img_as_float(imread(img_path))
            M, N = f.shape

            # Circular initial phi
            x0 = int(round(M / 2))
            y0 = int(round(N / 2))
            r = max(x0, y0) - max(x0, y0) / 5.0
            phi0 = levelSetFunction4e("circular", M, N, x0, y0, r)
            c_list = [contourc(phi0, [0, 0])]

            # Smooth image (none for this figure)
            fsmooth = f

            # Edge-marking function
            W = levelSetForce4e("gradient", [fsmooth, 1, 50])

            # Threshold W
            T = (np.max(W) + np.min(W)) / 2.0
            W = W > T

            # Process
            for niter in iterations:
                phi = phi0.copy()
                C = 0.5
                for i in range(1, niter + 1):
                    F = levelSetForce4e("geodesic", [phi, C, W])
                    phi = levelSetIterate4e(phi, F)
                    if i % 5 == 0:
                        phi = levelSetReInit4e(phi, 5, 0.5)
                c_list.append(contourc(phi, [0, 0]))

            # Display
            fig, axes = plt.subplots(2, 3, figsize=(10, 7))
            for idx in range(len(iterations) + 1):
                ax = axes.flat[idx]
                ax.imshow(f, cmap="gray")
                ax.axis("off")
                cc = c_list[idx]
                curve_display(cc[1, :], cc[0, :], "r.", ax=ax)

            plt.tight_layout()
            plt.savefig("Figure1129.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure113(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter11 script `Figure113.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from scipy.io import loadmat
            from helpers.libdip.snakeMap4e import snakeMap4e
            from helpers.libdip.snakeForce4e import snakeForce4e
            from helpers.libdip.snakeIterate4e import snakeIterate4e
            from helpers.libdip.snakeReparam4e import snakeReparam4e
            from helpers.libdipum.snake_display import snake_display
            from helpers.libdipum.data_path import dip_data

            # Parameters
            Alpha = 0.5
            Beta = 2.0
            Gamma = 5.0
            NPoints = 150
            T = 0.001
            Sig = 15
            NSig = 3
            Order = "both"
            NIter = 300

            # Data
            img_path = dip_data("noisy-elliptical-object.tif")
            g = imread(img_path)

            # to work with the same initial snake
            mat = loadmat("Figure112.mat")
            xi = mat["xi"].squeeze()
            yi = mat["yi"].squeeze()

            def process(
                g: Any,
                T: Any,
                Sig: Any,
                NSig: Any,
                NIter: Any,
                mode: Any,
                Alpha: Any,
                Beta: Any,
                Gamma: Any,
                xi: Any,
                yi: Any,
            ):
                """process."""
                emap = snakeMap4e(g, T, Sig, NSig, mode)

                # Snake force using plain gradient.
                FTx, FTy = snakeForce4e(emap, "gradient")

                # Normalize the forces.
                mag = np.sqrt(FTx**2 + FTy**2)
                FTx = FTx / (mag + 1e-10)
                FTy = FTy / (mag + 1e-10)

                x = xi.copy()
                y = yi.copy()

                # Iterate NIter times
                for _ in range(NIter):
                    x, y = snakeIterate4e(Alpha, Beta, Gamma, x, y, 1, FTx, FTy)
                    x, y = snakeReparam4e(x, y)

                return emap, x, y

            # (1) smooth, edge, smooth
            emap1, x1, y1 = process(
                g, T, Sig, NSig, NIter, "both", Alpha, Beta, Gamma, xi, yi
            )

            # (2) edge, smooth
            emap2, x2, y2 = process(
                g, T, Sig, NSig, NIter, "after", Alpha, Beta, Gamma, xi, yi
            )

            # (3) smooth, edge
            emap3, x3, y3 = process(
                g, T, Sig, NSig, NIter, "before", Alpha, Beta, Gamma, xi, yi
            )

            # (4) edge
            emap4, x4, y4 = process(
                g, T, Sig, NSig, NIter, "none", Alpha, Beta, Gamma, xi, yi
            )

            # Show results
            fig1, axes1 = plt.subplots(2, 2, figsize=(10, 8))
            axes1[0, 0].imshow(emap1, cmap="gray")
            axes1[0, 0].set_title("smooth, edge, smooth")
            axes1[0, 0].axis("off")

            axes1[1, 0].imshow(g, cmap="gray")
            axes1[1, 0].axis("off")
            plt.sca(axes1[1, 0])
            snake_display(x1, y1, "g.")

            axes1[0, 1].imshow(emap2, cmap="gray")
            axes1[0, 1].set_title("edge, smooth")
            axes1[0, 1].axis("off")

            axes1[1, 1].imshow(g, cmap="gray")
            axes1[1, 1].axis("off")
            plt.sca(axes1[1, 1])
            snake_display(x2, y2, "g.")

            plt.tight_layout()
            plt.savefig("Figure113.png")

            fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))
            axes2[0, 0].imshow(emap3, cmap="gray")
            axes2[0, 0].set_title("smooth, edge")
            axes2[0, 0].axis("off")

            axes2[1, 0].imshow(g, cmap="gray")
            axes2[1, 0].axis("off")
            plt.sca(axes2[1, 0])
            snake_display(x3, y3, "g.")

            axes2[0, 1].imshow(emap4, cmap="gray")
            axes2[0, 1].set_title("edge")
            axes2[0, 1].axis("off")

            axes2[1, 1].imshow(g, cmap="gray")
            axes2[1, 1].axis("off")
            plt.sca(axes2[1, 1])
            snake_display(x4, y4, "g.")

            plt.tight_layout()
            plt.savefig("Figure113Bis.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1131(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter11 script `Figure1131.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float

            from helpers.libdip.levelSetFunction4e import levelSetFunction4e
            from helpers.libdip.levelSetForce4e import levelSetForce4e
            from helpers.libdip.levelSetIterate4e import levelSetIterate4e
            from helpers.libdip.levelSetReInit4e import levelSetReInit4e
            from helpers.libdipum.contourc import contourc
            from helpers.libdipum.curve_display import curve_display
            from helpers.libdipum.data_path import dip_data

            # Parameters
            iterations = [500, 1000, 1500, 2000, 3500]
            mu = 2.0
            nu = 0.0
            lambda1 = 1
            lambda2 = 1

            # Data
            img_path = dip_data("cygnusloop.tif")
            f = img_as_float(imread(img_path))
            M, N = f.shape

            # Initial level set function.
            x0 = int(round(M / 2))
            y0 = int(round(N / 2))
            r = 110
            phi0 = levelSetFunction4e("circular", M, N, x0, y0, r)
            c_list = [contourc(phi0, [0, 0])]

            # Iterate
            for niter in iterations:
                phi = phi0.copy()
                C = 0.5
                for i in range(1, niter + 1):
                    F = levelSetForce4e(
                        "regioncurve", [f, phi, mu, nu, lambda1, lambda2], ["Fn", "Cn"]
                    )
                    phi = levelSetIterate4e(phi, F, 0.5)
                    if i % 5 == 0:
                        phi = levelSetReInit4e(phi, 5, 0.5)
                c_list.append(contourc(phi, [0, 0]))

            # Display contours
            fig, axes = plt.subplots(2, 3, figsize=(10, 7))
            for idx in range(len(iterations) + 1):
                ax = axes.flat[idx]
                ax.imshow(f, cmap="gray")
                ax.axis("off")
                cc = c_list[idx]
                if cc.shape[1] > 0:
                    curve_display(cc[1, 0::4], cc[0, 0::4], "r.", ax=ax)

            plt.tight_layout()
            plt.savefig("Figure1131.png")
            plt.show()

            # Binary mask display
            plt.figure()
            plt.imshow(phi <= 0, cmap="gray")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig("Figure1131Bis.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1132(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter11 script `Figure1132.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import os
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from skimage.measure import find_contours
            import scipy.io as sio

            from helpers.libdip.levelSetFunction4e import levelSetFunction4e
            from helpers.libdip.levelSetForce4e import levelSetForce4e
            from helpers.libdip.levelSetIterate4e import levelSetIterate4e
            from helpers.libdip.levelSetReInit4e import levelSetReInit4e
            from helpers.libdipum.curve_manual_input import curve_manual_input
            from helpers.libdipum.coord2mask import coord2mask
            from helpers.libdipum.curve_display import curve_display
            from helpers.libdipum.data_path import dip_data

            # Parameters
            mu = 0.5
            nu = 0
            lambda1 = 1
            lambda2 = 1
            iterations = [500, 1000, 1500]

            # Data
            img_path = dip_data("noisy-blobs.tif")
            f = img_as_float(imread(img_path))
            M, N = f.shape

            # Input initial phi manually.
            if os.path.exists("Figure1132.mat"):
                mat = sio.loadmat("Figure1132.mat")
                x = mat["y"].squeeze()
                y = mat["x"].squeeze()
            else:
                x, y, vx, vy = curve_manual_input(f, 200, "g.")
                sio.savemat("Figure1132.mat", {"x": x, "y": y, "vx": vx, "vy": vy})

            binmask = coord2mask(M, N, x, y)

            # Create and display initial level set function.
            phi_list = [levelSetFunction4e("mask", binmask)]
            c_list = [find_contours(phi_list[0], level=0.0)]

            for niter in iterations:
                phi = phi_list[0].copy()
                C = 0.5
                for i in range(1, niter + 1):
                    F = levelSetForce4e(
                        "regioncurve", [f, phi, mu, nu, lambda1, lambda2], ["Fn", "Cn"]
                    )
                    phi = levelSetIterate4e(phi, F)
                    if i % 5 == 0:
                        phi = levelSetReInit4e(phi, 5, 0.5)
                phi_list.append(phi)
                c_list.append(find_contours(phi, level=0.0))

            # Display
            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            for idx in range(len(iterations) + 1):
                ax = axes[0, idx]
                ax.imshow(f, cmap="gray")
                ax.axis("off")
                for cc in c_list[idx]:
                    # find_contours gives (row, col). curve_display(x,y) does plot(y,x),
                    # so pass x=row, y=col to display as plot(col,row).
                    curve_display(cc[:, 0], cc[:, 1], "g.", ax=ax)

                ax2 = axes[1, idx]
                ax2.imshow(phi_list[idx] <= 0, cmap="gray")
                ax2.axis("off")

            plt.tight_layout()
            plt.savefig("Figure1132.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1133(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter11 script `Figure1133.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import os
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            from skimage.measure import find_contours
            import scipy.io as sio

            from helpers.libdip.levelSetFunction4e import levelSetFunction4e
            from helpers.libdip.levelSetForce4e import levelSetForce4e
            from helpers.libdip.levelSetIterate4e import levelSetIterate4e
            from helpers.libdip.levelSetReInit4e import levelSetReInit4e
            from helpers.libdipum.curve_manual_input import curve_manual_input
            from helpers.libdipum.coord2mask import coord2mask
            from helpers.libdipum.curve_display import curve_display
            from helpers.libdipum.data_path import dip_data

            # Data
            img_path = dip_data("noisy-blobs.tif")
            f = img_as_float(imread(img_path))
            M, N = f.shape

            # Input initial phi manually.
            if os.path.exists("Figure1132.mat"):
                mat = sio.loadmat("Figure1132.mat")
                x = mat["y"].squeeze()
                y = mat["x"].squeeze()
            else:
                x, y, vx, vy = curve_manual_input(f, 200, "g.")
                sio.savemat("Figure1132.mat", {"x": x, "y": y, "vx": vx, "vy": vy})

            binmask = coord2mask(M, N, x, y)

            # Parameters
            mu_list = [0.5, 0.75, 3]
            nu = 0
            lambda1 = 1
            lambda2 = 1
            NIter = 1500

            # Initial level set function.
            phi0 = levelSetFunction4e("mask", binmask)

            phi_list = []
            c_list = []
            for mu in mu_list:
                phi = phi0.copy()
                C = 0.5
                for i in range(1, NIter + 1):
                    F = levelSetForce4e(
                        "regioncurve", [f, phi, mu, nu, lambda1, lambda2], ["Fn", "Cn"]
                    )
                    phi = levelSetIterate4e(phi, F)
                    if i % 5 == 0:
                        phi = levelSetReInit4e(phi, 5, 0.5)
                phi_list.append(phi)
                c_list.append(find_contours(phi, level=0.0))

            # Display
            fig, axes = plt.subplots(2, 3, figsize=(12, 6))
            for idx in range(len(mu_list)):
                ax = axes[0, idx]
                ax.imshow(f, cmap="gray")
                ax.axis("off")
                for cc in c_list[idx]:
                    # find_contours returns (row, col); curve_display(x,y) plots plot(y,x).
                    curve_display(cc[:, 0], cc[:, 1], "g.", ax=ax)

                ax2 = axes[1, idx]
                ax2.imshow(phi_list[idx] <= 0, cmap="gray")
                ax2.axis("off")

            plt.tight_layout()
            plt.savefig("Figure1133.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1134(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter11 script `Figure1134.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float

            from helpers.libdip.levelSetForce4e import levelSetForce4e
            from helpers.libdip.levelSetIterate4e import levelSetIterate4e
            from helpers.libdip.levelSetReInit4e import levelSetReInit4e
            from helpers.libdipum.contourc import contourc
            from helpers.libdipum.curve_display import curve_display
            from helpers.libdipum.data_path import dip_data

            # Parameters
            mu = 0.5
            nu = 0
            lambda1 = 1
            lambda2 = 1
            iterations = [100, 300, 500, 700, 1000]

            # Data
            img_path = dip_data("rose479by512.tif")
            f = img_as_float(imread(img_path))
            M, N = f.shape

            # Circular initial phi
            y, x = np.meshgrid(np.arange(1, N + 1), np.arange(1, M + 1))
            center = (int(round(M / 2)), int(round(N / 2)))
            r = max(center) - max(center) / 3.0
            phi_list = [np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) - r]

            # Initial contour
            c_list = [contourc(phi_list[0], [0, 0])]

            for niter in iterations:
                phi = phi_list[0].copy()
                C = 0.5
                for i in range(1, niter + 1):
                    F = levelSetForce4e(
                        "regioncurve", [f, phi, mu, nu, lambda1, lambda2], ["Fn", "Cn"]
                    )
                    phi = levelSetIterate4e(phi, F)
                    if i % 5 == 0:
                        phi = levelSetReInit4e(phi, 5, 0.5)
                phi_list.append(phi)
                c_list.append(contourc(phi, [0, 0]))

            # Display
            fig, axes = plt.subplots(2, 3, figsize=(10, 7))
            for idx in range(len(iterations) + 1):
                ax = axes.flat[idx]
                ax.imshow(f, cmap="gray")
                ax.axis("off")
                cc = c_list[idx]
                curve_display(cc[1, :], cc[0, :], "g.", ax=ax)

            plt.tight_layout()
            plt.savefig("Figure1134.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1135(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter11 script `Figure1135.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            import scipy.io as sio

            from helpers.libdip.intScaling4e import intScaling4e
            from helpers.libdip.snakeMap4e import snakeMap4e
            from helpers.libdip.snakeForce4e import snakeForce4e
            from helpers.libdip.snakeIterate4e import snakeIterate4e
            from helpers.libdip.snakeReparam4e import snakeReparam4e

            from helpers.libdip.levelSetForce4e import levelSetForce4e
            from helpers.libdip.levelSetIterate4e import levelSetIterate4e
            from helpers.libdip.levelSetReInit4e import levelSetReInit4e

            from helpers.libdipum.contourc import contourc
            from helpers.libdipum.curve_display import curve_display
            from helpers.libdipum.snake_display import snake_display
            from helpers.libdipum.data_path import dip_data

            # (1) Snake part (gvf)
            mat_path = dip_data("Figure118.mat")
            mat = sio.loadmat(mat_path)
            if "f" not in mat:
                # Fallback for local workspace files that only store xi/yi.
                mat = sio.loadmat("processing/Chapter11/Figure118.mat")
            f = mat["f"]
            xi = mat["yi"].squeeze()
            yi = mat["xi"].squeeze()
            M, N = f.shape

            T = 0.005
            Sig = 11
            NSig = 5
            Alpha = 10 * 0.05
            Beta = 0.5
            Gamma = 5

            emap = snakeMap4e(f, T, Sig, NSig, "both")
            emap = img_as_float(intScaling4e(emap))

            FTxb, FTyb = snakeForce4e(emap, "gvf", 0.2, 160)
            magb = np.sqrt(FTxb**2 + FTyb**2)
            FTxb = FTxb / (magb + 1e-10)
            FTyb = FTyb / (magb + 1e-10)

            x = xi.copy()
            y = yi.copy()
            for _ in range(400):
                x, y = snakeIterate4e(Alpha, Beta, Gamma, x, y, 1, FTxb, FTyb)
                x, y = snakeReparam4e(x, y)
            x, y = snakeReparam4e(x, y)

            # (2) Level Set (Edge based)
            f1 = img_as_float(imread(dip_data("rose479by512.tif")))
            M1, N1 = f1.shape
            y1, x1 = np.meshgrid(np.arange(1, N1 + 1), np.arange(1, M1 + 1))
            center = (int(round(M1 / 2)), int(round(N1 / 2)))
            r = max(center) - max(center) / 3.0
            phi0 = np.sqrt((x1 - center[0]) ** 2 + (y1 - center[1]) ** 2) - r

            W = levelSetForce4e("gradient", [f1, 1, 50])
            T = (np.max(W) + np.min(W)) / 2.0
            W = W > T

            phi = phi0.copy()
            C = 0.5
            for i in range(1, 800 + 1):
                F = levelSetForce4e("geodesic", [phi, C, W])
                phi = levelSetIterate4e(phi, F)
                if i % 5 == 0:
                    phi = levelSetReInit4e(phi, 5, 0.5)
            c = contourc(phi, [0, 0])

            # (3) Level Set (Region based) Chan-Vese
            mu = 0.5
            nu = 0
            lambda1 = 1
            lambda2 = 1
            phi = phi0.copy()
            for i in range(1, 800 + 1):
                F = levelSetForce4e(
                    "regioncurve", [f1, phi, mu, nu, lambda1, lambda2], ["Fn", "Cn"]
                )
                phi = levelSetIterate4e(phi, F)
                if i % 5 == 0:
                    phi = levelSetReInit4e(phi, 5, 0.5)
            c1 = contourc(phi, [0, 0])

            # Display
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(f, cmap="gray")
            axes[0].axis("off")
            plt.sca(axes[0])
            snake_display(x, y, "g.")
            axes[0].set_title("snake")

            axes[1].imshow(f1, cmap="gray")
            axes[1].axis("off")
            curve_display(c[1, :], c[0, :], "g.", ax=axes[1])
            axes[1].set_title("levelset geodesic")

            axes[2].imshow(f1, cmap="gray")
            axes[2].axis("off")
            curve_display(c1[1, :], c1[0, :], "g.", ax=axes[2])
            axes[2].set_title("levelset region based")

            plt.tight_layout()
            plt.savefig("Figure1135.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1136(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter11 script `Figure1136.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import os
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy.ndimage import convolve
            from skimage.io import imread
            from skimage.util import img_as_float
            import scipy.io as sio

            from helpers.libdip.intScaling4e import intScaling4e
            from helpers.libdip.snakeMap4e import snakeMap4e
            from helpers.libdip.snakeForce4e import snakeForce4e
            from helpers.libdip.snakeIterate4e import snakeIterate4e
            from helpers.libdip.snakeReparam4e import snakeReparam4e
            from helpers.libdip.levelSetFunction4e import levelSetFunction4e
            from helpers.libdip.levelSetForce4e import levelSetForce4e
            from helpers.libdip.levelSetIterate4e import levelSetIterate4e
            from helpers.libdip.levelSetReInit4e import levelSetReInit4e

            from helpers.libdip.gaussKernel4e import gaussKernel4e
            from helpers.libdipum.coord2mask import coord2mask
            from helpers.libdipum.contourc import contourc
            from helpers.libdipum.curve_display import curve_display
            from helpers.libdipum.curve_manual_input import curve_manual_input
            from helpers.libdipum.snake_display import snake_display
            from helpers.libdipum.data_path import dip_data

            # Data
            img_path = dip_data("breast-implant.tif")
            f = img_as_float(imread(img_path))
            if f.ndim == 3:
                f = f[..., 0]
            M, N = f.shape

            # (1) Snake part (gvf)
            T = 0.005
            Sig = 11
            NSig = 5
            Alpha = 10 * 0.05
            Beta = 0.5
            Gamma = 5

            t = np.arange(0, 2 * np.pi + 1e-12, 0.05)
            xi = 300 + 70 * np.cos(t)
            yi = 360 + 90 * np.sin(t)

            # Close the snake
            xi = np.append(xi, xi[0])
            yi = np.append(yi, yi[0])

            # Edge map
            emap = snakeMap4e(f, T, Sig, NSig, "both")
            emap = img_as_float(intScaling4e(emap))

            # Snake force
            FTx, FTy = snakeForce4e(emap, "gvf", 0.2, 160)
            mag = np.sqrt(FTx**2 + FTy**2)
            FTx = FTx / (mag + 1e-10)
            FTy = FTy / (mag + 1e-10)

            # Process
            x = xi.copy()
            y = yi.copy()
            for _ in range(80):
                x, y = snakeIterate4e(Alpha, Beta, Gamma, x, y, 1, FTx, FTy)
                x, y = snakeReparam4e(x, y)
            xSnake, ySnake = snakeReparam4e(x, y)

            # (2) Level Set (Edge based)
            n = 21
            sig = 5
            NIter = 1500

            if os.path.exists("Figure1124.mat"):
                mat = sio.loadmat("Figure1124.mat")
                # Keep same convention as Figure1124.py in this project
                x_ls = np.asarray(mat["y"]).squeeze()
                y_ls = np.asarray(mat["x"]).squeeze()
            else:
                x_ls, y_ls, vx, vy = curve_manual_input(f, 200, "g.")
                sio.savemat(
                    "Figure1124.mat", {"x": x_ls, "y": y_ls, "vx": vx, "vy": vy}
                )

            binmask = coord2mask(M, N, x_ls, y_ls)
            phi0 = levelSetFunction4e("mask", binmask)

            G = gaussKernel4e(n, sig)
            fsmooth = convolve(f, G, mode="nearest")

            W = levelSetForce4e("gradient", [fsmooth, 1, 50])
            T = (np.max(W) + np.min(W)) / 2.0
            WBin = W > T

            phi = phi0.copy()
            C = 0.5
            for I in range(1, NIter + 1):
                F = levelSetForce4e("geodesic", [phi, C, WBin])
                phi = levelSetIterate4e(phi, F)
                if I % 5 == 0:
                    phi = levelSetReInit4e(phi, 5, 0.5)
            cLevelSetEdge = contourc(phi, [0, 0])

            # (3) Level Set (Region based) Chan-Vese
            mu = 0.5
            nu = 0
            lambda1 = 1
            lambda2 = 1

            y1, x1 = np.meshgrid(np.arange(1, N + 1), np.arange(1, M + 1))
            center = [330, 350]
            r = 30
            phi0 = np.sqrt((x1 - center[0]) ** 2 + (y1 - center[1]) ** 2) - r

            phi = phi0.copy()
            for I in range(1, 800 + 1):
                F = levelSetForce4e(
                    "regioncurve", [f, phi, mu, nu, lambda1, lambda2], ["Fn", "Cn"]
                )
                phi = levelSetIterate4e(phi, F)
                if I % 5 == 0:
                    phi = levelSetReInit4e(phi, 5, 0.5)
            cLevelSetRegion = contourc(phi, [0, 0])

            # Display
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(f, cmap="gray")
            plt.axis("off")
            snake_display(xSnake, ySnake, "g.")
            plt.title("snake")

            plt.subplot(1, 3, 2)
            plt.imshow(f, cmap="gray")
            plt.axis("off")
            plt.title("Level Set edge")
            curve_display(cLevelSetEdge[1, :], cLevelSetEdge[0, :], "g.")

            plt.subplot(1, 3, 3)
            plt.imshow(f, cmap="gray")
            plt.axis("off")
            curve_display(cLevelSetRegion[1, :], cLevelSetRegion[0, :], "g.")

            plt.tight_layout()
            plt.savefig("Figure1136.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1137(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter11 script `Figure1137.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            import scipy.io as sio

            from helpers.libdipum.SnakeSegmentation import SnakeSegmentation
            from helpers.libdipum.LevelSetEdgebased import LevelSetEdgebased
            from helpers.libdipum.LevelSetRegionBased import LevelSetRegionBased
            from helpers.libdipum.coord2mask import coord2mask
            from helpers.libdipum.snake_display import snake_display
            from helpers.libdipum.curve_display import curve_display
            from helpers.libdipum.data_path import dip_data

            # %% Data
            img_path = dip_data("noisy-blobs.tif")
            f = img_as_float(imread(img_path))
            if f.ndim == 3:
                f = f[..., 0]
            M, N = f.shape

            # %% Initial boundary
            mat = sio.loadmat("Figure1132.mat")
            # Keep the same Figure1132.mat convention used in Chapter 11 scripts:
            # stored x/y are swapped relative to the snake/level-set internal row/col convention.
            x = np.asarray(mat["y"]).squeeze()
            y = np.asarray(mat["x"]).squeeze()

            # Binary mask
            binmask = coord2mask(M, N, x, y)

            # %% Parameters
            # Snake
            Snake = {
                "Mu": 0.2,
                "NIterConvergence": 180,
                "NIterForce": 160,
                "T": 0.005,
                "Alpha": 0.10,
                "Beta": 1.00,
                "Gamma": 0.20,
                "Sig": 11,
                "NSig": 5,
            }

            # Level set (edge based)
            LSEdgeBased = {
                "HSize": 11,
                "Sigma": 3,
                "p": 1,
                "lambda": 50,
                "niter": 2000,
            }

            # Level set (region based)
            LSRegionBased = {
                "mu": 0.5,
                "nu": 0,
                "lambda1": 1,
                "lambda2": 1,
                "niter": 2000,
            }

            # %% 1) Snake
            x_snake, y_snake, emap_snake = SnakeSegmentation(
                f,
                x,
                y,
                Snake["T"],
                Snake["Sig"],
                Snake["NSig"],
                Snake["Mu"],
                Snake["NIterForce"],
                Snake["NIterConvergence"],
                Snake["Alpha"],
                Snake["Beta"],
                Snake["Gamma"],
            )

            # %% 2) Level set edge based
            c0, fsmooth0, WBin0 = LevelSetEdgebased(
                f,
                binmask,
                LSEdgeBased["HSize"],
                LSEdgeBased["Sigma"],
                LSEdgeBased["p"],
                LSEdgeBased["lambda"],
                LSEdgeBased["niter"],
            )

            # %% 3) Level set region based
            c = LevelSetRegionBased(
                f,
                binmask,
                LSRegionBased["mu"],
                LSRegionBased["nu"],
                LSRegionBased["lambda1"],
                LSRegionBased["lambda2"],
                LSRegionBased["niter"],
            )

            # %% Display
            plt.figure(figsize=(13, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(f, cmap="gray")
            plt.axis("off")
            snake_display(x_snake[0::2], y_snake[0::2], "yo")
            plt.title("Snake enclosing the 3 blobs")

            plt.subplot(1, 3, 2)
            plt.imshow(f, cmap="gray")
            plt.axis("off")
            plt.title("Edge based level set")
            curve_display(c0[1, :], c0[0, :], "y.")

            plt.subplot(1, 3, 3)
            plt.imshow(f, cmap="gray")
            plt.axis("off")
            plt.title("region based level set")
            curve_display(c[1, :], c[0, :], "y.")

            plt.tight_layout()
            plt.savefig("Figure1137.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1138(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter11 script `Figure1138.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            import scipy.io as sio

            from helpers.libdipum.SnakeSegmentation import SnakeSegmentation
            from helpers.libdipum.LevelSetEdgebased import LevelSetEdgebased
            from helpers.libdipum.LevelSetRegionBased import LevelSetRegionBased
            from helpers.libdipum.snake_display import snake_display
            from helpers.libdipum.curve_display import curve_display
            from helpers.libdipum.data_path import dip_data

            # %% Parameters
            # Snake
            Snake = {
                "T": 0.01,
                "Sig": 11,
                "NSig": 5,
                "Mu": 0.2,
                "NIterForce": 160,
                "NIterConvergence": 35,
                "Alpha": 0.05,
                "Beta": 0.5,
                "Gamma": 2.5,
            }

            # Level set (edge based)
            LSEdgeBased = {
                "HSize": 21,
                "Sigma": 5,
                "p": 1,
                "lambda": 50,
                "niter": 500,
            }

            # Level set (region based)
            LSRegionBased = {
                "mu": 2.0,
                "nu": 0.0,
                "lambda1": 1,
                "lambda2": 1,
                "niter": 3500,
            }

            # %% Data
            img_path = dip_data("cygnusloop.tif")
            f = img_as_float(imread(img_path))
            if f.ndim == 3:
                f = f[..., 0]
            M, N = f.shape
            _ = (M, N)

            # %% Initial contour / mask
            mat_snake = sio.loadmat(dip_data("WorkspaceFig1138(a).mat"))
            xi = np.asarray(mat_snake["xi"]).squeeze()
            yi = np.asarray(mat_snake["yi"]).squeeze()

            mat_mask = sio.loadmat(dip_data("WorkspaceForFig1138(b).mat"))
            mask = np.asarray(mat_mask["mask"])

            # %% 1) Snake
            x, y, emap = SnakeSegmentation(
                f,
                xi,
                yi,
                Snake["T"],
                Snake["Sig"],
                Snake["NSig"],
                Snake["Mu"],
                Snake["NIterForce"],
                Snake["NIterConvergence"],
                Snake["Alpha"],
                Snake["Beta"],
                Snake["Gamma"],
            )

            # %% 2) Level set edge based
            c0, fsmooth0, WBin0 = LevelSetEdgebased(
                f,
                mask,
                LSEdgeBased["HSize"],
                LSEdgeBased["Sigma"],
                LSEdgeBased["p"],
                LSEdgeBased["lambda"],
                LSEdgeBased["niter"],
            )

            # %% 3) Level set region based
            c = LevelSetRegionBased(
                f,
                mask,
                LSRegionBased["mu"],
                LSRegionBased["nu"],
                LSRegionBased["lambda1"],
                LSRegionBased["lambda2"],
                LSRegionBased["niter"],
            )

            # %% Display
            plt.figure(figsize=(13, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(f, cmap="gray")
            plt.axis("off")
            snake_display(xi[0::2], yi[0::2], "r.")
            snake_display(x[0::2], y[0::2], "y.")
            plt.title("Snake enclosing the 3 blobs")

            plt.subplot(1, 3, 2)
            plt.imshow(f, cmap="gray")
            plt.axis("off")
            plt.title("Edge based level set")
            curve_display(c0[1, :], c0[0, :], "y.")

            plt.subplot(1, 3, 3)
            plt.imshow(f, cmap="gray")
            plt.axis("off")
            plt.title("region based level set")
            curve_display(c[1, :], c[0, :], "y.")

            plt.tight_layout()
            plt.savefig("Figure1138.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1139(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter11 script `Figure1139.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.util import img_as_float
            import scipy.io as sio

            from helpers.libdipum.SnakeSegmentation import SnakeSegmentation
            from helpers.libdipum.LevelSetEdgebased import LevelSetEdgebased
            from helpers.libdipum.data_path import dip_data

            # %% Parameters
            # Snake
            Snake = {
                "T": 0.01,
                "Sig": 11,
                "NSig": 5,
                "Mu": 0.2,
                "NIterForce": 160,
                "NIterConvergence": 35,
                "Alpha": 0.05,
                "Beta": 0.5,
                "Gamma": 2.5,
            }

            # Level set (edge based)
            LSEdgeBased = {
                "HSize": 21,
                "Sigma": 5,
                "p": 1,
                "lambda": 50,
                "niter": 500,
            }

            # %% Data
            img_path = dip_data("cygnusloop.tif")
            f = img_as_float(imread(img_path))
            if f.ndim == 3:
                f = f[..., 0]
            M, N = f.shape
            _ = (M, N)

            # %% Initial contour
            mat_snake = sio.loadmat(dip_data("WorkspaceFig1138(a).mat"))
            xi = mat_snake["xi"].squeeze()
            yi = mat_snake["yi"].squeeze()

            mat_mask = sio.loadmat(dip_data("WorkspaceForFig1138(b).mat"))
            mask = mat_mask["mask"]

            # %% 1) Snake
            x, y, emap = SnakeSegmentation(
                f,
                xi,
                yi,
                Snake["T"],
                Snake["Sig"],
                Snake["NSig"],
                Snake["Mu"],
                Snake["NIterForce"],
                Snake["NIterConvergence"],
                Snake["Alpha"],
                Snake["Beta"],
                Snake["Gamma"],
            )
            _ = (x, y)

            # %% 2) Level set edge based
            c0, fsmooth, WBin = LevelSetEdgebased(
                f,
                mask,
                LSEdgeBased["HSize"],
                LSEdgeBased["Sigma"],
                LSEdgeBased["p"],
                LSEdgeBased["lambda"],
                LSEdgeBased["niter"],
            )
            _ = (c0, fsmooth)

            # %% Display
            plt.figure(figsize=(9, 4))

            plt.subplot(1, 2, 1)
            plt.imshow(emap, cmap="gray")
            plt.axis("off")
            plt.title("Snake edge map")

            plt.subplot(1, 2, 2)
            plt.imshow(WBin, cmap="gray")
            plt.axis("off")
            plt.title("Edge based level set")

            plt.tight_layout()
            plt.savefig("Figure1139.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure113_skimage(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter11 script `Figure113_skimage.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            import scipy.io
            from skimage.segmentation import active_contour
            import os

            from helpers.libdip.snakeMap4e import snakeMap4e
            from helpers.libdipum.data_path import dip_data

            def Figure113_skimage():
                """
                Re-implementation of Figure 11.3 using skimage.segmentation.active_contour.
                Uses snakeMap4e for standard DIP4e edge map generation logic, but replaces the
                iterative solver with skimage's robust implementation.
                """

                plt.close("all")

                # 1. Load Data
                mat_path = dip_data("Figure112.mat")
                if not os.path.exists(mat_path):
                    print(f"Error: {mat_path} not found.")
                    return

                print(f"Loading {mat_path}...")
                mat_data = scipy.io.loadmat(mat_path)

                def get_var(name: Any):
                    """get_var."""
                    if name not in mat_data:
                        return None
                    val = mat_data[name]
                    return val.item() if val.size == 1 else np.squeeze(val)

                g = get_var("g")  # Input image
                T = get_var("T")  # Threshold
                Sig = get_var("Sig")  # Sigma for Gaussian
                NSig = get_var("NSig")  # Size of kernel
                # NIter = get_var('NIter') # Number of iterations (approx 230 in mat file, but we might want more for convergence)
                Alpha = get_var("Alpha")  # Continuity (membrane)
                Beta = get_var("Beta")  # Curvature (thin plate)
                Gamma = get_var("Gamma")  # Step size (Viscosity)

                xi = get_var("xi")  # Initial X (columns)
                yi = get_var("yi")  # Initial Y (rows)
                # Ensure xi, yi are correct shape (N,)
                if xi.ndim > 1:
                    xi = xi.flatten()
                if yi.ndim > 1:
                    yi = yi.flatten()

                # Construct initial snake (N, 2) -> (row, col) = (y, x)
                init_snake = np.stack([yi, xi], axis=1)

                # Helper to run snake on a pre-computed feature map (energy map)
                def run_snake_on_map(emap: Any, label: Any = ""):
                    """run_snake_on_map."""
                    print(f"Running snake for {label}...")
                    try:
                        # We treat the emap as the energy surface.
                        # w_line=1: Attract to bright regions (high values in emap).
                        # w_edge=0: Do NOT compute gradients of emap, use emap directly.
                        # This allows us to feed ANY processed map (e.g. smoothed binary edges) to the snake.
                        snake = active_contour(
                            emap,
                            init_snake,
                            alpha=Alpha,
                            beta=Beta,
                            w_line=1,
                            w_edge=0,
                            gamma=0.01,
                            max_iterations=2500,
                            boundary_condition="periodic",
                        )
                    except Exception as e:
                        print(f"Snake failed for {label}: {e}")
                        snake = init_snake
                    return snake, emap

                # The Logic matches Figure113.m / snakeMap4e options:

                # Case 1: "Both"
                # snakeMap4e(..., 'both') -> Filter Image -> Edge/Thresh -> Filter EdgeMap
                # This produces a smoothed version of the binary edges.
                print("Generating Map: Both...")
                emap1 = snakeMap4e(g, T, Sig, NSig, "both")
                snake1, _ = run_snake_on_map(emap1, label="Both")

                # Case 2: "After"
                # snakeMap4e(..., 'after') -> Edge/Thresh -> Filter EdgeMap
                print("Generating Map: After...")
                emap2 = snakeMap4e(g, T, Sig, NSig, "after")
                snake2, _ = run_snake_on_map(emap2, label="After")

                # Case 3: "Before"
                # snakeMap4e(..., 'before') -> Filter Image -> Edge/Thresh (No Post-Filter)
                # Result is Binary Edges of Smoothed Image.
                # Note: If T='auto', it's binary. Active contour on binary image might be jagged.
                # But this is what the book/MATLAB does.
                print("Generating Map: Before...")
                emap3 = snakeMap4e(g, T, Sig, NSig, "before")
                snake3, _ = run_snake_on_map(emap3, label="Before")

                # Case 4: "None"
                # snakeMap4e(..., 'none') -> Edge/Thresh (No Filter)
                # Binary Edges of Raw Image. (Very Noisy)
                print("Generating Map: None...")
                emap4 = snakeMap4e(g, T, Sig, NSig, "none")
                snake4, _ = run_snake_on_map(emap4, label="None")

                # 3. Visualization
                fig, axs = plt.subplots(2, 2, figsize=(10, 10))

                def plot_snake(
                    ax: Any, img_bg: Any, snake: Any, emap_disp: Any, title: Any
                ):
                    """plot_snake."""
                    # We display the EDGE MAP in the background to show what the snake sees/follows,
                    # or the original image? Figure 11.3 shows EDGE MAPS in top row, Snake on Original in bottom.
                    # But here we have 4 subplots. Let's show Snake on Original, but maybe an overlay?
                    # Let's stick to Snake on Original Data (g), but title explains the process.

                    ax.imshow(g, cmap="gray")
                    ax.plot(
                        init_snake[:, 1], init_snake[:, 0], "--r", lw=1, label="Init"
                    )
                    ax.plot(snake[:, 1], snake[:, 0], "-g", lw=2, label="Final")
                    ax.set_title(title)
                    ax.axis("off")

                # To strictly match Figure 11.3 layout (4 quadrants, showing results of 4 configs)
                plot_snake(axs[0, 0], g, snake1, emap1, "Both (Filter Img & Map)")
                plot_snake(axs[0, 1], g, snake2, emap2, "After (Filter Map Only)")
                plot_snake(axs[1, 0], g, snake3, emap3, "Before (Filter Img Only)")
                plot_snake(axs[1, 1], g, snake4, emap4, "None (No Filtering)")

                # Optional: Save edge maps for debugging?
                # fig_maps, axs_maps = plt.subplots(2, 2)
                # axs_maps[0,0].imshow(emap1, cmap='gray'); axs_maps[0,0].set_title("Map Both")
                # axs_maps[0,1].imshow(emap2, cmap='gray'); axs_maps[0,1].set_title("Map After")
                # ...

                plt.tight_layout()
                plt.savefig("Figure113_skimage.png")
                print("Saved Figure113_skimage.png")

                # Also save Figure113_maps.png to allow user to see the intermediate potentials
                fig_m, ax_m = plt.subplots(2, 2, figsize=(10, 10))
                ax_m[0, 0].imshow(emap1, cmap="gray")
                ax_m[0, 0].set_title("Energy Map: Both")
                ax_m[0, 1].imshow(emap2, cmap="gray")
                ax_m[0, 1].set_title("Energy Map: After")
                ax_m[1, 0].imshow(emap3, cmap="gray")
                ax_m[1, 0].set_title("Energy Map: Before")
                ax_m[1, 1].imshow(emap4, cmap="gray")
                ax_m[1, 1].set_title("Energy Map: None")
                plt.tight_layout()
                plt.savefig("Figure113_skimage_maps.png")
                print("Saved Figure113_skimage_maps.png")

                plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure114(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter11 script `Figure114.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from scipy.io import loadmat
            from helpers.libdip.snakeMap4e import snakeMap4e
            from helpers.libdip.snakeForce4e import snakeForce4e
            from helpers.libdipum.data_path import dip_data

            # Parameters
            T = 0.001
            Sig = 15
            NSig = 3
            Order = "both"
            NIter = 300

            # Data
            img_path = dip_data("noisy-elliptical-object.tif")
            g = imread(img_path)

            # Load initial snake data (overrides g if present, as in MATLAB)
            mat = loadmat("Figure112.mat")

            # Edge map
            emap = snakeMap4e(g, T, Sig, NSig, Order)

            # Snake force using plain gradient
            FTx, FTy = snakeForce4e(emap, "gradient")

            # Normalize forces
            mag = np.sqrt(FTx**2 + FTy**2)
            FTx = FTx / (mag + 1e-10)
            FTy = FTy / (mag + 1e-10)

            # Threshold by magnitude to suppress small vectors
            mag = np.sqrt(FTx**2 + FTy**2)
            FTx = np.where(mag > 0.35, FTx, 0)
            FTy = np.where(mag > 0.35, FTy, 0)

            # Reduce density
            FTxr = FTx[::15, ::15]
            FTyr = FTy[::15, ::15]

            # Display
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.quiver(
                np.flipud(FTyr),
                np.flipud(-FTxr),
                angles="xy",
                scale_units="xy",
                scale=1,
            )
            ax.set_title("Vector snake force")
            ax.set_aspect("equal", adjustable="box")

            plt.tight_layout()
            plt.savefig("Figure114.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure115(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter11 script `Figure115.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from scipy.io import loadmat
            from helpers.libdip.snakeMap4e import snakeMap4e
            from helpers.libdip.snakeForce4e import snakeForce4e
            from helpers.libdipum.data_path import dip_data

            # Parameters
            T = 0.001
            Sig = 15
            NSig = 3
            Order = "after"

            # Data
            img_path = dip_data("noisy-elliptical-object.tif")
            g = imread(img_path)

            # Load initial snake data (overrides g if present)
            mat = loadmat("Figure112.mat")

            # Edge map
            emap = snakeMap4e(g, T, Sig, NSig, Order)

            # Snake force using plain gradient
            FTx, FTy = snakeForce4e(emap, "gradient")

            # Normalize the forces
            mag = np.sqrt(FTx**2 + FTy**2)
            FTx = FTx / (mag + 1e-10)
            FTy = FTy / (mag + 1e-10)

            # Reduce density
            FTxr = FTx[::10, ::10]
            FTyr = FTy[::10, ::10]

            # Display
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.quiver(
                np.flipud(FTyr),
                np.flipud(-FTxr),
                angles="xy",
                scale_units="xy",
                scale=1,
            )
            ax.set_title("Vector snake force")
            ax.set_aspect("equal", adjustable="box")

            plt.tight_layout()
            plt.savefig("Figure115.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure116(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter11 script `Figure116.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from scipy.io import loadmat
            from helpers.libdip.snakeMap4e import snakeMap4e
            from helpers.libdip.snakeForce4e import snakeForce4e
            from helpers.libdip.snakeIterate4e import snakeIterate4e
            from helpers.libdip.snakeReparam4e import snakeReparam4e
            from helpers.libdipum.snake_display import snake_display
            from helpers.libdipum.data_path import dip_data

            # Parameters
            Alpha = 0.5
            Beta = 2.0
            Gamma = 5.0
            T = 0.001
            Sig = 15
            NSig = 3
            NIter = 265

            # Data
            img_path = dip_data("noisy-elliptical-object.tif")
            g = imread(img_path)

            # Load initial snake
            mat = loadmat("Figure112.mat")
            xi = mat["xi"].squeeze()
            yi = mat["yi"].squeeze()

            # Edge map
            emap = snakeMap4e(g, T, Sig, NSig, "both")

            # Snake force
            FTx, FTy = snakeForce4e(emap, "gradient")

            # Normalize forces
            mag = np.sqrt(FTx**2 + FTy**2)
            FTx = FTx / (mag + 1e-10)
            FTy = FTy / (mag + 1e-10)

            # Without reparametrization
            x1 = xi.copy()
            y1 = yi.copy()
            for _ in range(NIter):
                x1, y1 = snakeIterate4e(Alpha, Beta, Gamma, x1, y1, 1, FTx, FTy)

            # Reparametrization after last iteration
            x2 = xi.copy()
            y2 = yi.copy()
            for _ in range(NIter):
                x2, y2 = snakeIterate4e(Alpha, Beta, Gamma, x2, y2, 1, FTx, FTy)
            x2, y2 = snakeReparam4e(x2, y2)

            # Reparametrization every ten iterations
            x3 = xi.copy()
            y3 = yi.copy()
            for i in range(1, NIter + 1):
                x3, y3 = snakeIterate4e(Alpha, Beta, Gamma, x3, y3, 1, FTx, FTy)
                if i % 10 == 0:
                    x3, y3 = snakeReparam4e(x3, y3)

            # Reparametrization every iteration
            x4 = xi.copy()
            y4 = yi.copy()
            for _ in range(NIter):
                x4, y4 = snakeIterate4e(Alpha, Beta, Gamma, x4, y4, 1, FTx, FTy)
                x4, y4 = snakeReparam4e(x4, y4)

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))

            axes[0, 0].imshow(g, cmap="gray")
            axes[0, 0].axis("off")
            plt.sca(axes[0, 0])
            snake_display(x1, y1, "g.")

            axes[0, 1].imshow(g, cmap="gray")
            axes[0, 1].axis("off")
            plt.sca(axes[0, 1])
            snake_display(x2, y2, "g.")

            axes[1, 0].imshow(g, cmap="gray")
            axes[1, 0].axis("off")
            plt.sca(axes[1, 0])
            snake_display(x3, y3, "g.")

            axes[1, 1].imshow(g, cmap="gray")
            axes[1, 1].axis("off")
            plt.sca(axes[1, 1])
            snake_display(x4, y4, "g.")

            plt.tight_layout()
            plt.savefig("Figure116.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure117(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter11 script `Figure117.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            """Figure 11.7 - Different starting configuration for segmenting ellipse."""

            from typing import Any

            import os
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread

            from helpers.libdip.snakeMap4e import snakeMap4e
            from helpers.libdip.snakeForce4e import snakeForce4e
            from helpers.libdip.snakeIterate4e import snakeIterate4e
            from helpers.libdip.snakeReparam4e import snakeReparam4e
            from helpers.libdipum.snake_display import snake_display
            from helpers.libdipum.data_path import dip_data

            print("Running Figure117...")

            def Process(
                alpha: Any,
                beta: Any,
                gamma: Any,
                x: Any,
                y: Any,
                FTx: Any,
                FTy: Any,
                n_iter: Any,
            ):
                """Process."""
                # Iterate.
                for _ in range(n_iter):
                    x, y = snakeIterate4e(alpha, beta, gamma, x, y, 1, FTx, FTy)
                    x, y = snakeReparam4e(x, y)

                # Redistribute one last time.
                x, y = snakeReparam4e(x, y)
                return x, y

            # Parameters
            Alpha = 0.05
            Beta = [0.0, 2.0, 4.0]
            Gamma = 0.6
            T = 0.001
            Sig = 15
            NSig = 3
            NIter = [200, 400, 400]

            # Data
            img_path = dip_data("noisy-elliptical-object.tif")
            g = imread(img_path)

            # Initial snake coordinates: circle
            # MATLAB: t = 0:0.05:2*pi
            t = np.arange(0.0, 2.0 * np.pi + 0.05, 0.05)
            xi = 320.0 + 200.0 * np.cos(t)
            yi = 320.0 + 200.0 * np.sin(t)

            # Close snake
            xi = np.concatenate([xi, [xi[0]]])
            yi = np.concatenate([yi, [yi[0]]])

            # Edge map and external forces
            emap = snakeMap4e(g, T, Sig, NSig, "both")
            FTx, FTy = snakeForce4e(emap, "gradient")

            # Normalize force
            mag = np.sqrt(FTx**2 + FTy**2)
            FTx = FTx / (mag + 1e-10)
            FTy = FTy / (mag + 1e-10)

            # Process for each beta
            x_res = []
            y_res = []
            for k in range(len(Beta)):
                xk, yk = Process(
                    Alpha, Beta[k], Gamma, xi.copy(), yi.copy(), FTx, FTy, NIter[k]
                )
                x_res.append(xk)
                y_res.append(yk)

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            axes[0, 0].imshow(g, cmap="gray")
            axes[0, 0].axis("off")
            axes[0, 0].plot(np.r_[yi, yi[0]], np.r_[xi, xi[0]], ".g")

            axes[0, 1].imshow(g, cmap="gray")
            axes[0, 1].axis("off")
            plt.sca(axes[0, 1])
            snake_display(x_res[0], y_res[0], ".g")

            axes[1, 0].imshow(g, cmap="gray")
            axes[1, 0].axis("off")
            plt.sca(axes[1, 0])
            snake_display(x_res[1], y_res[1], ".g")

            axes[1, 1].imshow(g, cmap="gray")
            axes[1, 1].axis("off")
            plt.sca(axes[1, 1])
            snake_display(x_res[2], y_res[2], ".g")

            out_path = _os.path.join(_os.environ.get("DIP4E_OUTPUT_DIR", str(_Path(__file__).resolve().parents[2] / "output")), "Figure117.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved {out_path}")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure118(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter11 script `Figure118.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            """Figure 11.8 - Snake segmentation of 957-by-1024 rose."""

            import os
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy.io import loadmat, savemat
            from skimage.io import imread

            from helpers.libdipum.snake_manual_input import snake_manual_input
            from helpers.libdip.snakeMap4e import snakeMap4e
            from helpers.libdip.snakeForce4e import snakeForce4e
            from helpers.libdip.snakeIterate4e import snakeIterate4e
            from helpers.libdip.snakeReparam4e import snakeReparam4e
            from helpers.libdipum.snake_display import snake_display
            from helpers.libdip.intScaling4e import intScaling4e
            from helpers.libdipum.data_path import dip_data

            print("Running Figure118...")

            def _im2double_like(a: np.ndarray) -> np.ndarray:
                """_im2double_like."""
                a = np.asarray(a)
                if np.issubdtype(a.dtype, np.floating):
                    return a.astype(np.float64)
                if a.dtype == np.uint8:
                    return a.astype(np.float64) / 255.0
                if a.dtype == np.uint16:
                    return a.astype(np.float64) / 65535.0
                if a.dtype == np.int16:
                    return a.astype(np.float64) / 32768.0
                return a.astype(np.float64)

            # Data
            img_path = dip_data("rose957by1024.tif")
            f = imread(img_path)
            if f.ndim == 3:
                f = f[..., 0]

            mat_path = dip_data("Figure118.mat")
            data = loadmat(mat_path)
            if "f" in data:
                f = np.asarray(data["f"])
            xi = np.asarray(data.get("xi", np.array([]))).squeeze()
            yi = np.asarray(data.get("yi", np.array([]))).squeeze()

            # Guard against malformed workspace files (empty xi/yi).
            if np.size(xi) < 3 or np.size(yi) < 3:
                t0 = np.linspace(0, 2 * np.pi, 150, endpoint=False)
                rr = min(f.shape[:2]) * 0.18
                xc = f.shape[1] / 2.0
                yc = f.shape[0] / 2.0
                xi = xc + rr * np.cos(t0)
                yi = yc + rr * np.sin(t0)

            # Parameters
            T = 0.005
            Sig = 11
            NSig = 5
            NIter = 400
            Alpha = 10 * 0.05
            Beta = 0.5
            Gamma = 5

            # Edge map
            emap = snakeMap4e(f, T, Sig, NSig, "both")
            emap = _im2double_like(intScaling4e(emap))

            # Snake force
            FTx, FTy = snakeForce4e(emap, "gradient")

            # Normalize
            mag = np.sqrt(FTx**2 + FTy**2)
            FTx = FTx / (mag + 1e-10)
            FTy = FTy / (mag + 1e-10)

            x = np.asarray(xi).copy()
            y = np.asarray(yi).copy()

            # Iterate
            for _ in range(NIter):
                x, y = snakeIterate4e(Alpha, Beta, Gamma, x, y, 1, FTx, FTy)
                x, y = snakeReparam4e(x, y)

            # Redistribute once more
            x, y = snakeReparam4e(x, y)

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            axes[0, 0].imshow(f, cmap="gray")
            axes[0, 0].axis("off")
            plt.sca(axes[0, 0])
            snake_display(xi, yi, "g.")

            axes[0, 1].imshow(emap, cmap="gray")
            axes[0, 1].axis("off")

            axes[1, 0].imshow(emap, cmap="gray")
            axes[1, 0].axis("off")
            plt.sca(axes[1, 0])
            snake_display(x, y, "g.")

            axes[1, 1].imshow(f, cmap="gray")
            axes[1, 1].axis("off")
            plt.sca(axes[1, 1])
            snake_display(x, y, "g.")

            out_path = "Figure118.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved {out_path}")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure119(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter11 script `Figure119.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.libdip.snakeMap4e import snakeMap4e
            from helpers.libdip.snakeForce4e import snakeForce4e
            from helpers.libdip.snakeIterate4e import snakeIterate4e
            from helpers.libdip.snakeReparam4e import snakeReparam4e
            from helpers.libdipum.snake_display import snake_display
            from helpers.libdip.intScaling4e import intScaling4e
            from helpers.libdipum.data_path import dip_data

            # Data
            img_path = dip_data("U200.tif")
            f = imread(img_path)

            # Parameters
            T = 0.001
            Sig = 3
            NSig = 1
            NIter = 1000
            Alpha = 0.06
            Beta = 0
            Gamma = 1

            # Specify a circle
            t = np.arange(0, 2 * np.pi + 0.1, 0.1)
            xi = 100 + 80 * np.cos(t)
            yi = 100 + 80 * np.sin(t)

            # Close the snake
            xi = np.concatenate([xi, [xi[0]]])
            yi = np.concatenate([yi, [yi[0]]])

            # Edge map
            emap = snakeMap4e(f, T, Sig, NSig, "after")

            # Scale to range [0,1]
            emap = intScaling4e(emap)

            # Snake force
            FTx, FTy = snakeForce4e(emap, "gradient")

            # Normalize it
            mag = np.sqrt(FTx**2 + FTy**2)
            FTx = FTx / (mag + np.finfo(float).eps)
            FTy = FTy / (mag + np.finfo(float).eps)

            # Process
            x = xi.copy()
            y = yi.copy()

            for _ in range(NIter):
                x, y = snakeIterate4e(Alpha, Beta, Gamma, x, y, 1, FTx, FTy)
                x, y = snakeReparam4e(x, y)

            # Redistribute one last time
            x, y = snakeReparam4e(x, y)

            # Display figure 1
            fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))
            axes2[0, 0].imshow(f, cmap="gray")
            axes2[0, 0].axis("off")
            axes2[0, 0].plot(np.append(yi, yi[0]), np.append(xi, xi[0]), "k.")

            axes2[0, 1].imshow(emap, cmap="gray")
            axes2[0, 1].axis("off")

            axes2[1, 0].quiver(np.flipud(FTy[::2, ::2]), np.flipud(-FTx[::2, ::2]))
            axes2[1, 0].axis("off")

            axes2[1, 1].imshow(f, cmap="gray")
            axes2[1, 1].axis("off")
            plt.sca(axes2[1, 1])
            snake_display(x, y, "g.")

            plt.tight_layout()
            plt.savefig("Figure118.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)
