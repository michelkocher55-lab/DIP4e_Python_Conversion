from pathlib import Path as _Path
import os as _os

from typing import Any


class Chapter09Mixin:
    def figure911(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter09 script `Figure911.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from PIL import Image
            import sys
            from pathlib import Path

            # Add project root so local ia870 package can be imported when run directly.
            PROJECT_ROOT = Path(str(script_path)).resolve().parents[2]
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            import ia870 as ia
            from helpers.libdipum.data_path import dip_data

            # %% Figure911

            # %% Data
            f = np.array(Image.open(dip_data("fingerprint-noisy.tif")))
            if f.ndim == 3:
                f = f[..., 0]

            # %% SE
            B1 = ia.iasebox(1)

            # %% Erosion
            f1 = ia.iaero(f, B1)

            # %% Opening
            f2 = ia.iaopen(f, B1)

            # %% Dilation of the opening
            f3 = ia.iadil(f2, B1)

            # %% Closing of the opening
            f4 = ia.iaclose(f2, B1)

            # %% Display
            fig, ax = plt.subplots(
                2, 3, figsize=(12, 8), sharex=True, sharey=True, num=1
            )
            try:
                fig.canvas.manager.set_window_title("Figure 9.11")
            except Exception:
                pass

            ax = ax.ravel()

            ax[0].imshow(f, cmap="gray")
            ax[0].set_title("f")
            ax[0].axis("off")

            ax[1].imshow(f1, cmap="gray")
            ax[1].set_title(r"f1 = $\epsilon_{B1}(f)$")
            ax[1].axis("off")

            ax[2].imshow(f2, cmap="gray")
            ax[2].set_title(r"f2 = $\gamma_{B1}(f)$")
            ax[2].axis("off")

            ax[3].imshow(f3, cmap="gray")
            ax[3].set_title(r"f3 = $\delta_{B1}(f2)$")
            ax[3].axis("off")

            ax[4].imshow(f4, cmap="gray")
            ax[4].set_title(r"f4 = $\phi_{B1}(f2)$")
            ax[4].axis("off")

            ax[5].axis("off")

            plt.tight_layout()
            fig.savefig("Figure911.png", dpi=150, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure912(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter09 script `Figure912.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            import ia870 as ia
            from helpers.libgeneral.mmshow import mmshow

            # %% Figure912

            # %% Data
            X = np.zeros((14, 17), dtype=bool)
            X[1:10, 1:6] = True
            X[8:13, 7:12] = True
            X[4:7, 13:16] = True

            # %% Interval
            BBGImg = np.zeros((7, 7), dtype=bool)
            BFGImg = np.zeros((7, 7), dtype=bool)
            BFGImg[1:6, 1:6] = True
            BBGImg = ia.iaframe(BBGImg)

            BFG = ia.iaimg2se(BFGImg)
            BBG = ia.iaimg2se(BBGImg)

            I = ia.iase2hmt(BFG, BBG)

            # %% HMT
            Y_hmt = ia.iasupgen(X, I)

            Y_eroFG = ia.iaero(X, BFG)
            Y_eroBG = ia.iaero(ia.ianeg(X), BBG)

            inter = ia.iaintersec(Y_eroFG, Y_eroBG)
            ok = np.all(Y_hmt == inter)
            print(f"OK = {ok}")

            # %% Display
            Grid = np.ones_like(BFGImg, dtype=bool)

            fig = plt.figure(1, figsize=(14, 7))

            plt.subplot(2, 4, 1)
            plt.imshow(ia.iabshow(Grid, BFGImg))
            plt.title("B_{FG}")
            plt.axis("off")

            plt.subplot(2, 4, 2)
            plt.imshow(ia.iabshow(Grid, BBGImg))
            plt.title("B_{BG}")
            plt.axis("off")

            plt.subplot(2, 4, 3)
            plt.imshow(ia.iabshow(Grid, BFGImg, ia.ianeg(ia.iaunion(BFGImg, BBGImg))))
            plt.title("B = (B_{FG}, B_{BG})")
            plt.axis("off")

            plt.subplot(2, 4, 4)
            plt.imshow(X, cmap="gray")
            plt.title("X")
            plt.axis("off")

            plt.subplot(2, 4, 5)
            plt.imshow(Y_eroFG, cmap="gray")
            plt.title(r"$\epsilon_{B_{FG}}(X)$")
            plt.axis("off")

            plt.subplot(2, 4, 6)
            plt.imshow(ia.ianeg(X), cmap="gray")
            plt.title(r"$X^C$")
            plt.axis("off")

            plt.subplot(2, 4, 7)
            plt.imshow(Y_eroBG, cmap="gray")
            plt.title(r"$\epsilon_{B_{BG}}(X^C)$")
            plt.axis("off")

            plt.subplot(2, 4, 8)
            mmshow(X, inter)
            plt.title(r"$\epsilon_{B_{FG}}(X) \cap \epsilon_{B_{BG}}(X^C)$")
            plt.axis("off")

            plt.tight_layout()
            fig.savefig("Figure912.png", dpi=150, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure914(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter09 script `Figure914.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            import ia870 as ia
            from helpers.libgeneral.mmshow import mmshow

            # %% Figure914

            # %% Data
            X = np.zeros((7, 15), dtype=bool)
            X[1:6, 1:5] = True
            X[1:6, -5:-1] = True
            X[2:5, 5:10] = True
            X[3, 5] = False
            X[3, 9] = False

            # %% Interval
            I1FG = np.ones((3, 3), dtype=bool)
            I1FG[1, 1] = False
            I1BG = np.logical_not(I1FG)
            I1 = ia.iase2hmt(ia.iaimg2se(I1FG), ia.iaimg2se(I1BG))
            ia.iaintershow(I1)

            I2FG = np.zeros((3, 3), dtype=bool)
            I2FG[1:, 0:2] = True
            I2BG = np.logical_not(I2FG)
            I2 = ia.iase2hmt(ia.iaimg2se(I2FG), ia.iaimg2se(I2BG))
            ia.iaintershow(I2)

            I3FG = np.array(
                [
                    [False, False, False],
                    [True, True, False],
                    [False, False, False],
                ],
                dtype=bool,
            )
            I3BG = np.array(
                [
                    [False, False, True],
                    [False, False, True],
                    [False, False, True],
                ],
                dtype=bool,
            )
            I3 = ia.iase2hmt(ia.iaimg2se(I3FG), ia.iaimg2se(I3BG))
            ia.iaintershow(I3)

            # %% HMT
            Y1 = ia.iasupgen(X, I1)
            Y2 = ia.iasupgen(X, I2)
            Y3 = ia.iasupgen(X, I3)

            # %% Display
            fig = plt.figure(1, figsize=(10, 8))

            plt.subplot(2, 2, 1)
            plt.imshow(X, cmap="gray")
            plt.title("X")
            plt.axis("off")

            plt.subplot(2, 2, 2)
            mmshow(X, Y1)
            plt.title("HMT(X, I1)")
            plt.axis("off")

            plt.subplot(2, 2, 3)
            mmshow(X, Y2)
            plt.title("HMT(X, I2)")
            plt.axis("off")

            plt.subplot(2, 2, 4)
            mmshow(X, Y3)
            plt.title("HMT(X, I3)")
            plt.axis("off")

            plt.tight_layout()
            fig.savefig("Figure914.png", dpi=150, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure915(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter09 script `Figure915.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            import ia870 as ia

            # %% SE
            Bc4 = ia.iasecross()
            Bc8 = ia.iasebox()

            # %% Data
            X = np.ones((7, 12), dtype=bool)
            X[1:3, 4] = False
            X[1:3, -2] = False
            X = ia.iaintersec(X, np.logical_not(ia.iaframe(X)))

            # %% Erosion
            Xe4 = ia.iaero(X, Bc4)
            Xe8 = ia.iaero(X, Bc8)

            # %% Gradient
            Grad4 = ia.iasubm(X, Xe4)
            Grad8 = ia.iasubm(X, Xe8)

            # %% Display
            fig = plt.figure(1, figsize=(12, 8))

            plt.subplot(2, 3, 1)
            plt.imshow(X, cmap="gray")
            plt.title("X")
            plt.axis("off")

            plt.subplot(2, 3, 2)
            plt.imshow(Xe4, cmap="gray")
            plt.title(r"Xe4 = $\epsilon_{B_4}(X)$")
            plt.axis("off")

            plt.subplot(2, 3, 3)
            plt.imshow(Grad4, cmap="gray")
            plt.title(r"$\rho_4(X) = X-Xe4$")
            plt.axis("off")

            plt.subplot(2, 3, 5)
            plt.imshow(Xe8, cmap="gray")
            plt.title(r"Xe8 = $\epsilon_{B_8}(X)$")
            plt.axis("off")

            plt.subplot(2, 3, 6)
            plt.imshow(Grad8, cmap="gray")
            plt.title(r"$\rho_8(X) = X-Xe8$")
            plt.axis("off")

            plt.tight_layout()
            fig.savefig("Figure915.png", dpi=150, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure916(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter09 script `Figure916.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from PIL import Image
            import ia870 as ia
            from helpers.libdipum.data_path import dip_data

            # %% Figure916

            # %% Data
            f = np.array(Image.open(dip_data("lincoln.tif")))
            if f.ndim == 3:
                f = f[..., 0]

            # %% SE
            B1 = ia.iasebox(1)
            B0 = ia.iasebox(0)

            # %% Erosion
            f1 = ia.iaero(f, B1)

            # %% Inner gradient
            f2 = ia.iagradm(f, B0, B1)

            # %% Outer gradient
            f3 = ia.iagradm(f, B1, B0)

            # %% Beucher gradient
            f4 = ia.iagradm(f, B1, B1)

            # %% Display
            fig, ax = plt.subplots(
                2, 2, figsize=(10, 8), sharex=True, sharey=True, num=1
            )
            try:
                fig.canvas.manager.set_window_title("Figure 9.16")
            except Exception:
                pass

            ax = ax.ravel()

            ax[0].imshow(f, cmap="gray")
            ax[0].set_title("f")
            ax[0].axis("off")

            ax[1].imshow(f2, cmap="gray")
            ax[1].set_title(r"f2 = $f - \epsilon_{B1}(f)$")
            ax[1].axis("off")

            ax[2].imshow(f3, cmap="gray")
            ax[2].set_title(r"f3 = $\delta_{B1}(f) - f$")
            ax[2].axis("off")

            ax[3].imshow(f4, cmap="gray")
            ax[3].set_title(r"f4 = $\delta_{B1}(f) - \epsilon_{B1}(f)$")
            ax[3].axis("off")

            plt.tight_layout()
            fig.savefig("Figure916.png", dpi=150, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure917(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter09 script `Figure917.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            import ia870 as ia

            # %% Figure917
            # Hole filling

            # %% Parameters
            R = 3
            C = 3

            # %% Data
            f = np.zeros((10, 7), dtype=bool)
            f[1, 2:4] = True
            f[2:4, [1, 4]] = True
            f[4:6, [2, 4]] = True
            f[6:8, [1, 5]] = True
            f[8, 1:5] = True

            # %% SE
            B = ia.iasecross(1)

            # %% Manual hole filling
            X_list = []

            x0 = np.zeros_like(f, dtype=bool)
            x0[R - 1, C - 1] = True
            X_list.append(x0)

            x1 = ia.iaintersec(ia.ianeg(f), ia.iadil(X_list[0], B))
            X_list.append(x1)

            while True:
                xk = ia.iaintersec(ia.ianeg(f), ia.iadil(X_list[-1], B))
                X_list.append(xk)
                if np.array_equal(X_list[-1], X_list[-2]):
                    break

            g = ia.iaunion(f, X_list[-1])

            # %% Display
            fig = plt.figure(1, figsize=(12, 9))

            plt.subplot(3, 4, 1)
            plt.imshow(f, cmap="gray")
            plt.title("A")
            plt.axis("off")

            plt.subplot(3, 4, 2)
            plt.imshow(ia.ianeg(f), cmap="gray")
            plt.title("A^C")
            plt.axis("off")

            for iter_idx in range(len(X_list)):
                plt.subplot(3, 4, iter_idx + 3)
                plt.imshow(X_list[iter_idx], cmap="gray")
                plt.title(f"X_{{{iter_idx}}}")
                plt.axis("off")

            plt.tight_layout()
            fig.savefig("Figure917.png", dpi=150, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure918(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter09 script `Figure918.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            import matplotlib
            from PIL import Image
            import time
            import ia870 as ia
            from helpers.libdipum.data_path import dip_data

            # %% Figure918
            # Hole filling

            # %% Data
            f_img = np.array(Image.open(dip_data("region-filling-reflections.tif")))
            if f_img.ndim == 3:
                f_img = f_img[..., 0]
            f = f_img > 128

            # %% SE
            B = ia.iasecross(1)

            # %% Manual hole filling (interactive seed)
            fig = plt.figure(1, figsize=(6, 4))
            ax_seed = plt.subplot(1, 1, 1)
            ax_seed.imshow(f, cmap="gray")
            ax_seed.set_title("Click one seed point")
            ax_seed.axis("off")

            # Click one point on figure 1 (no Enter required).
            seed = {"pt": None}

            def _on_click(event: Any):
                """_on_click."""
                if (
                    event.inaxes is ax_seed
                    and event.xdata is not None
                    and event.ydata is not None
                ):
                    seed["pt"] = (event.xdata, event.ydata)
                    print(
                        f"Seed selected at (row={int(round(event.ydata))}, col={int(round(event.xdata))})"
                    )

            cid = fig.canvas.mpl_connect("button_press_event", _on_click)
            plt.tight_layout()
            suppressed_show = plt.show
            plt.show = _ctx["old_show"]
            try:
                plt.show(block=False)
                plt.pause(0.1)
                try:
                    # Bring seed window to front (helps avoid the first click being just focus).
                    fig.canvas.manager.window.raise_()
                except Exception:
                    pass

                print(f"Matplotlib backend: {matplotlib.get_backend()}")
                print("Click once in the image window to set the seed point.")

                t0 = time.time()
                timeout_s = 60.0
                while (
                    seed["pt"] is None
                    and plt.fignum_exists(fig.number)
                    and (time.time() - t0 < timeout_s)
                ):
                    plt.pause(0.05)
            finally:
                plt.show = suppressed_show

            fig.canvas.mpl_disconnect(cid)
            plt.close(fig)

            if seed["pt"] is None:
                # Fallback if click is not captured (backend/IDE issue).
                C = f.shape[1] // 2
                R = f.shape[0] // 2
                print(f"No click captured. Using fallback seed at (row={R}, col={C}).")
            else:
                C, R = seed["pt"]

            r0 = int(np.clip(np.round(R), 1, f.shape[0]))
            c0 = int(np.clip(np.round(C), 1, f.shape[1]))

            X_list = []
            x0 = np.zeros_like(f, dtype=bool)
            x0[r0 - 1, c0 - 1] = True
            X_list.append(x0)

            x1 = ia.iaintersec(ia.ianeg(f), ia.iadil(X_list[0], B))
            X_list.append(x1)

            while True:
                xk = ia.iaintersec(ia.ianeg(f), ia.iadil(X_list[-1], B))
                X_list.append(xk)
                if np.array_equal(X_list[-1], X_list[-2]):
                    break

            g = np.maximum(f, X_list[-1])

            # %% Display
            fig2 = plt.figure(2, figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(f, cmap="gray")
            plt.title("f")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(X_list[-1], cmap="gray")
            plt.title(
                f"X_{{{len(X_list)}}}, X_k = d(X_{{k-1}}) ∩ ~f, X_1 = d({r0}, {c0})"
            )
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(g, cmap="gray")
            plt.title(f"g = X_{{{len(X_list)}}} ∪ f")
            plt.axis("off")

            plt.tight_layout()
            fig2.savefig("Figure918.png", dpi=150, bbox_inches="tight")
            print("Saved Figure918.png from figure 2.")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure919(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter09 script `Figure919.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            import ia870 as ia

            # %% Figure919

            # %% SE
            Bc4 = ia.iasecross()
            Bc8 = ia.iasebox()

            # %% Data
            X = np.zeros((10, 10), dtype=bool)
            X[1:3, 6:9] = True
            X[1:3, -2] = True
            X[3, [5, 6, 8]] = True
            X[4, 5:9] = True
            X[5, 3:6] = True
            X[6, 2:5] = True
            X[7, [1, 4]] = True
            X[8, 2:5] = True

            # %% Conditional dilation
            LesX = np.zeros((10, 10, 7), dtype=bool)
            LesX[6, 3, 0] = True
            for iter_idx in range(1, 7):
                LesX[:, :, iter_idx] = ia.iaintersec(
                    ia.iadil(LesX[:, :, iter_idx - 1], ia.iasebox()), X
                )

            # %% Display
            fig = plt.figure(1, figsize=(10, 8))

            plt.subplot(2, 3, 1)
            plt.imshow(X, cmap="gray")
            plt.axis("off")

            plt.subplot(2, 3, 2)
            plt.imshow(LesX[:, :, 0], cmap="gray")
            plt.axis("off")

            plt.subplot(2, 3, 3)
            plt.imshow(LesX[:, :, 1], cmap="gray")
            plt.axis("off")

            plt.subplot(2, 3, 4)
            plt.imshow(LesX[:, :, 2], cmap="gray")
            plt.axis("off")

            plt.subplot(2, 3, 5)
            plt.imshow(LesX[:, :, 3], cmap="gray")
            plt.axis("off")

            plt.subplot(2, 3, 6)
            plt.imshow(LesX[:, :, 6], cmap="gray")
            plt.axis("off")

            plt.tight_layout()
            fig.savefig("Figure919.png", dpi=150, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure920(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter09 script `Figure920.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from PIL import Image
            from skimage.measure import regionprops
            import sys
            from pathlib import Path
            import ia870 as ia
            from helpers.libdipum.data_path import dip_data

            # %% Figure920

            # %% Init
            Fig = 1

            # %% Parameters
            Bc = ia.iasecross(1)

            # %% Data
            f = np.array(Image.open(dip_data("Chickenfilet-with-bones.tif")))
            if f.ndim == 3:
                f = f[..., 0]

            # %% Thresholding
            Threshold = 200
            X = f > Threshold

            Bin = np.arange(0, 256)
            Count, _ = np.histogram(
                f.ravel().astype(float), bins=np.arange(-0.5, 256.5, 1.0)
            )

            # %% Opening
            B = ia.iasebox(1)
            X1 = ia.iaopen(X, B)

            # %% Labeling
            L = ia.ialabel(X1, Bc)
            L = np.asarray(L)

            # %% Area stat
            Stat = regionprops(L.astype(np.int32))
            areas = [s.area for s in Stat]

            # %% Display
            fig = plt.figure(Fig, figsize=(12, 8))

            plt.subplot(2, 3, 1)
            plt.imshow(f, cmap="gray")
            plt.title("f")
            plt.axis("off")

            plt.subplot(2, 3, 2)
            plt.bar(Bin[1:], Count[1:])
            plt.title("Hist(f)")
            plt.axvline(Threshold, color="r")
            plt.axis("tight")
            plt.gca().set_box_aspect(1)

            plt.subplot(2, 3, 3)
            plt.imshow(X, cmap="gray")
            plt.title(f"X = T_{{{int(round(Threshold))}}}(f)")
            plt.axis("off")

            plt.subplot(2, 3, 4)
            plt.imshow(X1, cmap="gray")
            plt.title(r"X1 = $\gamma_B(X)$")
            plt.axis("off")

            plt.subplot(2, 3, 5)
            max_label = int(np.max(L))
            if max_label > 0:
                lut = plt.cm.jet(np.linspace(0, 1, max_label + 1))
                lut[0, :3] = 0.0
                from matplotlib.colors import ListedColormap

                cmap = ListedColormap(lut[:, :3])
                plt.imshow(L, cmap=cmap, vmin=0, vmax=max_label)
            else:
                plt.imshow(L, cmap="gray")
            plt.title("L = Label_4(X1)")
            plt.axis("off")

            plt.subplot(2, 3, 6)
            plt.bar(np.arange(1, len(areas) + 1), areas)
            plt.xlabel("Connected Part")
            plt.title("Area")
            plt.gca().set_box_aspect(1)
            plt.axis("tight")

            plt.tight_layout()
            fig.savefig("Figure920.png", dpi=150, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure921(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter09 script `Figure921.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            import ia870 as ia
            from helpers.libgeneral.mmshow import mmshow

            # %% Figure921

            # %% Init
            Fig = 1
            try:
                _raw = input("Figure921 (1) or Problem921c (2) : ").strip()
                Choice = int(_raw) if _raw else 1
            except Exception:
                Choice = 1

            # %% Data
            if Choice == 1:
                X = np.zeros((12, 11), dtype=bool)
                X[1, 3:6] = True
                X[2, 2:7] = True
                X[3, 2:6] = True
                X[4, 4:6] = True
                X[4, 7] = True
                X[5, 4:6] = True
                X[5, 7] = True
                X[6, 4:7] = True
                X[7, 6] = True
                X[8, 6] = True
                X[9, 5:7] = True
                X[10, 5] = True

                XX = np.zeros((14, 13), dtype=bool)
                XX[1:-1, 1:-1] = X
            elif Choice == 2:
                XX = np.zeros((13, 13), dtype=bool)
                XX[2:11, 5:8] = True
                XX[5:8, 2:11] = True
            else:
                raise ValueError("Plouc")

            NIter = 10

            # %% Interval
            BFG = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=bool)
            BBG = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=bool)

            B1 = ia.iase2hmt(ia.iaimg2se(BFG), ia.iaimg2se(BBG))
            B2 = ia.iainterot(B1, 90)
            B3 = ia.iainterot(B2, 90)
            B4 = ia.iainterot(B3, 90)

            # Optional interval visualization (as in MATLAB script)
            ia.iaintershow(B1)
            ia.iaintershow(B2)
            ia.iaintershow(B3)
            ia.iaintershow(B4)

            B = [B1, B2, B3, B4]
            NB = len(B)

            # %% Convex hull
            Frame = ia.iaframe(XX)
            ConvexHull = np.zeros_like(XX, dtype=bool)

            Y_all = []
            HMT_all = []
            MaxIter = []

            for interval_idx in range(NB):
                y_series = [XX.copy()]  # Y(:,:,Interval,1)
                h_series = []  # HMT(:,:,Interval,iter-1)

                OK = True
                iter_idx = 2
                while OK:
                    h = ia.iasupgen(y_series[iter_idx - 2], B[interval_idx])
                    y = ia.iaunion(y_series[iter_idx - 2], h)
                    y = ia.iaintersec(y, ia.ianeg(Frame))

                    h_series.append(h)
                    y_series.append(y)

                    if np.any(y_series[iter_idx - 2] != y):
                        iter_idx += 1
                        if iter_idx > NIter + 2:
                            # Safety cap; MATLAB version expects convergence before this.
                            OK = False
                            MaxIter.append(iter_idx)
                    else:
                        OK = False
                        MaxIter.append(iter_idx)

                Y_all.append(y_series)
                HMT_all.append(h_series)
                ConvexHull = ia.iaunion(ConvexHull, y_series[-1])

            # %% Display
            saved_figs = []

            for interval_idx in range(4):
                fig = plt.figure(Fig, figsize=(12, 6))
                saved_figs.append(fig)
                Fig += 1

                plt.subplot(2, 5, 1)
                plt.imshow(Y_all[interval_idx][0], cmap="gray")
                plt.title("X")
                plt.axis("off")

                max_it = MaxIter[interval_idx]
                # MATLAB: for iter = 2 : MaxIter(Interval)-1
                for iter_disp in range(2, max_it):
                    sp_idx = iter_disp
                    if sp_idx > 10:
                        break
                    plt.subplot(2, 5, sp_idx)

                    y_disp = Y_all[interval_idx][iter_disp - 1]
                    h_disp = HMT_all[interval_idx][iter_disp - 2]
                    mmshow(y_disp, h_disp)
                    plt.title(f"X^{{I={interval_idx + 1}}}_{{iter={iter_disp}}}")
                    plt.axis("off")

                plt.tight_layout()

            fig = plt.figure(Fig, figsize=(12, 8))
            saved_figs.append(fig)
            Fig += 1

            plt.subplot(2, 3, 1)
            plt.imshow(XX, cmap="gray")
            plt.title("X")
            plt.axis("off")

            for interval_idx in range(4):
                plt.subplot(2, 3, interval_idx + 2)
                plt.imshow(Y_all[interval_idx][-1], cmap="gray")
                plt.title(
                    f"Y^{{I={interval_idx + 1}}}_{{iter = {MaxIter[interval_idx]}}}"
                )
                plt.axis("off")

            plt.subplot(2, 3, 6)
            mmshow(ConvexHull, XX)
            plt.title("Convex hull, original data")
            plt.axis("off")

            plt.tight_layout()

            # %% Print 2 file
            for iter_idx in range(1, 6):
                if iter_idx - 1 < len(saved_figs):
                    saved_figs[iter_idx - 1].savefig(
                        f"Figure921_{iter_idx}.png", dpi=150, bbox_inches="tight"
                    )

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure922(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter09 script `Figure922.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            import ia870 as ia
            from helpers.libgeneral.mmshow import mmshow

            # %% Figure922
            # ibid 921 but with spatial limitation
            # Convex hull, it is not a thickening !!

            Fig = 1
            try:
                _raw = input("Figure921 (1) or Problem921c (2) : ").strip()
                Choice = int(_raw) if _raw else 1
            except Exception:
                Choice = 1

            # %% Data
            if Choice == 1:
                X = np.zeros((12, 11), dtype=bool)
                X[1, 3:6] = True
                X[2, 2:7] = True
                X[3, 2:6] = True
                X[4, 4:6] = True
                X[4, 7] = True
                X[5, 4:6] = True
                X[5, 7] = True
                X[6, 4:7] = True
                X[7, 6] = True
                X[8, 6] = True
                X[9, 5:7] = True
                X[10, 5] = True

                XX = np.zeros((14, 13), dtype=bool)
                XX[1:-1, 1:-1] = X
            elif Choice == 2:
                XX = np.zeros((13, 13), dtype=bool)
                XX[2:11, 5:8] = True
                XX[5:8, 2:11] = True
            else:
                raise ValueError("Plouc")

            NIter = 10

            # %% Mask (careful: mask is for extended XX)
            Mask = np.zeros_like(XX, dtype=bool)
            Mask[2:12, 3:9] = True

            # %% Interval
            BFG = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=bool)
            BBG = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=bool)

            B1 = ia.iase2hmt(ia.iaimg2se(BFG), ia.iaimg2se(BBG))
            B2 = ia.iainterot(B1, 90)
            B3 = ia.iainterot(B2, 90)
            B4 = ia.iainterot(B3, 90)

            ia.iaintershow(B1)
            ia.iaintershow(B2)
            ia.iaintershow(B3)
            ia.iaintershow(B4)

            B = [B1, B2, B3, B4]
            NB = len(B)

            # %% Convex hull
            Frame = ia.iaframe(XX)
            ConvexHull = np.zeros_like(XX, dtype=bool)

            Y_all = []
            HMT_all = []
            MaxIter = []

            for interval_idx in range(NB):
                y_series = [XX.copy()]  # Y(:,:,Interval,1)
                h_series = []  # HMT(:,:,Interval,iter-1)

                OK = True
                iter_idx = 2
                while OK:
                    # Difference from Fig921: spatial mask limitation on HMT.
                    h_raw = ia.iasupgen(y_series[iter_idx - 2], B[interval_idx])
                    h = ia.iaintersec(Mask, h_raw)

                    y = ia.iaunion(y_series[iter_idx - 2], h)
                    y = ia.iaintersec(y, ia.ianeg(Frame))  # remove border

                    h_series.append(h)
                    y_series.append(y)

                    if np.any(y_series[iter_idx - 2] != y):
                        iter_idx += 1
                        if iter_idx > NIter + 2:
                            OK = False
                            MaxIter.append(iter_idx)
                    else:
                        OK = False
                        MaxIter.append(iter_idx)

                Y_all.append(y_series)
                HMT_all.append(h_series)
                ConvexHull = ia.iaunion(ConvexHull, y_series[-1])

            # %% Display
            for interval_idx in range(4):
                fig = plt.figure(Fig, figsize=(12, 6))
                Fig += 1

                plt.subplot(2, 5, 1)
                plt.imshow(Y_all[interval_idx][0], cmap="gray")
                plt.title("X")
                plt.axis("off")

                max_it = MaxIter[interval_idx]
                for iter_disp in range(2, max_it):
                    if iter_disp > 10:
                        break
                    plt.subplot(2, 5, iter_disp)
                    y_disp = Y_all[interval_idx][iter_disp - 1]
                    h_disp = HMT_all[interval_idx][iter_disp - 2]
                    mmshow(y_disp, h_disp)
                    plt.title(f"X^{{I={interval_idx + 1}}}_{{iter={iter_disp}}}")
                    plt.axis("off")

                plt.tight_layout()

            fig5 = plt.figure(Fig, figsize=(12, 8))
            Fig += 1

            plt.subplot(2, 3, 1)
            plt.imshow(XX, cmap="gray")
            plt.title("X")
            plt.axis("off")

            for interval_idx in range(4):
                plt.subplot(2, 3, interval_idx + 2)
                plt.imshow(Y_all[interval_idx][-1], cmap="gray")
                plt.title(
                    f"Y^{{I={interval_idx + 1}}}_{{iter = {MaxIter[interval_idx]}}}"
                )
                plt.axis("off")

            plt.subplot(2, 3, 6)
            mmshow(ConvexHull, XX)
            plt.title("Convex hull, original data")
            plt.axis("off")

            plt.tight_layout()

            # %% Print
            # MATLAB: print('-f5', ..., 'Figure922.png')
            fig5.savefig("Figure922.png", dpi=150, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure923(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter09 script `Figure923.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            import ia870 as ia

            # %% Figure923

            # %% Init
            Fig = 1
            NR = 3
            NC = 8
            SDCCompare = 1

            # %% Data
            Choix = 1
            if Choix == 1:
                X = np.ones((7, 13), dtype=bool)
                X[0, :] = False
                X[-1, :] = False
                X[:, 0] = False
                X[:, -1] = False
                X[-2, 4:6] = False
                X[2:, -3:-1] = False
            elif Choix == 2:
                X = np.array(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ],
                    dtype=bool,
                )
            else:
                raise ValueError("Plouc")

            # %% SDC Manual Thinning
            fig1 = plt.figure(Fig, figsize=(16, 6))
            Fig += 1

            Y1 = X.copy()
            Iter1 = 1
            while True:
                YOld1 = Y1.copy()

                L = ia.iahomothin()
                ia.iaintershow(L)

                for cpt in range(1, 9):
                    HMT1 = ia.iasupgen(Y1, L)
                    Y1 = ia.iasubm(Y1, HMT1)

                    subplot_idx = cpt + (Iter1 - 1) * NC
                    if subplot_idx <= NR * NC:
                        plt.subplot(NR, NC, subplot_idx)
                        plt.imshow(Y1, cmap="gray")
                        plt.axis("off")
                        plt.title(f"theta={45 * (cpt - 1)}, iter={Iter1}")

                    L = ia.iainterot(L, 45)
                    ia.iaintershow(L)

                if np.array_equal(YOld1, Y1):
                    break
                else:
                    Iter1 += 1

            # %% SDC All in one thinning
            Y2 = ia.iathin(X, ia.iahomothin(), -1, 45, "CLOCKWISE")

            # %% Check
            if SDCCompare:
                OK2 = np.array_equal(Y1, Y2)
                print(f"OK2 = {OK2}")

            # %% Final display
            fig2 = plt.figure(Fig, figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(X, cmap="gray")
            plt.title("X")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(Y1, cmap="gray")
            plt.title("Y")
            plt.axis("off")

            plt.tight_layout()

            # %% Print
            fig1.savefig("Figure923.png", dpi=150, bbox_inches="tight")
            fig2.savefig("Figure923Bis.png", dpi=150, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure926(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter09 script `Figure926.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            import ia870 as ia

            # %% Figure926
            # Maximum ball skeleton
            # Reconstruction using a non homotopic skeleton

            # %% Init
            Fig = 1

            # %% Data
            X = np.ones((12, 7), dtype=bool)
            X[0, :] = False
            X[-1, :] = False
            X[:, 0] = False
            X[:, -1] = False

            X[2:6, 1] = False
            X[1, 2:6] = False
            X[2, 4:6] = False
            X[3, 4:6] = False
            X[4:6, 5] = False

            # %% Structuring element
            B = ia.iasebox(1)

            # %% Process
            levels = 3  # k = 0..2
            nr, nc = X.shape

            Ero = np.zeros((nr, nc, levels), dtype=bool)
            Open = np.zeros((nr, nc, levels), dtype=bool)
            IndividualSkeleton = np.zeros((nr, nc, levels), dtype=bool)
            Skeleton = np.zeros((nr, nc, levels), dtype=bool)
            DilIndividualSkeleton = np.zeros((nr, nc, levels), dtype=bool)
            ReconstructedImage = np.zeros((nr, nc, levels), dtype=bool)

            for k in range(levels):
                Ero[:, :, k] = ia.iaero(X, ia.iasebox(k))
                Open[:, :, k] = ia.iaopen(Ero[:, :, k], B)
                IndividualSkeleton[:, :, k] = ia.iasubm(Ero[:, :, k], Open[:, :, k])

                if k == 0:
                    Skeleton[:, :, k] = IndividualSkeleton[:, :, k]
                else:
                    Skeleton[:, :, k] = ia.iaaddm(
                        Skeleton[:, :, k - 1], IndividualSkeleton[:, :, k]
                    )

                DilIndividualSkeleton[:, :, k] = ia.iadil(
                    IndividualSkeleton[:, :, k], ia.iasebox(k)
                )

                if k == 0:
                    ReconstructedImage[:, :, k] = DilIndividualSkeleton[:, :, k]
                else:
                    ReconstructedImage[:, :, k] = ia.iaaddm(
                        ReconstructedImage[:, :, k - 1], DilIndividualSkeleton[:, :, k]
                    )

            # %% Display
            fig = plt.figure(Fig, figsize=(18, 9))

            for k in range(levels):
                plt.subplot(3, 6, 1 + 6 * k)
                plt.imshow(Ero[:, :, k], cmap="gray")
                plt.title(f"ero(A, {k}B)")
                plt.axis("off")

                plt.subplot(3, 6, 2 + 6 * k)
                plt.imshow(Open[:, :, k], cmap="gray")
                plt.title(f"open(ero(A, {k}B), B)")
                plt.axis("off")

                plt.subplot(3, 6, 3 + 6 * k)
                plt.imshow(IndividualSkeleton[:, :, k], cmap="gray")
                plt.title(f"S_{k}(A)")
                plt.axis("off")

                plt.subplot(3, 6, 4 + 6 * k)
                plt.imshow(Skeleton[:, :, k], cmap="gray")
                plt.title("union(S_k(A))")
                plt.axis("off")

                plt.subplot(3, 6, 5 + 6 * k)
                plt.imshow(DilIndividualSkeleton[:, :, k], cmap="gray")
                plt.title(f"S_{k}(A) dil {k}B")
                plt.axis("off")

                plt.subplot(3, 6, 6 + 6 * k)
                plt.imshow(ReconstructedImage[:, :, k], cmap="gray")
                plt.title(f"union(S_{k}(A) dil {k}B)")
                plt.axis("off")

            plt.tight_layout()
            fig.savefig("Figure926.png", dpi=150, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure927(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter09 script `Figure927.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            import ia870 as ia
            from helpers.libgeneral.mmshow import mmshow

            # %% Figure927
            # Pruning

            # %% Parameters
            NR = 3
            NC = 8

            # %% Data
            X = np.zeros((12, 16), dtype=bool)
            X[1, 6:9] = True
            X[2, [5, 9]] = True
            X[3, [1, 4]] = True
            X[4, [1, 3, 10]] = True
            X[5, [1, 2, 10, 11]] = True
            X[6, [2, 9, 11]] = True
            X[7, [3, 9, 11]] = True
            X[8, [2, 9, 11]] = True
            X[9, [2, 8, 11, 14]] = True
            X[10, np.r_[3:8, 12:14]] = True

            # %% Pruning
            BFG0 = ia.iaimg2se(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=bool))
            BBG0 = ia.iaimg2se(np.array([[0, 0, 0], [1, 0, 1], [1, 1, 1]], dtype=bool))

            Prune = X.copy()
            for Iter in range(1, 4):
                BFG = BFG0.copy()
                BBG = BBG0.copy()

                for cpt in range(1, 9):
                    if cpt == 1:
                        BFG = ia.iaimg2se(
                            np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=bool)
                        )
                        BBG = ia.iaimg2se(
                            np.array([[0, 1, 1], [0, 0, 1], [0, 1, 1]], dtype=bool)
                        )
                    elif cpt == 5:
                        BFG = ia.iaimg2se(
                            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=bool)
                        )
                        BBG = ia.iaimg2se(
                            np.array([[0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)
                        )

                    HMT = ia.iasupgen(Prune, ia.iase2hmt(BFG, BBG))
                    Prune = ia.iasubm(Prune, HMT)

                    BFG = ia.iaserot(BFG, 90, "CLOCKWISE")
                    BBG = ia.iaserot(BBG, 90, "CLOCKWISE")

            # %% Endpoints detection
            EndPoints = np.zeros_like(X, dtype=bool)
            for cpt in range(1, 9):
                if cpt == 1:
                    BFG = ia.iaimg2se(
                        np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=bool)
                    )
                    BBG = ia.iaimg2se(
                        np.array([[0, 1, 1], [0, 0, 1], [0, 1, 1]], dtype=bool)
                    )
                elif cpt == 5:
                    BFG = ia.iaimg2se(
                        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=bool)
                    )
                    BBG = ia.iaimg2se(
                        np.array([[0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)
                    )

                EndPoints = ia.iaaddm(
                    EndPoints, ia.iasupgen(Prune, ia.iase2hmt(BFG, BBG))
                )

                BFG = ia.iaserot(BFG, 90, "CLOCKWISE")
                BBG = ia.iaserot(BBG, 90, "CLOCKWISE")

            # %% Conditional dilation
            CondDil = EndPoints.copy()
            for cpt in range(1, 4):
                CondDil = ia.iaintersec(X, ia.iadil(CondDil, ia.iasebox(1)))

            # %% Final union
            Final = ia.iaunion(CondDil, Prune)

            # %% Display
            fig = plt.figure(1, figsize=(12, 8))
            try:
                fig.canvas.manager.set_window_title("Figure 9.27")
            except Exception:
                pass

            plt.subplot(2, 3, 1)
            plt.imshow(X, cmap="gray")
            plt.title("X")
            plt.axis("off")

            plt.subplot(2, 3, 2)
            plt.imshow(Prune, cmap="gray")
            plt.title("Prune = Thin(3, X)")
            plt.axis("off")

            plt.subplot(2, 3, 3)
            plt.imshow(EndPoints, cmap="gray")
            plt.title("EndPoints = HMT(Prune, E)")
            plt.axis("off")

            plt.subplot(2, 3, 4)
            mmshow(CondDil, EndPoints)
            plt.title("CondDil = delta^3_X(EndPoints)")
            plt.axis("off")

            plt.subplot(2, 3, 5)
            plt.imshow(Final, cmap="gray")
            plt.title("Final = Prune U CondDil")
            plt.axis("off")

            plt.tight_layout()
            fig.savefig("Figure927.png", dpi=150, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure93(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter09 script `Figure93.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt

            from helpers.libgeneral.mmshow import mmshow
            import ia870 as ia

            # %% Figure93

            # %% Data
            X = np.ones((7, 15), dtype=bool)
            X[0, :] = False
            X[-1, :] = False
            X[:, 0] = False
            X[:, -1] = False
            X[1, 5:10] = False  # MATLAB: X(2, 6:10) = false
            X[-2, 5:10] = False  # MATLAB: X(end-1, 6:10) = false

            # %% SE
            B = ia.iasebox()

            # %% Erosion
            Y = ia.iaero(X, B)

            # %% Display
            fig = plt.figure(1, figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(X, cmap="gray")
            plt.title("X")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(Y, cmap="gray")
            plt.title("Y")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            mmshow(X, Y)
            plt.title("X, Y")
            plt.axis("off")

            plt.tight_layout()
            fig.savefig("Figure93.png", dpi=150, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure931(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter09 script `Figure931.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from PIL import Image
            import ia870 as ia
            from helpers.libdipum.data_path import dip_data

            # %% Figure931

            # %% Parameters
            Bc4 = ia.iasecross()

            # %% Data (mask)
            img_name = dip_data("text-image.tif")
            g = np.array(Image.open(img_name))

            if g.ndim == 3:
                g = g[..., 0]
            g = g > 0

            # %% Erosion
            B = ia.iaseline(41, 90)
            X1 = ia.iaero(g, B)

            # %% Opening (marker)
            f0 = ia.iaopen(g, B)

            # %% Reconstruction by dilation
            try:
                _raw = input("Fast (1) or iterative (2) : ").strip()
                Choix = int(_raw) if _raw else 1
            except Exception:
                Choix = 1

            f_iters = [f0]

            if Choix == 1:
                X3 = ia.iainfrec(f0, g, Bc4)
            elif Choix == 2:
                k = 0
                while True:
                    nxt = ia.iaintersec(ia.iadil(f_iters[k], Bc4), g)
                    f_iters.append(nxt)
                    if np.array_equal(f_iters[k], f_iters[k + 1]):
                        break
                    else:
                        k += 1
                X3 = f_iters[k]
                print(k + 1)
            else:
                raise ValueError("Plouc")

            # %% Display
            fig, ax = plt.subplots(
                2, 2, figsize=(10, 8), sharex=True, sharey=True, num=1
            )
            try:
                fig.canvas.manager.set_window_title("Figure9.31")
            except Exception:
                pass

            ax = ax.ravel()

            ax[0].imshow(g, cmap="gray")
            ax[0].set_title("g")
            ax[0].axis("off")

            ax[1].imshow(X1, cmap="gray")
            ax[1].set_title(r"X1 = $\epsilon_B(g)$")
            ax[1].axis("off")

            ax[2].imshow(f0, cmap="gray")
            ax[2].set_title(r"X1 = $\gamma_B(g)$")
            ax[2].axis("off")

            ax[3].imshow(X3, cmap="gray")
            ax[3].set_title(r"X3 = $R^{\delta}(X1, g)$")
            ax[3].axis("off")

            plt.tight_layout()
            fig.savefig("Figure931.png", dpi=150, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure933(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter09 script `Figure933.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            import ia870 as ia
            from helpers.libgeneral import MKRLib
            from helpers.libdipum.data_path import dip_data

            # %% Figure933
            # Hole closing without marker 9.5-28, 9.5-29

            # %% Init
            Fig = 1

            # %% Data
            img_name = dip_data("text-image.tif")
            Mask = imread(img_name)

            if Mask.ndim == 3:
                Mask = Mask[..., 0]
            Mask = Mask > 0

            MaskC = ia.ianeg(Mask)

            # %% Marker
            Frame = np.zeros_like(Mask, dtype=bool)
            Frame[0, :] = True
            Frame[-1, :] = True
            Frame[:, 0] = True
            Frame[:, -1] = True

            Marker1 = ia.iaintersec(Frame, Mask)
            Marker2 = ia.iaintersec(np.logical_not(Marker1), Frame)

            # %% Reconstruction
            Temp = ia.iainfrec(Marker2, MaskC, ia.iasecross(1))
            FillHole = ia.ianeg(Temp)

            # %% Display (Figure 1)
            fig1, ax = plt.subplots(
                2, 2, figsize=(10, 8), sharex=True, sharey=True, num=Fig
            )
            Fig += 1
            ax = ax.ravel()

            ax[0].imshow(Mask, cmap="gray")
            ax[0].set_title("Mask")
            ax[0].axis("off")

            ax[1].imshow(Frame, cmap="gray")
            ax[1].set_title("Frame")
            ax[1].axis("off")

            ax[2].imshow(Marker1, cmap="gray")
            ax[2].set_title("Marker1 = Frame ∩ Mask")
            ax[2].axis("off")

            ax[3].imshow(ia.ianeg(Marker1), cmap="gray")
            ax[3].set_title("not Marker1")
            ax[3].axis("off")

            fig1.tight_layout()

            # %% Display (Figure 2)
            fig2, bx = plt.subplots(
                2, 2, figsize=(10, 8), sharex=True, sharey=True, num=Fig
            )
            bx = bx.ravel()

            bx[0].imshow(ia.iadil(Marker2), cmap="gray")
            bx[0].set_title("Marker2 = not(Marker1) and Frame")
            bx[0].axis("off")

            bx[1].imshow(MaskC, cmap="gray")
            bx[1].set_title("Mask complement")
            bx[1].axis("off")

            bx[2].imshow(Temp, cmap="gray")
            bx[2].set_title("Reconstruction by dilation")
            bx[2].axis("off")

            bx[3].imshow(FillHole, cmap="gray")
            bx[3].set_title("Complement of reconstruction")
            bx[3].axis("off")

            fig2.tight_layout()

            # %% Print
            fig1.savefig("Figure933.png", dpi=150, bbox_inches="tight")
            fig2.savefig("Figure933Bis.png", dpi=150, bbox_inches="tight")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure934(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter09 script `Figure934.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            import ia870 as ia
            from helpers.libdipum.data_path import dip_data

            # %% Figure934
            # Border clearing 9.5-31

            # %% Init
            Fig = 1

            # %% Mask
            img_name = dip_data("text-image.tif")
            Mask = imread(img_name)
            if Mask.ndim == 3:
                Mask = Mask[..., 0]
            Mask = Mask > 0

            # %% Marker
            Marker = np.zeros_like(Mask, dtype=bool)
            Marker[0, :] = True
            Marker[-1, :] = True
            Marker[:, 0] = True
            Marker[:, -1] = True
            Marker = np.logical_and(Marker, Mask)

            # %% Reconstruction
            BorderChar = ia.iainfrec(Marker, Mask, ia.iasecross(1))
            NonBorderChar = ia.iasubm(Mask, BorderChar)

            # %% Display
            fig, ax = plt.subplots(
                2, 2, figsize=(10, 8), sharex=True, sharey=True, num=Fig
            )
            ax = ax.ravel()

            ax[0].imshow(Mask, cmap="gray")
            ax[0].set_title("Mask")
            ax[0].axis("off")

            ax[1].imshow(Marker, cmap="gray")
            ax[1].set_title("Marker = Frame and Mask")
            ax[1].axis("off")

            ax[2].imshow(BorderChar, cmap="gray")
            ax[2].set_title("Border characters from reconstruction")
            ax[2].axis("off")

            ax[3].imshow(NonBorderChar, cmap="gray")
            ax[3].set_title("Non-border characters = Mask minus BorderChar")
            ax[3].axis("off")

            plt.tight_layout()
            fig.savefig("Figure934.png", dpi=150, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure937(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter09 script `Figure937.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from PIL import Image

            import ia870 as ia
            from helpers.libdipum.data_path import dip_data

            # %% Figure937

            # %% Data
            f = np.array(Image.open(dip_data("circuitboard-section.tif")))
            if f.ndim == 3:
                f = f[..., 0]

            # %% SE
            B1 = ia.iasedisk(2)

            # %% Erosion and dilation
            f1 = ia.iaero(f, B1)
            f2 = ia.iadil(f, B1)

            # %% Display
            fig, ax = plt.subplots(
                2, 2, figsize=(10, 8), sharex=True, sharey=True, num=1
            )
            try:
                fig.canvas.manager.set_window_title("Figure 9.35")
            except Exception:
                pass

            ax = ax.ravel()

            ax[0].imshow(f, cmap="gray")
            ax[0].set_title("f")
            ax[0].axis("off")

            ax[1].axis("off")

            ax[2].imshow(f1, cmap="gray")
            ax[2].set_title("f1 = erosion with disk radius 2")
            ax[2].axis("off")

            ax[3].imshow(f2, cmap="gray")
            ax[3].set_title("f2 = dilation with disk radius 2")
            ax[3].axis("off")

            plt.tight_layout()
            fig.savefig("Figure937.png", dpi=150, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure939(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter09 script `Figure939.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from PIL import Image
            import ia870 as ia
            from helpers.libgeneral.AddSE2Image import AddSE2Image
            from helpers.libdipum.data_path import dip_data

            # %% Figure939
            # Morphological opening and closing

            # %% Data
            f = np.array(Image.open(dip_data("circuitboard-section.tif")))
            if f.ndim == 3:
                f = f[..., 0]

            # %% SE
            B1 = ia.iasedisk(5)

            # %% Opening and closing
            f1 = ia.iaopen(f, B1)
            f2 = ia.iaclose(f, B1)

            # %% Display
            fig, ax = plt.subplots(
                2, 2, figsize=(10, 8), sharex=True, sharey=True, num=1
            )
            try:
                fig.canvas.manager.set_window_title("Figure 9.37")
            except Exception:
                pass

            ax = ax.ravel()

            ax[0].imshow(AddSE2Image(f, B1, int(np.max(f))), cmap="gray")
            ax[0].set_title("f, B1")
            ax[0].axis("off")

            ax[1].axis("off")

            ax[2].imshow(f1, cmap="gray")
            ax[2].set_title("f1 = opening with disk radius 5")
            ax[2].axis("off")

            ax[3].imshow(f2, cmap="gray")
            ax[3].set_title("f2 = closing with disk radius 5")
            ax[3].axis("off")

            plt.tight_layout()
            fig.savefig("Figure939.png", dpi=150, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure940(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter09 script `Figure940.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from PIL import Image

            import ia870 as ia
            from helpers.libgeneral.AddSE2Image import AddSE2Image
            from helpers.libgeneral.GPsnr import GPsnr
            from helpers.libgeneral.matlab_hist import matlab_hist
            from helpers.libdipum.data_path import dip_data

            # %% Figure940
            # Alternate Sequential Filtering

            # %% Data
            f = np.array(Image.open(dip_data("cygnusloop.tif")))
            if f.ndim == 3:
                f = f[..., 0]

            # %% SE
            Rad = [1, 3, 5]
            B = [ia.iasedisk(r) for r in Rad]

            # %% CO
            CO = np.zeros((f.shape[0], f.shape[1], 3), dtype=f.dtype)
            SNR_CO = np.zeros(3, dtype=float)
            InfoCO = [""] * 3

            for cpt in range(3):
                CO[:, :, cpt] = ia.iaclose(ia.iaopen(f, B[cpt]), B[cpt])
                gg = CO[:, :, cpt].astype(float)
                InfoCO[cpt] = f"C_{Rad[cpt]}(O_{Rad[cpt]}(f))"
                SNR_CO[cpt] = GPsnr(f.astype(float), gg)

            # %% ASF (manual, then iaasf overwrite as in MATLAB script)
            ASF = np.zeros((f.shape[0], f.shape[1], 3), dtype=f.dtype)
            SNR_ASF = np.zeros(3, dtype=float)

            for cpt in range(3):
                if cpt == 0:
                    ASF[:, :, cpt] = ia.iaclose(ia.iaopen(f, B[cpt]), B[cpt])
                else:
                    ASF[:, :, cpt] = ia.iaclose(
                        ia.iaopen(ASF[:, :, cpt - 1], B[cpt]), B[cpt]
                    )
                hh = ASF[:, :, cpt].astype(float)
                SNR_ASF[cpt] = GPsnr(f.astype(float), hh)

            for cpt in range(3):
                ASF[:, :, cpt] = ia.iaasf(
                    f.astype(np.uint8), "CO", ia.iasecross(1), cpt + 1
                )
                gg = ASF[:, :, cpt].astype(float)
                SNR_ASF[cpt] = GPsnr(f.astype(float), gg)

            InfoASF = [
                "C_1(O_1(f))",
                "C_3(O_3(C_1(O_1(f))))",
                "C_5(O_5(C_3(O_3(C_1(O_1(f))))))",
            ]

            # %% Display figure 1 (CO)
            fig1 = plt.figure(1, figsize=(10, 8))

            plt.subplot(2, 2, 1)
            plt.imshow(f, cmap="gray")
            plt.title("f0")
            plt.axis("off")

            for iter_idx in range(3):
                plt.subplot(2, 2, 2 + iter_idx)
                mx = int(np.max(CO[:, :, iter_idx]))
                plt.imshow(
                    AddSE2Image(CO[:, :, iter_idx], ia.iasedisk(Rad[iter_idx]), mx),
                    cmap="gray",
                )
                plt.title(f"{InfoCO[iter_idx]}, SNR = {SNR_CO[iter_idx]:.3f} dB")
                plt.axis("off")

            plt.tight_layout()
            fig1.savefig("Figure940.png", dpi=150, bbox_inches="tight")

            # %% Display figure 2 (ASF)
            fig2 = plt.figure(2, figsize=(10, 8))

            plt.subplot(2, 2, 1)
            plt.imshow(f, cmap="gray")
            plt.title("f0")
            plt.axis("off")

            for iter_idx in range(3):
                plt.subplot(2, 2, 2 + iter_idx)
                mx = int(np.max(ASF[:, :, iter_idx]))
                plt.imshow(
                    AddSE2Image(ASF[:, :, iter_idx], ia.iasedisk(Rad[iter_idx]), mx),
                    cmap="gray",
                )
                plt.title(f"{InfoASF[iter_idx]}, SNR = {SNR_ASF[iter_idx]:.3f} dB")
                plt.axis("off")

            plt.tight_layout()
            fig2.savefig("Figure940Bis.png", dpi=150, bbox_inches="tight")

            # %% Display figure 3 (differences/hist)
            fig3 = plt.figure(3, figsize=(10, 8))

            plt.subplot(2, 2, 1)
            plt.imshow(ia.iasubm(CO[:, :, -1], f), cmap="gray", vmin=0, vmax=50)
            plt.title("f3 - f")
            plt.axis("off")

            plt.subplot(2, 2, 2)
            e = f.astype(float) - CO[:, :, -1].astype(float)
            emin, emax = float(np.min(e)), float(np.max(e))
            centers = np.linspace(emin, emax, 10) if emax > emin else np.array([emin])
            counts = matlab_hist(e.ravel(), centers)
            plt.bar(
                centers,
                counts,
                width=(centers[1] - centers[0]) if centers.size > 1 else 1.0,
            )
            plt.title("Histogram of f - f3")

            plt.subplot(2, 2, 3)
            plt.imshow(ia.iasubm(ASF[:, :, -1], f), cmap="gray", vmin=0, vmax=50)
            plt.title("ASF(f) - f")
            plt.axis("off")

            plt.subplot(2, 2, 4)
            e1 = f.astype(float) - ASF[:, :, -1].astype(float)
            e1min, e1max = float(np.min(e1)), float(np.max(e1))
            centers1 = (
                np.linspace(e1min, e1max, 10) if e1max > e1min else np.array([e1min])
            )
            counts1 = matlab_hist(e1.ravel(), centers1)
            plt.bar(
                centers1,
                counts1,
                width=(centers1[1] - centers1[0]) if centers1.size > 1 else 1.0,
            )
            plt.title("Histogram of f - ASF(f)")

            plt.tight_layout()

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure940new(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter09 script `Figure940New.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from PIL import Image
            import sys
            from pathlib import Path

            # Add project root so local packages can be imported when run directly.
            PROJECT_ROOT = Path(str(script_path)).resolve().parents[2]
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            import ia870 as ia
            from helpers.libgeneral.AddSE2Image import AddSE2Image
            from helpers.libdipum.data_path import dip_data

            # %% Figure940New
            # Alternate Sequential Filtering

            # %% Data
            f = np.array(Image.open(dip_data("cygnusloop.tif")))
            if f.ndim == 3:
                f = f[..., 0]

            # %% SE
            Rad = list(range(1, 6))
            B = [ia.iasedisk(r) for r in Rad]

            # %% CO
            CO = np.zeros((f.shape[0], f.shape[1], len(Rad)), dtype=f.dtype)
            InfoCO = [""] * len(Rad)
            for cpt in range(len(Rad)):
                CO[:, :, cpt] = ia.iaclose(ia.iaopen(f, B[cpt]), B[cpt])
                InfoCO[cpt] = f"f_{cpt + 1} = C_{Rad[cpt]}(O_{Rad[cpt]}(f_{cpt}))"

            # %% ASF
            ASF = np.zeros((f.shape[0], f.shape[1], len(Rad)), dtype=f.dtype)
            for cpt in range(len(Rad)):
                ASF[:, :, cpt] = ia.iaasf(f, "CO", B[cpt])

            InfoASF = [
                "C_1(O_1(f))",
                "C_2(O_2(C_1(O_1(f))))",
                "C_3(O_3(C_2(O_2(C_1(O_1(f))))))",
                "ASFCO_4(f)",
                "ASFCO_5(f)",
            ]

            # %% ASFRec
            ASFRec = np.zeros((f.shape[0], f.shape[1], len(Rad)), dtype=f.dtype)
            for cpt in range(len(Rad)):
                ASFRec[:, :, cpt] = ia.iaasfrec(
                    f.astype(np.uint8), "CO", ia.iasecross(1), ia.iasecross(1), cpt + 1
                )

            InfoASFRec = [
                "ASFRec(f, CO, 1)",
                "ASFRec(f, CO, 2)",
                "ASFRec(f, CO, 3)",
                "ASFRec(f, CO, 4)",
                "ASFRec(f, CO, 5)",
            ]

            # %% Display figure 1 (CO)
            fig1 = plt.figure(1, figsize=(12, 8))
            try:
                fig1.canvas.manager.set_window_title("Figure 9.40")
            except Exception:
                pass

            plt.subplot(2, 3, 1)
            plt.imshow(f, cmap="gray")
            plt.title("f0")
            plt.axis("off")

            for iter_idx in range(len(Rad)):
                plt.subplot(2, 3, 2 + iter_idx)
                mx = int(np.max(CO[:, :, iter_idx]))
                plt.imshow(
                    AddSE2Image(CO[:, :, iter_idx], B[iter_idx], mx), cmap="gray"
                )
                plt.title(InfoCO[iter_idx])
                plt.axis("off")

            plt.tight_layout()
            fig1.savefig("Figure940New.png", dpi=150, bbox_inches="tight")

            # %% Display figure 2 (ASF)
            fig2 = plt.figure(2, figsize=(12, 8))

            plt.subplot(2, 3, 1)
            plt.imshow(f, cmap="gray")
            plt.title("f0")
            plt.axis("off")

            for iter_idx in range(len(Rad)):
                plt.subplot(2, 3, 2 + iter_idx)
                mx = int(np.max(ASF[:, :, iter_idx]))
                plt.imshow(
                    AddSE2Image(ASF[:, :, iter_idx], B[iter_idx], mx), cmap="gray"
                )
                plt.title(InfoASF[iter_idx])
                plt.axis("off")

            plt.tight_layout()
            fig2.savefig("Figure940NewBis.png", dpi=150, bbox_inches="tight")

            # %% Display figure 3 (ASFRec)
            fig3 = plt.figure(3, figsize=(12, 8))

            plt.subplot(2, 3, 1)
            plt.imshow(f, cmap="gray")
            plt.title("f0")
            plt.axis("off")

            for iter_idx in range(len(Rad)):
                plt.subplot(2, 3, 2 + iter_idx)
                mx = int(np.max(ASFRec[:, :, iter_idx]))
                plt.imshow(
                    AddSE2Image(ASFRec[:, :, iter_idx], B[iter_idx], mx), cmap="gray"
                )
                plt.title(InfoASFRec[iter_idx])
                plt.axis("off")

            plt.tight_layout()
            fig3.savefig("Figure940NewTer.png", dpi=150, bbox_inches="tight")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure941(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter09 script `Figure941.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from PIL import Image
            import sys
            from pathlib import Path

            # Add project root so local packages can be imported when run directly.
            PROJECT_ROOT = Path(str(script_path)).resolve().parents[2]
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            import ia870 as ia
            from helpers.libgeneral.AddSE2Image import AddSE2Image
            from helpers.libdipum.data_path import dip_data

            # %% Figure941
            # Morphological gradient

            # %% Data
            f = np.array(Image.open(dip_data("headCT.tif")))
            if f.ndim == 3:
                f = f[..., 0]

            # %% SE
            B = ia.iasedisk(2)

            # %% Morphological gradients
            g1 = ia.iadil(f, B)
            g2 = ia.iaero(f, B)
            g3 = ia.iasubm(g1, g2)  # Beucher gradient
            g4 = ia.iasubm(f, g2)  # Internal gradient
            g5 = ia.iasubm(g1, f)  # External gradient

            # %% Display
            fig, ax = plt.subplots(
                2, 3, figsize=(12, 8), sharex=True, sharey=True, num=1
            )
            try:
                fig.canvas.manager.set_window_title("Figure 9.39")
            except Exception:
                pass

            ax = ax.ravel()

            ax[0].imshow(AddSE2Image(f, ia.iasedisk(2), int(np.max(f))), cmap="gray")
            ax[0].set_title("f, B")
            ax[0].axis("off")

            ax[1].imshow(g1, cmap="gray")
            ax[1].set_title("g1 = dilation of f")
            ax[1].axis("off")

            ax[2].imshow(g2, cmap="gray")
            ax[2].set_title("g2 = erosion of f")
            ax[2].axis("off")

            ax[3].imshow(g3, cmap="gray")
            ax[3].set_title("Beucher gradient = g1 - g2")
            ax[3].axis("off")

            ax[4].imshow(g4, cmap="gray")
            ax[4].set_title("Internal gradient = f - g2")
            ax[4].axis("off")

            ax[5].imshow(g5, cmap="gray")
            ax[5].set_title("External gradient = g1 - f")
            ax[5].axis("off")

            plt.tight_layout()
            fig.savefig("Figure941.png", dpi=150, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure942(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter09 script `Figure942.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from PIL import Image
            from skimage.filters import threshold_otsu
            import ia870 as ia
            from helpers.libdipum.data_path import dip_data

            # %% Figure942
            # Rice grains statistics

            # %% Data
            f = np.array(Image.open(dip_data("rice-shaded.tif")))
            if f.ndim == 3:
                f = f[..., 0]

            # %% SE
            B = ia.iasedisk(40, "2D", "OCTAGON")

            # %% Top Hat
            g1 = ia.iaopen(f, B)
            g2 = ia.iasubm(f, g1)

            # %% Thresholding
            level = threshold_otsu(f)
            X = f > level

            level1 = threshold_otsu(g2)
            Y = g2 > level1

            # %% Noise cleaning
            Y1 = ia.iaclose(ia.iaopen(Y, ia.iasecross()), ia.iasecross())

            # %% Edge off
            Y2 = ia.iaedgeoff(Y, ia.iasecross())

            # %% Labeling
            Yl = ia.ialabel(Y2, ia.iasebox())

            # %% Statistics
            Area = ia.iablob(Yl, "area", "data")

            # %% Display figure 1
            fig1, ax = plt.subplots(
                2, 3, figsize=(12, 8), sharex=True, sharey=True, num=1
            )
            ax = ax.ravel()

            ax[0].imshow(f, cmap="gray")
            ax[0].set_title("f")
            ax[0].axis("off")

            ax[1].imshow(X, cmap="gray")
            ax[1].set_title(f"X threshold on f, Otsu = {int(round(level))}")
            ax[1].axis("off")

            ax[2].axis("off")

            ax[3].imshow(g1, cmap="gray")
            ax[3].set_title("g1 = opening with disk radius 40")
            ax[3].axis("off")

            ax[4].imshow(g2, cmap="gray")
            ax[4].set_title("g2 = f - g1")
            ax[4].axis("off")

            ax[5].imshow(Y, cmap="gray")
            ax[5].set_title(f"Y threshold on g2, Otsu = {int(round(level1))}")
            ax[5].axis("off")

            fig1.tight_layout()
            fig1.savefig("Figure942.png", dpi=150, bbox_inches="tight")

            # %% Display figure 2
            fig2, bx = plt.subplots(2, 2, figsize=(10, 8), num=2)
            bx = bx.ravel()

            bx[0].imshow(Y1, cmap="gray")
            bx[0].set_title("Cleaning by ASF")
            bx[0].axis("off")

            bx[1].imshow(Y2, cmap="gray")
            bx[1].set_title("Edge off")
            bx[1].axis("off")

            lbl_img = np.asarray(ia.iaglblshow(Yl))
            if lbl_img.ndim == 3 and lbl_img.shape[0] in (3, 4):
                # ia870 may return channel-first (C, H, W); convert to (H, W, C).
                lbl_img = np.transpose(lbl_img, (1, 2, 0))
            bx[2].imshow(lbl_img)
            bx[2].set_title(f"Labeling, number of objects = {int(np.max(Yl))}")
            bx[2].axis("off")

            area_vals = np.asarray(Area).ravel().astype(float)
            bx[3].hist(area_vals, bins=20)
            bx[3].set_title("Area distribution")

            fig2.tight_layout()
            fig2.savefig("Figure942Bis.png", dpi=150, bbox_inches="tight")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure943(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter09 script `Figure943.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from PIL import Image
            from scipy.ndimage import convolve
            import ia870 as ia
            from helpers.libdipum.data_path import dip_data

            # %% Figure943
            # Granulometry

            # %% Data
            f = np.array(Image.open(dip_data("wood-dowels.tif")))
            if f.ndim == 3:
                f = f[..., 0]

            # %% Smoothing (fspecial gaussian + imfilter equivalent)
            HSize = 5
            Sigma = 1
            ax = np.arange(-(HSize // 2), HSize // 2 + 1)
            xx, yy = np.meshgrid(ax, ax)
            h = np.exp(-(xx**2 + yy**2) / (2.0 * Sigma**2))
            h = h / np.sum(h)

            g = convolve(f.astype(float), h, mode="reflect")
            g = np.clip(np.round(g), 0, 255).astype(f.dtype)

            # %% Opening
            g1 = np.zeros((g.shape[0], g.shape[1], 35), dtype=g.dtype)
            LesArea = np.zeros(35, dtype=float)
            for cpt in range(1, 36):
                opened = ia.iaopen(g, ia.iasedisk(cpt))
                g1[:, :, cpt - 1] = opened
                LesArea[cpt - 1] = float(np.sum(opened))

            # %% Display figure 1
            LesRadii = [10, 20, 25, 30]
            fig1, axarr = plt.subplots(
                2, 3, figsize=(12, 8), sharex=True, sharey=True, num=1
            )
            try:
                fig1.canvas.manager.set_window_title("Figure 9.41")
            except Exception:
                pass

            ax = axarr.ravel()

            ax[0].imshow(f, cmap="gray")
            ax[0].set_title("f")
            ax[0].axis("off")

            ax[1].imshow(g, cmap="gray")
            ax[1].set_title(f"g = gaussian smooth, sigma={Sigma}, size={HSize}")
            ax[1].axis("off")

            for i, r in enumerate(LesRadii):
                ax[2 + i].imshow(g1[:, :, r - 1], cmap="gray")
                ax[2 + i].set_title(f"g{r} = opening with disk radius {r}")
                ax[2 + i].axis("off")

            fig1.tight_layout()
            fig1.savefig("Figure943.png", dpi=150, bbox_inches="tight")

            # %% Display figure 2
            fig2, bx = plt.subplots(1, 2, figsize=(12, 4), num=2)
            try:
                fig2.canvas.manager.set_window_title("Figure 9.41 bis")
            except Exception:
                pass

            bx[0].bar(np.arange(1, 36), LesArea)
            bx[0].set_xlabel("Radius")
            bx[0].set_title("Volume versus radius")
            bx[0].axis("tight")
            bx[0].set_box_aspect(1)

            dvol = -np.diff(LesArea)
            bx[1].bar(np.arange(1, len(dvol) + 1), dvol)
            bx[1].set_xlabel("Radius")
            bx[1].set_title("Negative derivative of volume")
            bx[1].axis("tight")
            bx[1].set_box_aspect(1)

            fig2.tight_layout()
            fig2.savefig("Figure943Bis.png", dpi=150, bbox_inches="tight")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure945(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter09 script `Figure945.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from PIL import Image
            from skimage.filters import threshold_otsu
            import ia870 as ia
            from helpers.libgeneral.mmshow import mmshow
            from helpers.libdipum.data_path import dip_data

            # %% Figure945

            # %% Data
            f = np.array(Image.open(dip_data("dark-blobs-on-light-background.tif")))
            if f.ndim == 3:
                f = f[..., 0]

            # %% Closing the dark patches
            g = ia.iaclose(f, ia.iasedisk(29, "2D", "OCTAGON"))

            # %% Closing the clear patches
            g1 = ia.iaopen(g, ia.iasedisk(59, "2D", "OCTAGON"))

            # %% Computing the boundary
            level_val = threshold_otsu(g1)
            level_norm = float(level_val) / 255.0
            X = g1 > level_val
            Y = ia.iagradm(X, ia.iasecross(1), ia.iasecross(0))
            Yd = ia.iadil(Y, ia.iasecross(2))

            # %% Display
            try:
                _raw = input("Enonce (1) ou corrige (2) : ").strip()
                Choix = int(_raw) if _raw else 1
            except Exception:
                Choix = 1

            fig1 = plt.figure(1, figsize=(10, 8))
            try:
                fig1.canvas.manager.set_window_title("Figure 9.45")
            except Exception:
                pass

            if Choix == 1:
                plt.subplot(1, 2, 1)
                plt.imshow(f, cmap="gray")
                plt.axis("off")

                plt.subplot(1, 2, 2)
                mmshow(f, Yd, Yd, Yd)
                plt.title("f with boundary Y")
                plt.axis("off")
            else:
                ax1 = plt.subplot(2, 2, 1)
                mmshow(f, Yd, Yd, Yd)
                plt.title("f with boundary Y")
                plt.axis("off")

                ax2 = plt.subplot(2, 2, 2)
                plt.imshow(g, cmap="gray", vmin=0, vmax=255)
                plt.title("g = closing with disk radius 29")
                plt.axis("off")

                ax3 = plt.subplot(2, 2, 3)
                plt.imshow(g1, cmap="gray", vmin=0, vmax=255)
                plt.title("g1 = opening with disk radius 59")
                plt.axis("off")

                ax4 = plt.subplot(2, 2, 4)
                plt.imshow(Yd, cmap="gray")
                plt.title(
                    f"Boundary of thresholded g1, Otsu={int(round(level_norm * 255))}"
                )
                plt.axis("off")

            plt.tight_layout()

            # Figure with histograms
            fig2 = plt.figure(2, figsize=(10, 8))

            bins = np.arange(0, 256)

            def _hist255(img: Any):
                """_hist255."""
                c, _ = np.histogram(
                    np.asarray(img).ravel(), bins=np.arange(-0.5, 256.5, 1.0)
                )
                return c

            plt.subplot(2, 2, 1)
            plt.bar(bins, _hist255(f))
            plt.title("hist(f)")

            plt.subplot(2, 2, 2)
            plt.bar(bins, _hist255(g))
            plt.title("hist(g)")

            plt.subplot(2, 2, 3)
            plt.bar(bins, _hist255(g1))
            plt.title("hist(g1)")

            plt.tight_layout()

            # %% Print
            if Choix == 1:
                fig1.savefig("Figure945Enonce.png", dpi=150, bbox_inches="tight")
            else:
                fig1.savefig("Figure945.png", dpi=150, bbox_inches="tight")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure946(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter09 script `Figure946.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from PIL import Image
            from skimage.filters import threshold_otsu
            import ia870 as ia
            from helpers.libgeneral.AddSE2Image import AddSE2Image
            from helpers.libdipum.data_path import dip_data

            # %% Figure946
            # Segment calculator symbols

            # %% Parameters
            Bc4 = ia.iasecross()

            # %% Data
            Mask = np.array(Image.open(dip_data("calculator.tif")))
            if Mask.ndim == 3:
                Mask = Mask[..., 0]

            # %% Thresholding
            Level = threshold_otsu(Mask)
            X = Mask > Level

            # %% Structuring elements
            B1 = ia.iaseline(71, 0)
            B2 = ia.iaseline(11, 0)
            B3 = ia.iaseline(21, 0)

            # %% Removing clear horizontal objects (Morphological)
            Marker = ia.iaero(Mask, B1)
            g1 = ia.iadil(Marker, B1)
            OK = np.all(g1 == ia.iaopen(Mask, B1))
            print(f"OK = {OK}")

            # %% Removing clear horizontal objects (Geodesical)
            g2 = ia.iainfrec(Marker, Mask, Bc4)
            Mask1 = ia.iasubm(Mask, g2)

            # %% Removing clear vertical objects (Geodesical)
            Marker1 = ia.iaero(Mask1, B2)
            g4 = ia.iainfrec(Marker1, Mask1, Bc4)
            g5 = ia.iadil(g4, B3)
            Marker2 = ia.iaintersec(Mask1, g5)
            g6 = ia.iainfrec(Marker2, Mask1, Bc4)

            # %% Final thresholding
            Level1 = threshold_otsu(g6)
            X1 = g6 > Level1

            # %% Display figure 1
            fig1, ax = plt.subplots(
                2, 3, figsize=(12, 8), sharex=True, sharey=True, num=1
            )
            try:
                fig1.canvas.manager.set_window_title("Figure 9.46")
            except Exception:
                pass

            ax = ax.ravel()
            ax[0].imshow(Mask, cmap="gray")
            ax[0].set_title("Mask")
            ax[0].axis("off")

            ax[1].imshow(X, cmap="gray")
            ax[1].set_title(f"X threshold on Mask, Otsu={int(round(Level))}")
            ax[1].axis("off")

            ax[2].imshow(AddSE2Image(Marker, B1, int(np.max(Marker))), cmap="gray")
            ax[2].set_title("Marker = erosion with line 71")
            ax[2].axis("off")

            ax[3].imshow(g1, cmap="gray")
            ax[3].set_title("g1 = opening with line 71")
            ax[3].axis("off")

            ax[4].imshow(g2, cmap="gray")
            ax[4].set_title("g2 = reconstruction from Marker in Mask")
            ax[4].axis("off")

            ax[5].imshow(Mask1, cmap="gray")
            ax[5].set_title("Mask1 = Mask minus g2")
            ax[5].axis("off")

            fig1.tight_layout()
            fig1.savefig("Figure946.png", dpi=150, bbox_inches="tight")

            # %% Display figure 2
            fig2, bx = plt.subplots(
                2, 3, figsize=(12, 8), sharex=True, sharey=True, num=2
            )
            try:
                fig2.canvas.manager.set_window_title("Figure 9.46 bis")
            except Exception:
                pass

            bx = bx.ravel()
            bx[0].imshow(Mask1, cmap="gray")
            bx[0].set_title("Mask1 = Mask minus g2")
            bx[0].axis("off")

            bx[1].imshow(Marker1, cmap="gray")
            bx[1].set_title("Marker1 = erosion with line 11")
            bx[1].axis("off")

            bx[2].imshow(g4, cmap="gray")
            bx[2].set_title("g4 = reconstruction from Marker1 in Mask1")
            bx[2].axis("off")

            bx[3].imshow(g5, cmap="gray")
            bx[3].set_title("g5 = dilation of g4 with line 21")
            bx[3].axis("off")

            bx[4].imshow(Marker2, cmap="gray")
            bx[4].set_title("Marker2 = Mask1 and g5")
            bx[4].axis("off")

            bx[5].imshow(g6, cmap="gray")
            bx[5].set_title("g6 = reconstruction from Marker2 in Mask1")
            bx[5].axis("off")

            fig2.tight_layout()
            fig2.savefig("Figure946Bis.png", dpi=150, bbox_inches="tight")

            # %% Display figure 3
            fig3, cx = plt.subplots(
                2, 2, figsize=(10, 8), sharex=True, sharey=True, num=3
            )
            try:
                fig3.canvas.manager.set_window_title("Figure 9.46 ter")
            except Exception:
                pass

            cx = cx.ravel()

            cx[0].imshow(Mask, cmap="gray")
            cx[0].set_title("Mask")
            cx[0].axis("off")

            cx[1].imshow(X, cmap="gray")
            cx[1].set_title(f"X threshold on Mask, Otsu={int(round(Level))}")
            cx[1].axis("off")

            cx[2].imshow(X1, cmap="gray")
            cx[2].set_title(f"X1 threshold on g6, Otsu={int(round(Level1))}")
            cx[2].axis("off")

            cx[3].axis("off")

            fig3.tight_layout()
            fig3.savefig("Figure946Ter.png", dpi=150, bbox_inches="tight")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure95(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter09 script `Figure95.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            import PIL.Image
            import ia870 as ia
            from helpers.libdipum.data_path import dip_data

            # %% Figure95

            # %% Data
            # Keep PIL reader: this TIFF can be interpreted inverted by some loaders.
            f = np.array(PIL.Image.open(dip_data("wirebond-mask.tif")))

            # %% SE
            B1 = ia.iasebox(5)
            B2 = ia.iasebox(7)
            B3 = ia.iasebox(22)

            # %% Erosion
            f1 = ia.iaero(f, B1)
            f2 = ia.iaero(f, B2)
            f3 = ia.iaero(f, B3)

            # %% Display
            fig = plt.figure(num=1, figsize=(10, 10))
            try:
                fig.canvas.manager.set_window_title("Figure 9.5")
            except Exception:
                pass

            plt.subplot(2, 2, 1)
            plt.imshow(f, cmap="gray")
            plt.title("f")
            plt.axis("off")

            plt.subplot(2, 2, 2)
            plt.imshow(f1, cmap="gray")
            plt.title(r"f1 = $\epsilon_{B1}(f)$")
            plt.axis("off")

            plt.subplot(2, 2, 3)
            plt.imshow(f2, cmap="gray")
            plt.title(r"f2 = $\epsilon_{B2}(f)$")
            plt.axis("off")

            plt.subplot(2, 2, 4)
            plt.imshow(f3, cmap="gray")
            plt.title(r"f3 = $\epsilon_{B3}(f)$")
            plt.axis("off")

            plt.tight_layout()
            fig.savefig("Figure95.png", dpi=150, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure97(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter09 script `Figure97.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            import ia870 as ia
            from helpers.libdipum.data_path import dip_data

            # %% Figure97

            # %% Data
            # historical filename in MATLAB source: Fig0907(a)(text_gaps_1_and_2_pixels).tif
            f = imread(dip_data("text-broken.tif"))
            if f.ndim == 3:
                f = f[..., 0]

            # %% SE
            B1 = ia.iasecross(1)

            # %% Dilation
            f1 = ia.iadil(f, B1)

            # %% Display
            fig, ax = plt.subplots(
                1, 2, figsize=(10, 5), sharex=True, sharey=True, num=1
            )
            try:
                fig.canvas.manager.set_window_title("Figure 9.7")
            except Exception:
                pass

            ax[0].imshow(f, cmap="gray")
            ax[0].set_title("f")
            ax[0].axis("off")

            ax[1].imshow(f1, cmap="gray")
            ax[1].set_title(r"f1 = $\delta_{B1}(f)$")
            ax[1].axis("off")

            plt.tight_layout()
            fig.savefig("Figure97.png", dpi=150, bbox_inches="tight")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

