from pathlib import Path as _Path
import os as _os

from typing import Any


class Chapter12Mixin:
    def equation12_4(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Equation12_4.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.measure import find_contours
            from scipy.spatial.distance import pdist, squareform
            from helpers.libdipum.diameter import diameter
            from helpers.libdipum.x2majoraxis import x2majoraxis
            from helpers.libdipum.data_path import dip_data
            import os
            import itertools

            def Equation12_4():
                """Run Equation 12.4 demonstration and visualize diameter results."""
                # Equation12_4.m

                # Paths
                # legbone.tif location check
                paths = [dip_data("legbone.tif"), "legbone.tif"]
                img_path = None
                for p in paths:
                    if os.path.exists(p):
                        img_path = p
                        break

                if img_path is None:
                    print("Error: legbone.tif not found.")
                    return

                print(f"Reading image from {img_path}...")
                X = imread(img_path)
                # Ensure boolean
                if X.dtype != bool:
                    X = X > 0

                # Diameter
                # diameter returns a list of results. We assume 1 region.
                print("Computing Diameter (efficient)...")
                S_list = diameter(X)
                if not S_list:
                    print("No regions found.")
                    return

                S = S_list[0]

                # x2majoraxis (For demo, though not plotted in subplot 1/2)
                # [C, Theta] = x2majoraxis (S.MajorAxis, X);
                # My python diameter returns MajorAxis as [[r1, c1], [r2, c2]]
                # x2majoraxis expects A, B.
                print("Aligning to Major Axis...")
                C, Theta = x2majoraxis(S.MajorAxis, X)

                # Diameter, brute force ...
                print("Computing Diameter (brute force)...")
                # B = bwboundaries (X);
                contours = find_contours(X, 0.5)
                # contours returns float coordinates. Using pixels [row, col]
                # We want integer pixel coordinates of the boundary?
                # bwboundaries finds the set of pixels on the boundary.
                # find_contours finds standard marching squares contours.
                # For accurate pixel-to-pixel distance, we should get boundary pixels.
                from skimage.segmentation import find_boundaries

                boundary_mask = find_boundaries(X, mode="inner")
                pts_r, pts_c = np.where(boundary_mask)
                B = np.column_stack((pts_r, pts_c))

                # Brute force on boundary pixels
                # MATLAB: Couples = nchoosek (1:N, 2); D = hypot(...);
                # We use pdist.
                dists = pdist(B, "euclidean")
                DMax = np.max(dists)

                # Find indices
                dist_mat = squareform(dists)  # Can be large
                idx_flat = np.argmax(dist_mat)
                idx_i, idx_j = np.unravel_index(idx_flat, dist_mat.shape)

                BestFrom = B[idx_i]
                BestTo = B[idx_j]

                # Check consistency
                print(f"Diameter (Efficient): {S.Diameter:.4f}")
                print(f"Diameter (Brute Force): {DMax:.4f}")

                # Display
                fig = plt.figure(figsize=(12, 6))

                # Subplot 1: Brute Force
                ax1 = fig.add_subplot(1, 2, 1)
                ax1.imshow(X, cmap="gray")
                ax1.plot(BestFrom[1], BestFrom[0], "or", markersize=8)  # Col, Row
                ax1.plot(BestTo[1], BestTo[0], "or", markersize=8)
                # Green face color?
                ax1.plot(BestFrom[1], BestFrom[0], "og", markersize=4)
                ax1.plot(BestTo[1], BestTo[0], "og", markersize=4)
                ax1.set_title(
                    f"Brute force, N={len(B) * (len(B) - 1) // 2}, D={DMax:.2f}"
                )

                # Subplot 2: Calculated
                ax2 = fig.add_subplot(1, 2, 2)
                ax2.imshow(X, cmap="gray")

                # Major Axis (Red)
                # S.MajorAxis is [[r1 c1], [r2 c2]]
                # Plot line: plot([c1, c2], [r1, r2])
                maj = S.MajorAxis
                ax2.plot(
                    [maj[0, 1], maj[1, 1]], [maj[0, 0], maj[1, 0]], "r-", linewidth=2
                )

                # Minor Axis (Green)
                mino = S.MinorAxis
                ax2.plot(
                    [mino[0, 1], mino[1, 1]],
                    [mino[0, 0], mino[1, 0]],
                    "g-",
                    linewidth=2,
                )

                # Basic Rectangle (Blue)
                # S.BasicRectangle: [[r1 c1], [r2 c2], [r3 c3], [r4 c4]]
                rect = S.BasicRectangle
                # Close the loop
                rect_plot = np.vstack((rect, rect[0]))
                ax2.plot(rect_plot[:, 1], rect_plot[:, 0], "b-", linewidth=1)

                ax2.set_title(f"X, Maj(r), Min(g), Rect(b), D={S.Diameter:.2f}")

                plt.tight_layout()
                plt.savefig("Equation12_4.png")
                plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def example1214(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Example1214.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            import sys
            from helpers.libdip.principalComponents4e import principalComponents4e

            def get_choice():
                """get_choice."""
                print("Example from the book (1), Larger example (2) : ")
                try:
                    val = input()
                    return int(val)
                except:
                    return 1

            def eigsort(C: Any):
                """eigsort."""
                # Sort eigenvalues/vectors descending
                vals, vecs = np.linalg.eig(C)
                idx = np.argsort(vals)[::-1]
                D = np.diag(vals[idx])
                V = vecs[:, idx]
                return V, D

            def Example1114(Choix: Any = None):
                """Example1114."""
                if Choix is None:
                    Choix = get_choice()

                print(f"Running Example1114 (Choix={Choix})...")

                # Data
                if Choix == 1:
                    N = 4
                    X = np.array(
                        [[0, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
                    )

                elif Choix == 2:
                    N = 200
                    t = np.linspace(0, 1, N)
                    f0 = 3
                    Sigma = 0.4
                    f1 = np.sin(2 * np.pi * f0 * t)
                    f2 = np.cos(2 * np.pi * f0 * t)

                    # Noise
                    # randn(size(t))
                    r1 = Sigma * np.random.randn(N)
                    r2 = Sigma * np.random.randn(N)
                    r3 = Sigma * np.random.randn(N)

                    X = np.vstack([f1 + r1, f2 + r2, f1 + f2 + r3])

                else:
                    print("Invalid choice.")
                    return

                # Mean and Covariance (Normalized by N)
                # X rows are variables.
                mx = np.mean(X, axis=1).reshape(-1, 1)

                # Cov
                # np.cov default is ddof=1 (N-1).
                # We need ddof=0 (N).
                CX = np.cov(X, ddof=0)

                print("Mean mx:")
                print(mx)
                print("Covariance CX (Norm by N):")
                print(CX)

                # Eigenvalues
                V, D = eigsort(CX)
                print("Eigenvalues D:")
                print(np.diag(D))

                # Hotelling Transform
                A = V.T  # Eigenvectors in rows

                # Center X
                # X - mx
                X_centered = X - mx

                # Y
                Y = A @ X_centered

                if Choix == 1:
                    print("Y:")
                    print(Y)

                my = np.mean(Y, axis=1).reshape(-1, 1)
                CY = np.cov(Y, ddof=0)
                print("Mean Y:")
                print(my)  # Should be 0
                print("Cov Y:")
                print(CY)  # Should be diagonal D

                # Complete Reconstruction
                # XHat = A' * Y + mx
                XHatComplete = A.T @ Y + mx
                EComplete = X - XHatComplete

                if Choix == 1:
                    print("XHatComplete:")
                    print(XHatComplete)
                    print("EComplete:")
                    print(EComplete)

                MSEComplete = np.mean(
                    EComplete**2, axis=1
                )  # Mean over samples (columns)
                MSECompleteSum = np.sum(MSEComplete)
                print(f"MSECompleteSum: {MSECompleteSum}")

                # Partial Reconstruction (Keep 2 components)
                A2 = A[:2, :]  # First 2 rows
                Y2 = A2 @ X_centered

                XHat2 = A2.T @ Y2 + mx
                E2 = X - XHat2

                if Choix == 1:
                    print("Y2:")
                    print(Y2)
                    print("XHat2:")
                    print(XHat2)
                    print("E2:")
                    print(E2)

                mE = np.mean(E2, axis=1)
                CE = np.cov(E2, ddof=0)
                MSE2 = np.mean(E2**2, axis=1)
                MSE2Sum = np.sum(MSE2)
                print(f"MSE2Sum: {MSE2Sum}")

                # Use of principalComponents (Norm by N-1)
                # X' is passed. Rows=Samples.
                print("Running principalComponents4e (Norm by N-1)...")
                try:
                    P = principalComponents4e(X.T, 2)
                    print(f"Result type: {type(P)}")
                    print(f"Result keys/dir: {dir(P)}")
                    if hasattr(P, "ems"):
                        print("P.ems:", P.ems)
                    else:
                        print("Error: P object has no 'ems' attribute.")
                        if isinstance(P, dict):
                            print("P is a dict. Keys:", P.keys())
                except Exception as e:
                    print(f"Error calling principalComponents4e: {e}")
                    import traceback

                    traceback.print_exc()
                    return

                # Display (Choix 2)
                if Choix == 2:
                    fig = plt.figure(figsize=(10, 8))

                    # 1. X'
                    plt.subplot(2, 2, 1)
                    plt.plot(X.T)
                    plt.title(f"X, lambda={np.diag(D)}")
                    plt.grid(True)
                    # Store axis limits?

                    # 2. Y2'
                    plt.subplot(2, 2, 2)
                    plt.plot(Y2.T)
                    plt.title("Y2")
                    plt.grid(True)

                    # 3. Xrec'
                    plt.subplot(2, 2, 3)
                    plt.plot(XHat2.T)
                    plt.title("X_rec")
                    plt.grid(True)

                    # 4. Error
                    plt.subplot(2, 2, 4)
                    plt.plot(E2.T)
                    plt.title(f"Err, MSE={MSE2}, Sum={MSE2Sum:.4f}")
                    plt.grid(True)

                    plt.tight_layout()
                    plt.savefig(f"Example1114_Choix{Choix}.png")
                    plt.show()

            if True:
                if len(sys.argv) > 1:
                    try:
                        Example1114(int(sys.argv[1]))
                    except:
                        Example1114()
                else:
                    Example1114()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1201(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1201.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.color import rgb2gray
            from helpers.libgeneral.N8 import N8
            from helpers.libgeneral.mmshow import mmshow
            from helpers.libdipum.freemanChainCode import freemanChainCode
            import sys
            from helpers.libdipum.data_path import dip_data

            def get_choice():
                """get_choice."""
                print("Select image to process:")
                print("1: Fig12.01")
                print("2: Fig12.2")
                print("3: OtherToyExample")
                print("4: Exam")
                print("5: Apple (apple12.jpg)")
                print("6: Fig12.01 Modif")

                try:
                    val = input("Enter choice (1-6): ")
                    return int(val)
                except ValueError:
                    return 1  # Default

            def Figure1201(Choix: Any = None):
                """Figure1201."""
                if Choix is None:
                    Choix = get_choice()

                print(f"Running Figure1201 with Choix={Choix}...")

                # Data Selection
                if Choix == 1:
                    # Fig12.01
                    X = np.zeros((7, 7), dtype=bool)
                    # MATLAB: X(2, 3:6)=1; X(3, 2:5)=1; X(4, 3:5)=1; X(5, 2:5)=1; X(6, 2:5)=1;
                    X[1, 2:6] = True
                    X[2, 1:5] = True
                    X[3, 2:5] = True
                    X[4, 1:5] = True
                    X[5, 1:5] = True

                elif Choix == 2:
                    # Fig12.2
                    X = np.zeros((7, 7), dtype=bool)
                    # X(2, 4)=1
                    X[1, 3] = True
                    # X(3, 3)=1; X(3, 5)=1
                    X[2, 2] = True
                    X[2, 4] = True
                    # X(4, 3)=1
                    X[3, 2] = True
                    # X(5, 2)=1; X(5, 4)=1
                    X[4, 1] = True
                    X[4, 3] = True
                    # X(6, 2:4)=1
                    X[5, 1:4] = True

                elif Choix == 3:
                    # OtherToyExample
                    X = np.zeros((7, 17), dtype=bool)
                    # X(2, 3:15)=1
                    X[1, 2:15] = True
                    # X(3:6, 3:7)=1 -> Rows 3 to 6 (inclusive), Cols 3 to 7 (inclusive)
                    # python: rows 2:6, cols 2:7
                    X[2:6, 2:7] = True

                elif Choix == 4:
                    # Exam
                    X = np.zeros((5, 5), dtype=bool)
                    # X(2, 3:4)=1
                    X[1, 2:4] = True
                    # X(3, 2:4)=1
                    X[2, 1:4] = True
                    # X(4, 2:3)=1
                    X[3, 1:3] = True

                elif Choix == 5:
                    # Apple
                    path = dip_data("apple12.jpg")
                    img = imread(path)
                    if img.ndim == 3:
                        gray = rgb2gray(img)
                    else:
                        gray = img

                    # Threshold
                    X = gray > 0.5
                    # Check corners to ensure object is white on black
                    if X[0, 0]:
                        X = ~X

                elif Choix == 6:
                    # Fig12.01 Modif
                    X = np.zeros((8, 7), dtype=bool)
                    # X(2, 3:6)=1; X(3, 2:5)=1; X(4, 3:5)=1; X(5, 2:5)=1; X(6, 2:5)=1; X(7, 6)=1
                    X[1, 2:6] = True
                    X[2, 1:5] = True
                    X[3, 2:5] = True
                    X[4, 1:5] = True
                    X[5, 1:5] = True
                    X[6, 5] = True  # X(7, 6) -> row 6, col 5

                else:
                    print("Invalid choice.")
                    return

                H, W = X.shape

                # 2. Get Uppermost Leftmost point b0
                rows_nz, cols_nz = np.where(X)
                if len(rows_nz) == 0:
                    print("Empty image.")
                    return

                r0 = rows_nz[0]
                c0_idx = cols_nz[0]
                b0 = r0 * W + c0_idx  # Row-Major index

                # West Neighbor c0
                neighbor_c0 = b0 - 1

                # 3. Get b(1) and c(1)
                N = N8(b0, neighbor_c0, W)

                found_start = False
                for i, idx in enumerate(N):
                    if 0 <= idx < H * W:
                        if X.flat[idx]:
                            b1 = idx
                            c1 = N[i - 1]
                            found_start = True
                            break

                if not found_start:
                    print(f"Isolated point at ({r0}, {c0_idx})?")
                    # Just plot X
                    plt.imshow(X)
                    plt.show()
                    return

                # 4. Main Loop
                b_list = [b1]
                c_list = [c1]
                curr_b = b1
                curr_c = c1

                Yn = np.zeros_like(X, dtype=float)
                k = 1
                Yn.flat[curr_b] = k

                max_iter = H * W * 10

                while True:
                    N_curr = N8(curr_b, curr_c, W)
                    found_next = False
                    for i, idx in enumerate(N_curr):
                        if 0 <= idx < H * W and X.flat[idx]:
                            next_b = idx
                            next_c = N_curr[i - 1]
                            found_next = True
                            break

                    if not found_next:
                        break

                    k += 1
                    curr_b = next_b
                    curr_c = next_c

                    b_list.append(curr_b)
                    c_list.append(curr_c)
                    Yn.flat[curr_b] = k

                    if curr_b == b_list[0] and b_list[-2] == b0 and k >= 3:
                        break

                    if k > max_iter:
                        print("Runaway loop.")
                        break

                final_b = b_list[:-1]

                # 5. Contour Map Y
                Y = np.zeros_like(X, dtype=bool)
                Y.flat[final_b] = True

                # 6. Freeman Code
                rows = [idx // W for idx in final_b]
                cols = [idx % W for idx in final_b]
                B_coords = np.column_stack((rows, cols))

                res = freemanChainCode(B_coords, 8)
                print("FCC:", res.fcc)
                print("MinMag:", res.mm)
                print("Diff:", res.diff)

                # 7. Display
                B0Matrix = np.zeros_like(X, dtype=bool)
                C0Matrix = np.zeros_like(X, dtype=bool)
                B0Matrix.flat[b0] = True
                C0Matrix.flat[neighbor_c0] = True

                fig = plt.figure(figsize=(10, 10))

                plt.subplot(2, 2, 1)
                plt.imshow(X, cmap="gray")
                plt.title(f"X (Choix={Choix})")

                plt.subplot(2, 2, 2)
                mmshow(Y, B0Matrix, C0Matrix)
                plt.title("Boundary(Y), b0(Red), c0(Green)")

                plt.subplot(2, 2, 3)
                masked_Yn = np.ma.masked_where(Yn == 0, Yn)
                plt.imshow(X, cmap="gray", alpha=0.3)
                plt.imshow(masked_Yn, cmap="jet", interpolation="nearest")
                plt.colorbar(label="Order k")
                plt.title("Ordered Boundary")

                plt.subplot(2, 2, 4)
                plt.stem(final_b)
                plt.title("Linear Index Sequence")
                plt.xlabel("k")

                plt.tight_layout()

                # Filename
                filename = f"Figure1201_Choix{Choix}.png"
                if Choix == 1:
                    filename = "Figure1201.png"
                elif Choix == 2:
                    filename = "Figure1202.png"

                plt.savefig(filename)
                print(f"Saved plot to {filename}")
                plt.show()

            if True:
                # Check args
                if len(sys.argv) > 1:
                    try:
                        val = int(sys.argv[1])
                        Figure1201(val)
                    except:
                        Figure1201()
                else:
                    Figure1201()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1205(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1205.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from skimage.color import rgb2gray
            from skimage.filters import threshold_otsu
            import ia870
            from scipy.ndimage import uniform_filter

            from helpers.libdipum.bwboundaries import bwboundaries
            from helpers.libdipum.freemanChainCode import freemanChainCode
            from helpers.libdipum.bsubsamp import bsubsamp
            from helpers.libdipum.connectpoly import connectpoly
            from helpers.libdipum.data_path import dip_data

            print("Running Figure1205...")

            # 1. Data
            path = dip_data("noisy-stroke.tif")
            f = imread(path)
            if f.ndim == 3:
                f = rgb2gray(f)

            # 2. Denoising
            f1 = uniform_filter(f.astype(float), size=9, mode="reflect")
            f1 = np.clip(f1, 0, 255).astype(np.uint8)

            # 3. Thresholding
            thresh = threshold_otsu(f1)
            bw = f1 > thresh

            # Area opening
            X1 = ia870.iaareaopen(bw, 100, ia870.iasebox())

            # 5. Boundaries
            # contours = find_contours(X1, 0.5)
            contours = bwboundaries(X1, 8)

            if len(contours) == 0:
                print("No contours found.")

            # Longest boundary
            longest_idx = np.argmax([len(c) for c in contours])
            B = contours[longest_idx]

            print(f"Longest boundary length: {len(B)}")

            # 6. Boundary subsampling
            r = 50
            B1 = bsubsamp(B, 1 / r, f.shape[0], f.shape[1])

            # Scale back
            B1 = B1 * r

            # 7. Connected polygon
            B1C = connectpoly(B1[:, 0], B1[:, 1])

            print(f"Connected polygon length: {len(B1C)}")

            # 8. Freeman Chain Code
            res = freemanChainCode(B1C, 8)
            print("FCC computed. Length:", len(res.fcc))

            # 9. Display
            fig, axes = plt.subplots(2, 3, figsize=(14, 9))
            ax = axes.ravel()

            # f
            ax[0].imshow(f, cmap="gray")
            ax[0].set_title(f"f, Size={f.shape}")

            # f1
            ax[1].imshow(f1, cmap="gray")
            ax[1].set_title("f1 (Average 9x9)")

            # X1
            ax[2].imshow(X1, cmap="gray")
            ax[2].set_title(f"X1 (Thresh={thresh:.1f}, Lambda=100)")

            # N (original boundary, previously B)
            ax[3].plot(B[:, 1], -B[:, 0], "r")
            ax[3].axis("equal")
            ax[3].set_title("N")

            # B1 (subsampled boundary)
            ax[4].plot(B1[:, 1], -B1[:, 0], "o", color="green", fillstyle="none")
            ax[4].axis("equal")
            ax[4].set_title(f"B1 (r={r})")

            # B1C rendered as zero-order hold (horizontal/vertical only) between knots.
            if len(B1) > 1:
                step_pts = [B1[0]]
                for p, q in zip(B1[:-1], B1[1:]):
                    # Horizontal then vertical (Manhattan link)
                    step_pts.append([p[0], q[1]])
                    step_pts.append([q[0], q[1]])
                step_pts = np.asarray(step_pts)
            else:
                step_pts = B1

            ax[5].plot(step_pts[:, 1], -step_pts[:, 0], "b", linewidth=1.0)
            ax[5].plot(B1[:, 1], -B1[:, 0], "o", color="black", markersize=3)
            ax[5].axis("equal")
            ax[5].set_title("B1C")

            plt.tight_layout()
            plt.savefig("Figure1205.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1209_copy(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1209 copy.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import os

            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from im2minperpoly import im2minperpoly
            from connectpoly import connectpoly
            from bound2im import bound2im
            from skimage.morphology import dilation, square
            from bwboundaries import bwboundaries
            from helpers.libdipum.data_path import dip_data

            def Figure1209():
                """Figure1209."""
                print("Running Figure1209 (Minimum Perimeter Polygon)...")

                # 1. Data
                path = dip_data("mapleleaf.tif")
                if not os.path.exists(path):
                    # Alternatives
                    alts = [dip_data("mapleleaf.tif"), dip_data("mapleleaf.tif")]
                    for a in alts:
                        if os.path.exists(a):
                            path = a
                            break

                if not os.path.exists(path):
                    print("mapleleaf.tif not found.")
                    return

                B = imread(path)
                # B should be binary?
                # Ensure binary
                if B.ndim == 3:
                    B = B[:, :, 0]
                B = B > 0  # Assume >0 is object

                LesCellSize = [4, 6, 8, 16, 32]
                M, N = B.shape

                # Boundaries
                boundaries = bwboundaries(B, conn=8)
                if boundaries:
                    b = boundaries[0]
                    # b is [(r,c)...]. bwboundaries returns start=end?
                    # My impl appends start at end?
                    # trace_boundary logic usually repeats start.
                    # MATLAB bwboundaries also repeats start.
                else:
                    b = np.array([])

                # bound2im with autoscale logic?
                # Logic in MATLAB: bIm = bound2im (b, M, N);
                # My bound2im creates empty image size M,N and puts b.
                bIm = bound2im(b, M, N)

                B2_list = []
                LesX_list = []

                for cellsize in LesCellSize:
                    print(f"Processing Cell Size {cellsize}...")
                    X, Y, R = im2minperpoly(B, cellsize)
                    LesX_list.append(len(X))

                    # ConnectPoly
                    if len(X) > 0:
                        b2 = connectpoly(X, Y)
                        B2 = bound2im(b2, M, N)
                    else:
                        B2 = np.zeros_like(B)

                    B2_list.append(B2)

                # Display
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                ax = axes.ravel()

                # 1. Original
                ax[0].imshow(B, cmap="gray")
                ax[0].set_title(f"X, size = {B.shape}")

                # 2. Boundary
                ax[1].imshow(bIm, cmap="gray")
                ax[1].set_title(f"8 conn., N_Ver = {len(b)}")

                # Loops
                SE = square(4)  # mmsecross(4) ~ approx square? Or diamond?
                # mmsecross is plus shape.
                # Use cross structuring element.
                se_cross = np.array(
                    [
                        [0, 1, 0, 0],
                        [0, 1, 0, 0],
                        [1, 1, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0],
                    ],
                    dtype=bool,
                )  # 4x4?
                # mmsecross(R): usually radius R.
                # Let's use simple square(4) for visibility.

                for i, cellsize in enumerate(LesCellSize):
                    idx = i + 2
                    if idx < 8:
                        B2 = B2_list[i]
                        # Dilate for visibility
                        B2_dil = dilation(B2, SE)

                        # mmshow(B, green=B2_dil, blue=B2_dil)
                        # mmshow implementation only supports one mask currently?
                        # My mmshow supports (f, mask, color).
                        # MATLAB mmshow(f, mask1, mask2) overlays both.
                        # I can overlay manually.

                        # Plot
                        plt.sca(ax[idx])
                        # Base
                        ax[idx].imshow(B, cmap="gray")
                        # Overlay
                        # Using alpha blending or contour?
                        # My mmshow overlays mask.
                        # Let's just overlay B2_dil in Red.

                        # Create RGB
                        rgb = np.dstack((B, B, B)).astype(float)
                        # If B is bool, 0/1.

                        # Overlay
                        # Red channel for mask
                        mask = B2_dil > 0
                        if mask.any():
                            rgb[mask] = [1, 0, 0]  # Red

                        ax[idx].imshow(rgb)
                        ax[idx].set_title(f"CS={cellsize}, N_Ver={LesX_list[i]}")
                        ax[idx].axis("off")

                plt.tight_layout()
                plt.savefig("Figure1209.png")
                plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1209(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1209.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.libgeneral.im2minperpoly import im2minperpoly
            from helpers.libdipum.connectpoly import connectpoly
            from helpers.libdipum.bound2im import bound2im
            from helpers.libdipum.bwboundaries import bwboundaries
            from helpers.libdipum.data_path import dip_data

            print("Running Figure1209 (Minimum Perimeter Polygon)...")

            # 1. Data
            path = dip_data("mapleleaf.tif")
            B = imread(path)
            if B.ndim == 3:
                B = B[:, :, 0]
            B = B > 0  # Assume >0 is object

            LesCellSize = [2, 4, 6, 8, 16, 32]
            M, N = B.shape

            # Boundaries
            boundaries = bwboundaries(B, conn=8)
            if boundaries:
                b = boundaries[0]
            else:
                b = np.array([])

            bIm = bound2im(b, M, N)

            B2_list = []
            B2_coords = []
            LesX_list = []

            for cellsize in LesCellSize:
                print(f"Processing Cell Size {cellsize}...")
                X, Y, R = im2minperpoly(B, cellsize)
                LesX_list.append(len(X))

                # ConnectPoly
                if len(X) > 0:
                    b2 = connectpoly(X, Y)
                    B2 = bound2im(b2, M, N)
                    B2_coords.append(b2)
                else:
                    B2 = np.zeros_like(B)
                    B2_coords.append(np.zeros((0, 2), dtype=int))

                B2_list.append(B2)

            # Display
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            ax = axes.ravel()

            # 1. Original
            ax[0].imshow(B, cmap="gray")
            ax[0].set_title(f"X, size = {B.shape}")
            ax[0].axis("off")

            # 2. Boundary (negative display with dotted boundary points)
            ax[1].imshow(np.ones_like(B), cmap="gray", vmin=0, vmax=1)
            if len(b) > 0:
                ax[1].plot(b[:, 1], b[:, 0], "k.", markersize=1.0)
            ax[1].set_title(f"8 conn., N_Ver = {len(b)}")
            ax[1].set_aspect("equal")
            ax[1].axis("off")

            for i, cellsize in enumerate(LesCellSize):
                idx = i + 2
                if idx < 8:
                    pts = B2_coords[i]
                    # Overlay approximated contour on original image.
                    ax[idx].imshow(B, cmap="gray")
                    if len(pts) > 0:
                        ax[idx].plot(pts[:, 1], pts[:, 0], "r.", markersize=1.1)
                    ax[idx].set_title(f"CS={cellsize}, N_Ver={LesX_list[i]}")
                    ax[idx].set_aspect("equal")
                    ax[idx].axis("off")

            plt.tight_layout()
            plt.savefig("Figure1209.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1211(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1211.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.libdipum.bwboundaries import bwboundaries
            from helpers.libdipum.signature import signature
            from helpers.libdipum.data_path import dip_data

            print("Running Figure1211 (Boundary Signatures)...")

            # 1. Load Data
            # Paths based on previous search
            path_square = dip_data("binary-square-distorted.tif")
            path_triangle = dip_data("binary-triangle-distorted.tif")
            f1 = imread(path_square)
            f2 = imread(path_triangle)

            # 2. Boundaries
            B1_list = bwboundaries(f1, conn=8)
            B2_list = bwboundaries(f2, conn=8)

            if len(B1_list) == 0 or len(B2_list) == 0:
                print("Error: No boundaries found.")

            B1 = B1_list[0]
            B2 = B2_list[0]

            # 3. Signatures
            dist1, angle1 = signature(B1)
            dist2, angle2 = signature(B2)

            # 4. Display
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes[0, 0].imshow(f1, cmap="gray")
            axes[0, 0].set_title("B1 (Square Distorted)")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(f2, cmap="gray")
            axes[0, 1].set_title("B2 (Triangle Distorted)")
            axes[0, 1].axis("off")

            # Signatures
            axes[1, 0].plot(angle1, dist1, "k-")
            axes[1, 0].set_title(r"Signature $S_1(\theta)$")
            axes[1, 0].set_xlabel("Angle (degrees)")
            axes[1, 0].set_ylabel("Distance")
            axes[1, 0].grid(True)
            axes[1, 0].set_xlim([0, 360])

            axes[1, 1].plot(angle2, dist2, "k-")
            axes[1, 1].set_title(r"Signature $S_2(\theta)$")
            axes[1, 1].set_xlabel("Angle (degrees)")
            axes[1, 1].set_ylabel("Distance")
            axes[1, 1].grid(True)
            axes[1, 1].set_xlim([0, 360])

            plt.tight_layout()

            plt.savefig("Figure1211.png")
            print("Saved Figure1211.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1211bis(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1211Bis.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import os
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            import ia870 as ia
            from helpers.libdipum.bwboundaries import bwboundaries
            from helpers.libdipum.signature import signature
            from helpers.libdipum.data_path import dip_data

            # Helper for robust image reading
            def read_image_robust(path: Any):
                """read_image_robust."""
                # Try skimage first
                try:
                    img = imread(path)
                    return img
                except Exception as e:
                    # Try PIL
                    try:
                        from PIL import Image

                        img_pil = Image.open(path)
                        # PIL opens as object, convert to numpy
                        return np.array(img_pil)
                    except ImportError:
                        pass
                    except Exception:
                        pass

                    # Try OpenCV
                    try:
                        import cv2

                        img_cv = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                        if img_cv is not None:
                            return img_cv
                    except ImportError:
                        pass

                    # Re-raise original error if all fail
                    print(f"Failed to read image {path} with skimage, PIL, and OpenCV.")
                    raise e

            def Figure1211Bis():
                """Figure1211Bis."""
                print("Figure 12.11 Bis (Signatures with ia870)")

                # 1. Input Choice
                try:
                    choix = int(input("Artificial data (1) or airplane (2) : "))
                except ValueError:
                    print("Invalid input.")
                    return

                f = None
                B1 = None
                title_str = ""

                if choix == 1:
                    # Artificial
                    f = np.zeros((100, 100), dtype=bool)
                    f[50, 50] = True
                    f_uint8 = f.astype(np.uint8)

                    try:
                        choix1 = int(input("Diamond (1), Square (2) or Disk (3) : "))
                    except ValueError:
                        print("Invalid input.")
                        return

                    if choix1 == 1:
                        B0 = ia.iasecross(20)
                        title_str = "Diamond"
                    elif choix1 == 2:
                        B0 = ia.iasebox(20)
                        title_str = "Square"
                    elif choix1 == 3:
                        B0 = ia.iasedisk(20)
                        title_str = "Disk"
                    else:
                        print("Plouc (Invalid choice)")
                        return

                    # Dilate
                    f_dil = ia.iadil(f_uint8, B0)
                    f_binary = f_dil > 0
                    B1 = ia.iasedisk(1)

                elif choix == 2:
                    # Airplane
                    try:
                        choix1 = int(input("Plane number from 1 to 4 : "))
                    except ValueError:
                        print("Invalid input.")
                        return

                    filename = f"Plane{choix1}.tif"
                    try:
                        path = dip_data(filename)
                    except FileNotFoundError:
                        local_path = os.path.join(".", filename)
                        if os.path.exists(local_path):
                            path = local_path
                        else:
                            print(f"{filename} not found.")
                            return

                    if not path:
                        return

                    print(f"Loading {path}...")
                    try:
                        f_in = read_image_robust(path)
                    except Exception as e:
                        print(f"Error loading image: {e}")
                        return

                    if f_in.ndim == 3:
                        f_in = f_in[:, :, 0]

                    # Process
                    f_neg = ia.ianeg(f_in)
                    f_opened = ia.iaareaopen(f_neg, 4)
                    f_binary = f_opened > 0
                    B1 = ia.iasedisk(4)
                    title_str = f"Plane {choix1}"

                else:
                    print("Plouc (Invalid choice)")
                    return

                # Boundary Extraction
                boundaries = bwboundaries(f_binary, conn=8)
                if not boundaries:
                    print("No boundaries found.")
                    return
                B = boundaries[0]  # Take first

                # Signature
                try:
                    dist, angle = signature(B)
                except Exception as e:
                    print(f"Error computing signature: {e}")
                    return

                # Centroid
                y0 = np.mean(B[:, 0])
                x0 = np.mean(B[:, 1])

                # Display
                fig, axes = plt.subplots(2, 2, figsize=(10, 8))

                # 1. Image with Marker
                # Mask logic from MATLAB: mmshow(f, mmdil(Mask, B1))
                # We can overlay.
                axes[0, 0].imshow(f_binary, cmap="gray")
                axes[0, 0].plot(x0, y0, "rx")  # Simple centroid marker
                axes[0, 0].set_title(f"{title_str}, GC")
                axes[0, 0].axis("off")

                # 2. Angle
                axes[0, 1].plot(angle, "b-")
                axes[0, 1].set_title("Angle")
                axes[0, 1].set_xlabel("Contour Sample")
                axes[0, 1].axis("tight")

                # 3. Distance (ST)
                axes[1, 0].plot(dist, "k-")
                axes[1, 0].set_title("Distance to GC")
                axes[1, 0].set_xlabel("Contour Sample")
                axes[1, 0].axis("tight")

                # 4. Signature (Dist vs Angle)
                axes[1, 1].plot(angle, dist, "k-")
                axes[1, 1].set_title("Distance to GC (Signature)")
                axes[1, 1].set_xlabel("Angle")
                axes[1, 1].axis("tight")

                plt.tight_layout()
                out_name = f"Figure1211Bis_{title_str.replace(' ', '')}.png"
                plt.savefig(out_name)
                print(f"Saved {out_name}")
                plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1213(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1213.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            import ia870 as ia

            print("Running Figure1213 (Distance Transform with ia870)...")

            # Data
            # print("Generating X3...")
            X3 = ~ia.iaframe(np.ones((200, 400), dtype=bool), 20, 20)

            # 2. Distance Transform
            DT3 = ia.iadist(X3, ia.iasebox(), "EUCLIDEAN")

            # 3. Display
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            axes[0].imshow(X3, cmap="gray")
            axes[0].set_title("X")
            axes[0].axis("off")

            axes[1].imshow(DT3, cmap="gray")
            axes[1].set_title("DT(X)")
            axes[1].axis("off")

            plt.tight_layout()
            plt.savefig("Figure1213.png")
            print("Saved Figure1213.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1214(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1214.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            import ia870 as ia
            from helpers.libdipum.skeleton import skeleton
            from helpers.libdipum.data_path import dip_data

            print("Running Figure1214 (Skeleton comparisons)...")

            def _imshow_ready(x: Any):
                """Convert ia870 color output from (C,H,W) to (H,W,C) for matplotlib."""
                x = np.asarray(x)
                if x.ndim == 3 and x.shape[0] in (3, 4) and x.shape[-1] not in (3, 4):
                    x = np.moveaxis(x, 0, -1)
                return x

            # Parameters / structuring elements
            Iab = ia.iahomothin()
            Iab1 = ia.iaendpoints()
            Bc4 = ia.iasecross()

            # Data
            X = imread(dip_data("blood-vessels.tif"))
            if X.ndim == 3:
                X = X[:, :, 0]
            X = X > 0
            X = ia.ianeg(X)
            # Skeleton by thinning
            XThin1 = ia.iathin(X, Iab)

            # Pruning
            XThin2 = ia.iathin(XThin1, Iab1, 30)

            # Distance transform and regional maxima
            DT = ia.iadist(X, Bc4, "EUCLIDEAN")
            XThin3 = ia.iaregmax(DT, Bc4)

            # Fast marching skeleton
            S = skeleton(X, verbose=True)

            # Display figure 1
            fig1, ax = plt.subplots(2, 2, figsize=(10, 8))

            ax[0, 0].imshow(X, cmap="gray")
            ax[0, 0].set_title("X")
            ax[0, 0].axis("off")

            ax[0, 1].imshow(
                _imshow_ready(ia.iagshow(X, ia.iadil(XThin1))), interpolation="nearest"
            )
            ax[0, 1].set_title("Skeleton by thinning")
            ax[0, 1].axis("off")

            ax[1, 0].imshow(
                _imshow_ready(ia.iagshow(X, ia.iadil(XThin2))), interpolation="nearest"
            )
            ax[1, 0].set_title("Pruning")
            ax[1, 0].axis("off")

            ax[1, 1].imshow(
                _imshow_ready(ia.iagshow(DT, ia.iadil(XThin3, ia.iasecross(2)))),
                interpolation="nearest",
            )
            ax[1, 1].set_title("RMAX(DT(X))")
            ax[1, 1].axis("off")

            fig1.tight_layout()
            fig1.savefig("Figure1214.png")

            # Display figure 2 (fast marching)
            fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
            ax2.imshow(X, cmap="gray")
            ax2.set_title("Fast Marching")
            ax2.set_axis_off()
            ax2.invert_yaxis()  # MATLAB axis ij

            rng = np.random.default_rng(0)
            for L in S:
                L = np.asarray(L)
                if L.size == 0:
                    continue
                color = rng.random(3)
                ax2.plot(L[:, 1], L[:, 0], "-", color=color, linewidth=1.0)

            fig2.tight_layout()
            fig2.savefig("Figure1214Bis.png")

            print("Saved Figure1214.png and Figure1214Bis.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1216(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1216.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np

            from helpers.libdipum.bwboundaries import bwboundaries
            from helpers.libdipum.freemanChainCode import freemanChainCode

            print("Running Figure1216 (Freeman chain code examples)...")

            # Parameters
            Conn = 4

            # Data
            X = []
            X.append(np.ones((2, 2), dtype=bool))
            X.append(np.ones((2, 3), dtype=bool))
            X.append(np.ones((3, 3), dtype=bool))
            x4 = np.ones((3, 3), dtype=bool)
            x4[0, -1] = False
            X.append(x4)
            X.append(np.ones((2, 4), dtype=bool))

            # Boundaries and Freeman chain code
            for iter_idx, Xi in enumerate(X, start=1):
                B_list = bwboundaries(Xi, conn=Conn)
                if not B_list:
                    print(f"iter {iter_idx}: no boundary found")
                    continue

                B = B_list[0]
                c = freemanChainCode(B, Conn)

                # MATLAB prints `c` in the loop; print all key fields explicitly.
                print(f"iter {iter_idx}:")
                print("  x0y0   =", c.x0y0)
                print("  fcc    =", c.fcc)
                print("  mm     =", c.mm)
                print("  diff   =", c.diff)
                print("  diffmm =", c.diffmm)
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1217(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1217.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np

            from helpers.libdipum.bwboundaries import bwboundaries
            from helpers.libdipum.freemanChainCode import freemanChainCode

            print("Running Figure1217 (Freeman chain code)...")

            # Parameters
            Conn = 4

            # Data
            X = np.ones((4, 7), dtype=bool)
            X[0, 5:7] = False
            X[3, [0, 5, 6]] = False

            # Boundaries
            B_list = bwboundaries(X, Conn)
            if not B_list:
                raise RuntimeError("No boundary found.")
            B = B_list[0]

            # Freeman chain code
            c = freemanChainCode(B, Conn)

            print("c.x0y0   =", c.x0y0)
            print("c.fcc    =", c.fcc)
            print("c.mm     =", c.mm)
            print("c.diff   =", c.diff)
            print("c.diffmm =", c.diffmm)
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1219(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1219.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from scipy.ndimage import binary_dilation

            from helpers.libgeneral.mmshow import mmshow
            from helpers.libdipum.bwboundaries import bwboundaries
            from helpers.libdipum.bound2im import bound2im
            from helpers.libdipum.frdescp import frdescp
            from helpers.libdipum.ifrdescp import ifrdescp
            from helpers.libdipum.data_path import dip_data

            print("Running Figure1219 (Fourier descriptors of boundary)...")

            # Data
            image_path = dip_data("Fig1116(a)(chromo_binary).tif")
            f = imread(image_path)
            if f.ndim == 3:
                f = f[:, :, 0]
            f = f > 0
            NbRows, NbCols = f.shape

            # Boundaries
            B_list = bwboundaries(f, 8)
            if not B_list:
                raise RuntimeError("No boundary found in image.")
            B = B_list[0]
            BIm = bound2im(B, NbRows, NbCols)
            np_ = B.shape[0]

            # Fourier descriptor
            Z = frdescp(B)

            # Inverse fourier descriptor
            NbFourierDescriptor = [1434, 286, 144, 72, 36, 18, 8]
            BAppIm = []
            for nd in NbFourierDescriptor:
                nd_use = min(nd, len(Z))
                if nd_use % 2 == 1:
                    nd_use -= 1
                BApp = ifrdescp(Z, nd_use)
                BAppIm.append(bound2im(BApp, NbRows, NbCols))

            # Display 1
            IxCenter = int(np.round(len(B) / 2.0) + 1)
            omega = ((np.arange(1, len(B) + 1) - IxCenter) / IxCenter) * np.pi

            fig1, ax = plt.subplots(2, 2, figsize=(11, 8))

            plt.sca(ax[0, 0])
            mmshow(f.astype(float), binary_dilation(BIm), binary_dilation(BIm))
            ax[0, 0].set_title("f, Boundary(f)")
            ax[0, 0].axis("off")

            ax[0, 1].plot(
                np.arange(1, np_ + 1), B[:, 0], "r", np.arange(1, np_ + 1), B[:, 1], "g"
            )
            ax[0, 1].set_xlabel("k")
            ax[0, 1].set_title("x[k], y[k]")
            ax[0, 1].axis("tight")

            with np.errstate(divide="ignore"):
                mag_db = 20.0 * np.log10(np.abs(Z))
            ax[1, 0].plot(omega, mag_db)
            ax[1, 0].set_title("|Z(ω)|, Z(ω) = fft(B2[k])")
            ax[1, 0].set_xlabel("ω")
            ax[1, 0].axis("tight")

            ax[1, 1].plot(omega, np.unwrap(np.angle(Z)) * 180.0 / np.pi)
            ax[1, 1].set_title("∠(Z(ω))")
            ax[1, 1].set_xlabel("ω")
            ax[1, 1].axis("tight")

            fig1.tight_layout()
            fig1.savefig("Figure1219.png")

            # Display 2
            fig2, ax2 = plt.subplots(2, 4, figsize=(14, 7))
            ax2 = ax2.ravel()

            ax2[0].imshow(BIm, cmap="gray", interpolation="nearest")
            ax2[0].set_title("Boundary (f)")
            ax2[0].axis("off")

            for cpt, nd in enumerate(NbFourierDescriptor, start=1):
                plt.sca(ax2[cpt])
                mmshow(BIm.astype(float), BAppIm[cpt - 1], BAppIm[cpt - 1])
                ax2[cpt].set_title(f"IFD {min(nd, len(Z))}/{len(Z)}")
                ax2[cpt].axis("off")

            # Approximate MATLAB linkaxes behavior.
            xlim = ax2[0].get_xlim()
            ylim = ax2[0].get_ylim()
            for a in ax2[1:]:
                a.set_xlim(xlim)
                a.set_ylim(ylim)

            fig2.tight_layout()
            fig2.savefig("Figure1219Bis.png")

            print("Saved Figure1219.png and Figure1219Bis.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1222(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1222.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.libdipum.data_path import dip_data

            from helpers.libdip.binaryRegionProps4e import binaryRegionProps4e

            print("Running Figure1222 (Shape descriptors)...")

            # Parameters
            FileNames = [
                "wingding-circle-solid.tif",
                "wingding-star-6pt.tif",
                "wingding-square-solid.tif",
                "wingding-teardrop.tif",
            ]

            # Data
            X = []
            for name in FileNames:
                img = imread(dip_data(name))
                if img.ndim == 3:
                    img = img[:, :, 0]
                X.append(img > 0)

            # Properties
            Compactness1 = []
            Circularity1 = []
            Eccentricity1 = []
            for x in X:
                p = binaryRegionProps4e(x)
                Compactness1.append(p["comp"])
                Circularity1.append(p["circ"])
                Eccentricity1.append(p["ecc"])

            Compactness1 = np.array(Compactness1, dtype=float)
            Circularity1 = np.array(Circularity1, dtype=float)
            Eccentricity1 = np.array(Eccentricity1, dtype=float)

            # Display 1
            fig1, ax = plt.subplots(2, 2, figsize=(10, 8))
            ax = ax.ravel()
            for i in range(len(FileNames)):
                ax[i].imshow(X[i], cmap="gray", interpolation="nearest")
                ax[i].set_title(
                    f"{Compactness1[i]:.6g}, {Circularity1[i]:.6g}, {Eccentricity1[i]:.6g}"
                )
                ax[i].axis("off")
            fig1.tight_layout()
            fig1.savefig("Figure1222.png")

            # Display 2
            fig2 = plt.figure(figsize=(10, 7))
            ax3 = fig2.add_subplot(111, projection="3d")

            for i, name in enumerate(FileNames):
                ax3.stem(
                    [Circularity1[i]],
                    [Eccentricity1[i]],
                    [Compactness1[i]],
                    linefmt="C0-",
                    markerfmt="C0o",
                    basefmt=" ",
                )
                ax3.text(Circularity1[i], Eccentricity1[i], Compactness1[i], name)

            ax3.set_xlabel("Circularity")
            ax3.set_ylabel("Eccentricity")
            ax3.set_zlabel("Compactness")
            ax3.legend(["circle", "star", "square", "drop"])
            ax3.grid(True)
            ax3.view_init(elev=28, azim=134)

            fig2.tight_layout()
            fig2.savefig("Figure1223.png")

            print("Saved Figure1222.png and Figure1223.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1224(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1224.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.libdipum.data_path import dip_data

            print("Running Figure1224 (Americas at night ratios)...")

            # Data
            Names = [
                "americas-at-night1",
                "americas-at-night2",
                "americas-at-night3",
                "americas-at-night4",
            ]
            OtherNames = ["Canada", "USA", "Central America", "South America"]

            f = []
            X1 = []
            X2 = []
            Total = 0

            missing_inputs = []
            for name in Names:
                filename = f"{name}.tif"
                try:
                    img = imread(dip_data(filename))
                except FileNotFoundError:
                    missing_inputs.append(filename)
                    continue
                if img.ndim == 3:
                    img = img[:, :, 0]
                f.append(img)

                x1, x2 = np.where(img != 0)  # Light is there
                X1.append(x1)
                X2.append(x2)
                Total += x1.size

            if missing_inputs:
                fig, ax = plt.subplots(figsize=(9, 3))
                ax.axis("off")
                ax.text(
                    0.0,
                    0.9,
                    "Figure1224 input data missing:",
                    transform=ax.transAxes,
                    fontsize=11,
                    fontweight="bold",
                )
                ax.text(
                    0.0,
                    0.7,
                    "\n".join(missing_inputs),
                    transform=ax.transAxes,
                    fontsize=10,
                    family="monospace",
                )
                ax.text(
                    0.0,
                    0.25,
                    "Add these files to AllDataFiles or set DIP4E_DATA_DIR.",
                    transform=ax.transAxes,
                    fontsize=10,
                )
                fig.tight_layout()
                fig.savefig("Figure1224.png")
                print(
                    "Saved Figure1224.png placeholder (missing inputs): "
                    + ", ".join(missing_inputs)
                )
                plt.show()
                return self._collect_new_figures(pre_fig_nums)

            # Ratio
            Ratio = np.zeros(len(Names), dtype=float)
            for i in range(len(Names)):
                Ratio[i] = X1[i].size / Total if Total > 0 else 0.0

            # Display
            fig, ax = plt.subplots(2, 2, figsize=(10, 8))
            ax = ax.ravel()

            for i in range(len(Names)):
                ax[i].imshow(f[i], cmap="gray")
                ax[i].set_title(
                    f"{OtherNames[i]} {100 * Ratio[i]:.6g} % of the energy is spent here"
                )
                ax[i].axis("off")

            fig.tight_layout()
            fig.savefig("Figure1224.png")

            print("Saved Figure1224.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1228(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1228.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            import ia870 as ia
            from helpers.libdipum.data_path import dip_data

            print("Running Figure1228 (Morphology + labeling + skeleton)...")

            # Init
            Bc8 = ia.iasebox()

            # Data
            image_path = dip_data("WashingtonDC-Band4-NearInfrared-512.tif")
            f = imread(image_path)
            if f.ndim == 3:
                f = f[:, :, 0]

            # Erosion
            fe = ia.iaero(f, ia.iasedisk(1))

            # Thresholding (as in MATLAB script)
            T = 65
            X = fe < T

            # Labeling
            XLabel = ia.ialabel(X, Bc8)
            NConnComp = int(np.max(XLabel))
            XAreaLabel = ia.iablob(XLabel, "AREA", "image")

            # Keeping the largest connected component
            Y = XAreaLabel >= np.max(XAreaLabel)

            # Skeleton
            Sk = ia.iathin(Y, ia.iahomothin(), -1, 45, "CLOCKWISE")

            # Display
            fig, ax = plt.subplots(2, 3, figsize=(12, 8))
            ax = ax.ravel()

            ax[0].imshow(f, cmap="gray")
            ax[0].set_title("f")
            ax[0].axis("off")

            ax[1].imshow(fe, cmap="gray")
            ax[1].set_title("fe = ε_B(f)")
            ax[1].axis("off")

            ax[2].imshow(X, cmap="gray")
            ax[2].set_title(f"X = fe < {T}")
            ax[2].axis("off")

            ax[3].imshow(np.moveaxis(ia.iaglblshow(XLabel), 0, -1))
            ax[3].set_title(f"label(X, Bc_8), N_CC = {NConnComp}")
            ax[3].axis("off")

            ax[4].imshow(Y, cmap="gray")
            ax[4].set_title(f"Largest CC, A = {int(np.sum(Y))}")
            ax[4].axis("off")

            ax[5].imshow(Sk, cmap="gray")
            ax[5].set_title("Sk = skel(Y)")
            ax[5].axis("off")

            # linkaxes approximation
            xlim = ax[0].get_xlim()
            ylim = ax[0].get_ylim()
            for a in ax[1:]:
                a.set_xlim(xlim)
                a.set_ylim(ylim)

            fig.tight_layout()
            fig.savefig("Figure1228.png")

            print("Saved Figure1228.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1229(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1229.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            from typing import Any
            import os
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.libdipum.data_path import dip_data

            print("Running Figure1229 (Histogram-based texture statistics)...")

            # Data
            files = [
                "OpticalMicroscope-superconductor-smooth-texture.tif",
                "OpticalMicroscope-cholesterol-rough-texture.tif",
                "OpticalMicroscope-microporcessor-regular-texture.tif",
            ]
            names = ["Smooth", "Coarse", "Regular"]

            def _to_gray_u8(img: Any):
                """_to_gray_u8."""
                if img.ndim == 3:
                    img = img[:, :, 0]
                if img.dtype != np.uint8:
                    # preserve integer grayscale behavior from MATLAB hist 0..255
                    img = np.clip(img, 0, 255).astype(np.uint8)
                return img

            def _imcrop_matlab(img: Any, rect: Any):
                """_imcrop_matlab."""
                # MATLAB rect = [x, y, w, h], output includes both ends => (h+1, w+1)
                x, y, w, h = rect
                x = int(round(x))
                y = int(round(y))
                w = int(round(w))
                h = int(round(h))
                return img[y : y + h + 1, x : x + w + 1]

            f = []
            for fn in files:
                img = imread(dip_data(fn))
                f.append(_to_gray_u8(img))

            fc = [
                _imcrop_matlab(f[0], [80, 230, 68, 73]),
                _imcrop_matlab(f[1], [35, 162, 75, 80]),
                _imcrop_matlab(f[2], [16, 13, 64, 80]),
            ]

            # Histogram based statistical analysis
            z = np.arange(256, dtype=float)
            m = np.zeros(len(fc), dtype=float)
            mu2 = np.zeros(len(fc), dtype=float)
            mu3 = np.zeros(len(fc), dtype=float)
            NormVar = np.zeros(len(fc), dtype=float)
            Uniformity = np.zeros(len(fc), dtype=float)
            Entropy = np.zeros(len(fc), dtype=float)
            p_list = []

            for i, img in enumerate(fc):
                hist = np.bincount(img.ravel(), minlength=256).astype(float)
                p = hist / np.sum(hist)
                p_list.append(p)

                m[i] = np.sum(z * p)
                mu2[i] = np.sum(((z - m[i]) ** 2) * p)

                if mu2[i] > 0:
                    mu3[i] = np.sum(((z - m[i]) ** 3) * p) / (mu2[i] ** 1.5)
                else:
                    mu3[i] = 0.0

                NormVar[i] = mu2[i] / (255.0**2)
                Uniformity[i] = np.sum(p**2)
                nz = p > 0
                Entropy[i] = -np.sum(p[nz] * np.log2(p[nz]))

            R = 1.0 - 1.0 / (1.0 + NormVar)

            # Display
            fig, ax = plt.subplots(3, 3, figsize=(12, 10))
            ax = ax.ravel()

            for i in range(len(f)):
                ax[i].imshow(f[i], cmap="gray")
                ax[i].set_title(names[i])
                ax[i].axis("off")

                ax[i + 3].imshow(fc[i], cmap="gray")
                ax[i + 3].set_title(
                    f"μ={m[i]:.3g}, σ={np.sqrt(mu2[i]):.3g}, R = {R[i]:.3g}"
                )
                ax[i + 3].axis("off")

                ax[i + 6].plot(z, p_list[i])
                ax[i + 6].set_xlim(0, 255)
                ax[i + 6].set_aspect("auto")
                ax[i + 6].set_xlabel("z")
                ax[i + 6].set_ylabel("p(z)")
                ax[i + 6].set_title(
                    f"{mu3[i]:.3g}, {Uniformity[i]:.3g}, {Entropy[i]:.3g}"
                )

            fig.tight_layout()
            fig.savefig("Figure1229.png")

            print("Saved Figure1229.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1231(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1231.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.libdipum.data_path import dip_data

            print("Running Figure1231 (Three texture strips)...")

            # Data
            f1 = imread(dip_data("strip-uniform-noise.tif"))
            f2 = imread(dip_data("strip-2Dsinusoidal-waveform.tif"))
            f3 = imread(dip_data("strip-cktboard-section.tif"))

            # Display
            fig, ax = plt.subplots(3, 1, figsize=(8, 10))

            ax[0].imshow(f1, cmap="gray")
            ax[0].axis("off")

            ax[1].imshow(f2, cmap="gray")
            ax[1].axis("off")

            ax[2].imshow(f3, cmap="gray")
            ax[2].axis("off")

            fig.tight_layout()
            fig.savefig("Figure1231.png")

            print("Saved Figure1231.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1232(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1232.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.libdipum.data_path import dip_data

            from helpers.libgeneral.graycomatrix import graycomatrix
            from helpers.libgeneral.graycoprops import graycoprops

            print("Running Figure1232 (GLCM texture analysis)...")

            Names = [
                "strip-uniform-noise",
                "strip-2Dsinusoidal-waveform",
                "strip-cktboard-section",
            ]

            f = []
            for name in Names:
                img = imread(dip_data(f"{name}.tif"))
                if img.ndim == 3:
                    img = img[:, :, 0]
                f.append(img)

            # Co-occurrence matrix and properties
            glcm = []
            props = []
            MaxProb = np.zeros(len(Names), dtype=float)
            Entropy = np.zeros(len(Names), dtype=float)

            for i in range(len(Names)):
                # Horizontal, one-pixel distance
                G, _ = graycomatrix(f[i], NumLevels=256, Offset=np.array([[0, 1]]))
                G2 = G[:, :, 0]
                glcm.append(G2)

                st = graycoprops(
                    G2, ["contrast", "correlation", "homogeneity", "energy"]
                )
                props.append(
                    {
                        "Contrast": float(st["Contrast"][0]),
                        "Correlation": float(st["Correlation"][0]),
                        "Homogeneity": float(st["Homogeneity"][0]),
                        "Energy": float(st["Energy"][0]),
                    }
                )

                p = G2 / np.sum(G2) if np.sum(G2) > 0 else G2.astype(float)
                MaxProb[i] = float(np.max(p)) if p.size else 0.0

                nz = p > 0
                Entropy[i] = -np.sum(p[nz] * np.log2(p[nz])) if np.any(nz) else 0.0

            # Display
            fig, ax = plt.subplots(2, 3, figsize=(13, 8))
            ax = ax.ravel()

            for i in range(len(Names)):
                ax[i].imshow(f[i], cmap="gray")
                ax[i].set_title(
                    f"p_max = {MaxProb[i]:.2g}, "
                    f"rho = {props[i]['Correlation']:.2g}, "
                    f"C = {props[i]['Contrast']:.3g}"
                )
                ax[i].axis("off")

                if i == 2:
                    ax[i + 3].imshow(glcm[i], cmap="gray", vmin=0, vmax=40)
                else:
                    ax[i + 3].imshow(glcm[i], cmap="gray")

                ax[i + 3].set_title(
                    f"U = {props[i]['Energy']:.2g}, "
                    f"H = {props[i]['Homogeneity']:.3g}, "
                    f"E = {Entropy[i]:.2g}"
                )
                ax[i + 3].axis("off")

            fig.tight_layout()
            fig.savefig("Figure1232.png")

            print("Saved Figure1232.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1233(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1233.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.libdipum.data_path import dip_data

            from helpers.libgeneral.graycomatrix import graycomatrix
            from helpers.libgeneral.graycoprops import graycoprops

            print("Running Figure1233 (Correlation vs horizontal offset)...")

            # Data
            Names = [
                "strip-uniform-noise",
                "strip-2Dsinusoidal-waveform",
                "strip-cktboard-section",
            ]

            f = []
            for name in Names:
                img = imread(dip_data(f"{name}.tif"))
                if img.ndim == 3:
                    img = img[:, :, 0]
                f.append(img)

            # Co-occurrence matrix property (correlation) vs offset
            Properties = np.zeros((len(Names), 50), dtype=float)

            for cpt in range(len(Names)):
                for offset in range(1, 51):
                    G, _ = graycomatrix(
                        f[cpt], NumLevels=256, Offset=np.array([[0, offset]])
                    )
                    st = graycoprops(G, "correlation")
                    Properties[cpt, offset - 1] = st["Correlation"][0]

            # Display
            fig, ax = plt.subplots(2, 3, figsize=(13, 8))
            ax = ax.ravel()

            for i in range(len(Names)):
                ax[i].imshow(f[i], cmap="gray")
                ax[i].axis("off")

                ax[i + 3].plot(np.arange(1, 51), Properties[i, :])
                ax[i + 3].set_xlabel("Horizontal offset")
                ax[i + 3].set_title("correlation")
                ax[i + 3].set_xlim(1, 50)
                ax[i + 3].set_ylim(-1, 1)

            fig.tight_layout()
            fig.savefig("Figure1233.png")

            print("Saved Figure1233.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1235(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1235.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.libdipum.data_path import dip_data

            from helpers.libdipum.specxture import specxture

            print("Running Figure1235 (Matches and spectra)...")

            # Data
            f1 = imread(dip_data("matches-random.tif"))
            f2 = imread(dip_data("matches-aligned.tif"))

            if f1.ndim == 3:
                f1 = f1[:, :, 0]
            if f2.ndim == 3:
                f2 = f2[:, :, 0]

            # Spectrum (cartesian mode)
            F1 = np.fft.fft2(f1)
            F2 = np.fft.fft2(f2)

            # Spectrum (polar mode)
            F1Rho, F1Theta, _ = specxture(f1)
            F2Rho, F2Theta, _ = specxture(f2)

            # Display
            fig, ax = plt.subplots(2, 2, figsize=(10, 8))

            ax[0, 0].imshow(f1, cmap="gray")
            ax[0, 0].set_title("f_1[k, l]")
            ax[0, 0].axis("off")

            ax[0, 1].imshow(f2, cmap="gray")
            ax[0, 1].set_title("f_2[k, l]")
            ax[0, 1].axis("off")

            ax[1, 0].imshow(np.log10(np.abs(np.fft.fftshift(F1))), cmap="gray")
            ax[1, 0].set_title("|F_1[m, n]|")
            ax[1, 0].axis("off")

            ax[1, 1].imshow(np.log10(np.abs(np.fft.fftshift(F2))), cmap="gray")
            ax[1, 1].set_title("|F_2[m, n]|")
            ax[1, 1].axis("off")

            fig.tight_layout()
            fig.savefig("Figure1235.png")

            print("Saved Figure1235.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1236(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1236.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.libdipum.data_path import dip_data
            from helpers.libdipum.specxture import specxture

            print("Running Figure1236 (Radial and angular spectra)...")

            # Data
            f1 = imread(dip_data("matches-random.tif"))
            f2 = imread(dip_data("matches-aligned.tif"))

            if f1.ndim == 3:
                f1 = f1[:, :, 0]
            if f2.ndim == 3:
                f2 = f2[:, :, 0]

            # Spectrum (cartesian mode)
            F1 = np.fft.fft2(f1)
            F2 = np.fft.fft2(f2)

            # Spectrum (polar mode)
            F1Rho, F1Theta, _ = specxture(f1)
            F2Rho, F2Theta, _ = specxture(f2)

            # Display
            fig, ax = plt.subplots(2, 2, figsize=(10, 8))

            ax[0, 0].plot(F1Rho)
            ax[0, 0].set_xlabel("rho")
            ax[0, 0].set_title("|F_1(rho)|")
            ax[0, 0].axis("tight")
            # ax[0, 0].set_ylim(0, 9)
            ax[0, 0].set_box_aspect(1)

            ax[0, 1].plot(F1Theta)
            ax[0, 1].set_xlabel("theta")
            ax[0, 1].set_title("|F_1(theta)|")
            ax[0, 1].axis("tight")
            # ax[0, 1].set_ylim(1.8, 2.7)
            ax[0, 1].set_box_aspect(1)

            ax[1, 0].plot(F2Rho)
            ax[1, 0].set_xlabel("rho")
            ax[1, 0].set_title("|F_2(rho)|")
            ax[1, 0].axis("tight")
            # ax[1, 0].set_ylim(0, 6)
            ax[1, 0].set_box_aspect(1)

            ax[1, 1].plot(F2Theta)
            ax[1, 1].set_xlabel("theta")
            ax[1, 1].set_title("|F_2(theta)|")
            ax[1, 1].axis("tight")
            # ax[1, 1].set_ylim(1.6, 3.6)
            ax[1, 1].set_box_aspect(1)

            fig.tight_layout()
            fig.savefig("Figure1236.png")

            print("Saved Figure1236.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1237(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1237.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from scipy.ndimage import rotate

            from helpers.libdipum.invmoments import invmoments
            from helpers.libdipum.data_path import dip_data

            print("Running Figure1237 (Invariant moments under transformations)...")

            # Parameters
            Offset = 150
            r = 2
            Theta1 = 45
            Theta2 = 90

            # Data
            image_path = dip_data("Fig1123(a)(Original_Padded_to_568_by_568).tif")
            fP = imread(image_path)
            if fP.ndim == 3:
                fP = fP[:, :, 0]

            NR, NC = fP.shape
            Motif = fP[84:484, 84:484]  # MATLAB 85:484

            fT = np.zeros_like(fP, dtype=np.uint8)
            h, w = Motif.shape
            fT[Offset : Offset + h, Offset : Offset + w] = Motif

            fHS = Motif[::r, ::r]
            fHSP = np.pad(
                fHS, ((184, 184), (184, 184)), mode="constant", constant_values=0
            )

            fM = np.fliplr(Motif)
            fMP = np.pad(fM, ((84, 84), (84, 84)), mode="constant", constant_values=0)

            fR45 = rotate(
                Motif.astype(float),
                Theta1,
                reshape=True,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
            fR45 = np.clip(np.rint(fR45), 0, 255).astype(np.uint8)

            fR90 = rotate(
                Motif.astype(float),
                Theta2,
                reshape=True,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
            fR90 = np.clip(np.rint(fR90), 0, 255).astype(np.uint8)
            fR90P = np.pad(
                fR90, ((84, 84), (84, 84)), mode="constant", constant_values=0
            )

            # Invariant moments
            imgs = [fP, fT, fHSP, fMP, fR45, fR90P]
            Phi = np.zeros((6, 7), dtype=float)
            for i, img in enumerate(imgs):
                m = invmoments(img)
                # MATLAB expression: abs(log10(invmoments(...))).
                # Use complex log to preserve sign information for negative moments.
                mc = m.astype(np.complex128)
                mc[m == 0] = np.finfo(float).eps
                Phi[i, :] = np.abs(np.log10(mc))

            # Display phi transpose in console
            print("Phi.T =")
            print(Phi.T)

            # Figure 1
            fig1, ax = plt.subplots(2, 3, figsize=(12, 8))
            ax = ax.ravel()

            ax[0].imshow(fP, cmap="gray")
            ax[0].set_title("Original")
            ax[0].axis("off")

            ax[1].imshow(fT, cmap="gray")
            ax[1].set_title(f"Translation, dk = dl = {Offset}")
            ax[1].axis("off")

            ax[2].imshow(fHSP, cmap="gray")
            ax[2].set_title(f"Resizing, r = {r}")
            ax[2].axis("off")

            ax[3].imshow(fMP, cmap="gray")
            ax[3].set_title("Mirroring")
            ax[3].axis("off")

            ax[4].imshow(fR45, cmap="gray")
            ax[4].set_title(f"Rotation, theta = {Theta1}")
            ax[4].axis("off")

            ax[5].imshow(fR90P, cmap="gray")
            ax[5].set_title(f"Rotation, theta = {Theta2}")
            ax[5].axis("off")

            fig1.tight_layout()
            fig1.savefig("Figure1237.png")

            # Figure 2
            fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
            ax2.plot(Phi, "o-")
            ax2.set_xlabel("Transformation")
            ax2.set_ylabel("Phi_i")
            ax2.set_title("Moments")
            ax2.set_xticks(np.arange(6))
            ax2.set_xticklabels(
                ["None", "Translation", "Scaling", "Mirror", "Rotation45", "Rotation90"]
            )
            ax2.legend(["Phi_1", "Phi_2", "Phi_3", "Phi_4", "Phi_5", "Phi_6", "Phi_7"])

            fig2.tight_layout()
            fig2.savefig("Figure1237Bis.png")

            print("Saved Figure1237.png and Figure1237Bis.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1238to42(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1238to42.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from skimage.io import imread
            from helpers.libdipum.data_path import dip_data

            from helpers.libdipum.imstack2vectors import imstack2vectors
            from helpers.libdipum.principalComponents import principalComponents

            print("Running Figure1238to42 (PCA on WashingtonDC multispectral stack)...")

            # Parameters
            PCAKeep = 2

            # Data
            files = [
                "WashingtonDC-Band1-Blue-512.tif",
                "WashingtonDC-Band2-Green-512.tif",
                "WashingtonDC-Band3-Red-512.tif",
                "WashingtonDC-Band4-NearInfrared-512.tif",
                "washingtonDC-Band5-MiddleInfrared-512.tif",
                "washingtonDC-Band6-ThermalInfrared-512.tif",
            ]

            f = []
            for fn in files:
                img = imread(dip_data(fn))
                if img.ndim == 3:
                    img = img[:, :, 0]
                f.append(img.astype(float))

            S = np.stack(f, axis=2)  # (NR, NC, 6)
            Size = S[:, :, 0].shape

            # Stack -> vectors
            X, R = imstack2vectors(S)  # (NR*NC, 6)

            # Principal components (complete 6/6)
            P = principalComponents(X, 6)
            d = np.diag(P["Cy"])

            g = np.zeros((Size[0], Size[1], 6), dtype=float)
            for cpt in range(6):
                g[:, :, cpt] = np.reshape(P["Y"][:, cpt], Size, order="F")

            # Principal components (partial 2/6)
            P1 = principalComponents(X, PCAKeep)
            d1 = np.diag(P1["Cy"])

            h = np.zeros((Size[0], Size[1], 6), dtype=float)
            for cpt in range(6):
                h[:, :, cpt] = np.reshape(P1["X"][:, cpt], Size, order="F")

            e = h - S
            MSE = np.zeros(6, dtype=float)
            for cpt in range(6):
                temp = e[:, :, cpt]
                MSE[cpt] = np.sum(temp**2) / np.prod(Size)
            TotalMSE = P1["ems"]

            # Console output (MATLAB style)
            print("MSE =", MSE)
            print("TotalMSE =", TotalMSE)

            # Display 1: originals
            fig1, ax1 = plt.subplots(2, 3, figsize=(12, 8))
            ax1 = ax1.ravel()
            info = [
                "visible blue",
                "visible green",
                "visible red",
                "near infra red",
                "middle infra red",
                "thermal infra red",
            ]
            for cpt in range(6):
                ax1[cpt].imshow(S[:, :, cpt], cmap="gray", vmin=0, vmax=255)
                ax1[cpt].set_title(f"Original, {info[cpt]}")
                ax1[cpt].axis("off")
            fig1.tight_layout()
            fig1.savefig("Figure1238.png")

            # Display 2: covariance/eigen info
            fig2, ax2 = plt.subplots(2, 3, figsize=(12, 8))
            ax2 = ax2.ravel()

            ax2[0].imshow(P["Cx"], cmap="gray")
            ax2[0].set_title("P.Cx, complete")
            ax2[0].axis("image")

            ax2[1].imshow(P["Cy"], cmap="gray")
            ax2[1].set_title("P.Cy, complete")
            ax2[1].axis("image")

            ax2[2].bar(np.arange(1, len(d) + 1), d)
            ax2[2].set_title("The 6 eigenvalues")
            ax2[2].axis("tight")

            ax2[3].plot(P["A"].T)
            ax2[3].set_title("The 6 eigenvectors")
            ax2[3].axis("tight")

            ax2[4].imshow(P1["Cy"], cmap="gray")
            ax2[4].set_title(f"P.Cy, PCAKeep = {PCAKeep}")
            ax2[4].axis("image")
            # Keep a simple tick-label mimic of MATLAB code.
            if P1["Cy"].shape[0] == 2:
                ax2[4].set_xticks([0, 1])
                ax2[4].set_xticklabels(["1", "100"])

            ax2[5].plot(P1["A"].T)
            ax2[5].set_title(f"The {PCAKeep} most significant eigenvectors")
            ax2[5].axis("tight")

            fig2.tight_layout()
            fig2.savefig("Figure1238Bis.png")

            # Display 3: PCA6 components
            fig3, ax3 = plt.subplots(2, 3, figsize=(12, 8))
            ax3 = ax3.ravel()
            for cpt in range(6):
                ax3[cpt].imshow(g[:, :, cpt], cmap="gray")
                ax3[cpt].set_title(f"PCA6, $\\lambda_{{{cpt + 1}}}$ = {d[cpt]:.3g}")
                ax3[cpt].axis("off")
            fig3.tight_layout()
            fig3.savefig("Figure1240.png")

            # Display 4: reconstructed with q=2
            fig4, ax4 = plt.subplots(2, 3, figsize=(12, 8))
            ax4 = ax4.ravel()
            for cpt in range(6):
                ax4[cpt].imshow(h[:, :, cpt], cmap="gray")
                ax4[cpt].set_title(f"Rec_PCA, q = {PCAKeep}")
                ax4[cpt].axis("off")
            fig4.tight_layout()
            fig4.savefig("Figure1241.png")

            # Display 5: reconstruction errors
            fig5, ax5 = plt.subplots(2, 3, figsize=(12, 8))
            ax5 = ax5.ravel()
            for cpt in range(6):
                ax5[cpt].imshow(e[:, :, cpt], cmap="gray")
                ax5[cpt].set_title(f"e_PCA, q = {PCAKeep}, MSE = {MSE[cpt]:.1g}")
                ax5[cpt].axis("off")
            fig5.tight_layout()
            fig5.savefig("Figure1242.png")

            print(
                "Saved Figure1238.png, Figure1238Bis.png, Figure1240.png, Figure1241.png, Figure1242.png"
            )
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1246(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1246.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy.ndimage import correlate
            from skimage.util import random_noise

            print("Running Figure1246 (Spread of derivatives)...")

            # Data
            fconstOrig = 0.75 * np.ones((600, 600), dtype=float)

            fedgeOrig = 0.75 * np.ones((600, 600), dtype=float)
            fedgeOrig[300:, :] = 0.0  # MATLAB: end/2:end

            fcrOrig = 0.75 * np.ones((600, 600), dtype=float)
            fcrOrig[300:, :300] = 0.0  # MATLAB: end/2:end,1:end/2

            # Add noise (Gaussian variance = .003)
            fconst = random_noise(fconstOrig, mode="gaussian", mean=0.0, var=0.003)
            fedge = random_noise(fedgeOrig, mode="gaussian", mean=0.0, var=0.003)
            fcr = random_noise(fcrOrig, mode="gaussian", mean=0.0, var=0.003)

            # Select areas (MATLAB 1-based inclusive: 285:315)
            xlow, xhigh = 285, 315
            ylow, yhigh = 285, 315
            fconst = fconst[xlow - 1 : xhigh, ylow - 1 : yhigh]
            fedge = fedge[xlow - 1 : xhigh, ylow - 1 : yhigh]
            fcr = fcr[xlow - 1 : xhigh, ylow - 1 : yhigh]

            # Filter kernels
            wy = np.array([-1.0, 0.0, 1.0], dtype=float).reshape(1, 3)
            wx = wy.T

            # Derivatives (MATLAB imfilter default correlation, zero padding)
            dxconst = correlate(fconst, wx, mode="constant", cval=0.0)
            dxedge = correlate(fedge, wx, mode="constant", cval=0.0)
            dxcr = correlate(fcr, wx, mode="constant", cval=0.0)

            dyconst = correlate(fconst, wy, mode="constant", cval=0.0)
            dyedge = correlate(fedge, wy, mode="constant", cval=0.0)
            dycr = correlate(fcr, wy, mode="constant", cval=0.0)

            # Strip borders: MATLAB 3:end-3 => Python 2:-3
            dxconst = dxconst[2:-3, 2:-3]
            dxedge = dxedge[2:-3, 2:-3]
            dxcr = dxcr[2:-3, 2:-3]
            dyconst = dyconst[2:-3, 2:-3]
            dyedge = dyedge[2:-3, 2:-3]
            dycr = dycr[2:-3, 2:-3]

            # Convert to vectors
            dxconstv = dxconst.ravel()
            dxedgev = dxedge.ravel()
            dxcrv = dxcr.ravel()
            dyconstv = dyconst.ravel()
            dyedgev = dyedge.ravel()
            dycrv = dycr.ravel()

            # Display
            fig, ax = plt.subplots(2, 3, figsize=(12, 8))
            ax = ax.ravel()

            ax[0].imshow(fconstOrig, cmap="gray", vmin=0, vmax=1)
            ax[0].set_title("No line")
            ax[0].axis("off")

            ax[1].imshow(fedgeOrig, cmap="gray", vmin=0, vmax=1)
            ax[1].set_title("One line")
            ax[1].axis("off")

            ax[2].imshow(fcrOrig, cmap="gray", vmin=0, vmax=1)
            ax[2].set_title("Two lines")
            ax[2].axis("off")

            ax[3].plot(dyconstv, dxconstv, "k.", markersize=2)
            ax[3].set_xlabel("f_{xx}")
            ax[3].set_ylabel("f_{yy}")
            ax[3].axis([-1, 1, -1, 1])

            ax[4].plot(dyedgev, dxedgev, "k.", markersize=2)
            ax[4].set_xlabel("f_{xx}")
            ax[4].set_ylabel("f_{yy}")
            ax[4].axis([-1, 1, -1, 1])

            ax[5].plot(dycrv, dxcrv, "k.", markersize=2)
            ax[5].set_xlabel("f_{xx}")
            ax[5].set_ylabel("f_{yy}")
            ax[5].axis([-1, 1, -1, 1])

            fig.tight_layout()
            fig.savefig("Figure1246.png")

            print("Saved Figure1246.png")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1247(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1247.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            """Figure 12.47 - Corner detection sensitivity and quality level."""

            import os
            import numpy as np
            import matplotlib.pyplot as plt
            from helpers.libgeneral.corner import corner
            from helpers.libdipum.data_path import dip_data

            def im2double(arr: np.ndarray) -> np.ndarray:
                """im2double."""
                a = np.asarray(arr)
                if np.issubdtype(a.dtype, np.floating):
                    return a.astype(np.float64)
                if np.issubdtype(a.dtype, np.integer):
                    info = np.iinfo(a.dtype)
                    if info.min < 0:
                        return (a.astype(np.float64) - info.min) / (info.max - info.min)
                    return a.astype(np.float64) / info.max
                if a.dtype == np.bool_:
                    return a.astype(np.float64)
                return a.astype(np.float64)

            print("Running Figure1247...")

            # Data
            img_path = dip_data("checkerboard-noisy1.tif")
            In1 = im2double(plt.imread(img_path))

            # Corner detection
            C1 = corner(In1)
            C2 = corner(In1, SensitivityFactor=0.1)
            C3 = corner(In1, SensitivityFactor=0.1, QualityLevel=0.1)
            C4 = corner(In1, QualityLevel=0.1)
            C5 = corner(In1, QualityLevel=0.3)

            # Display
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            axes = axes.ravel()

            for ax in axes:
                ax.imshow(In1, cmap="gray", vmin=0, vmax=1)
                ax.axis("off")

            axes[1].plot(C1[:, 0] - 1, C1[:, 1] - 1, "yo", markersize=4)
            axes[2].plot(C2[:, 0] - 1, C2[:, 1] - 1, "yo", markersize=4)
            axes[3].plot(C3[:, 0] - 1, C3[:, 1] - 1, "yo", markersize=4)
            axes[4].plot(C4[:, 0] - 1, C4[:, 1] - 1, "yo", markersize=4)
            axes[5].plot(C5[:, 0] - 1, C5[:, 1] - 1, "yo", markersize=4)

            fig.tight_layout()
            out_path = _os.path.join(_os.environ.get("DIP4E_OUTPUT_DIR", str(_Path(__file__).resolve().parents[2] / "output")), "Figure1247.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved {out_path}")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1248(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1248.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            """Figure 12.48 - Corner detection on noisy checkerboard #2."""

            import os
            import numpy as np
            import matplotlib.pyplot as plt
            from helpers.libgeneral.corner import corner
            from helpers.libdipum.data_path import dip_data

            def im2double(arr: np.ndarray) -> np.ndarray:
                """im2double."""
                a = np.asarray(arr)
                if np.issubdtype(a.dtype, np.floating):
                    return a.astype(np.float64)
                if np.issubdtype(a.dtype, np.integer):
                    info = np.iinfo(a.dtype)
                    if info.min < 0:
                        return (a.astype(np.float64) - info.min) / (info.max - info.min)
                    return a.astype(np.float64) / info.max
                if a.dtype == np.bool_:
                    return a.astype(np.float64)
                return a.astype(np.float64)

            print("Running Figure1248...")

            # Data
            img_path = dip_data("checkerboard-noisy2.tif")
            In2 = im2double(plt.imread(img_path))

            # Corner detection
            C1 = corner(In2)  # default k=0.04, QualityLevel=0.01
            C2 = corner(In2, SensitivityFactor=0.249)
            C3 = corner(In2, QualityLevel=0.15)

            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            axes = axes.ravel()

            for ax in axes:
                ax.imshow(In2, cmap="gray", vmin=0, vmax=1)
                ax.axis("off")

            axes[1].plot(C1[:, 0] - 1, C1[:, 1] - 1, "yo", markersize=4)
            axes[2].plot(C2[:, 0] - 1, C2[:, 1] - 1, "yo", markersize=4)
            axes[3].plot(C3[:, 0] - 1, C3[:, 1] - 1, "yo", markersize=4)

            fig.tight_layout()
            out_path = _os.path.join(_os.environ.get("DIP4E_OUTPUT_DIR", str(_Path(__file__).resolve().parents[2] / "output")), "Figure1248.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved {out_path}")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1249(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1249.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            """Figure 12.49 - Harris corner detection on building image."""

            import os
            import numpy as np
            import matplotlib.pyplot as plt
            from helpers.libgeneral.corner import corner
            from helpers.libdipum.data_path import dip_data

            def im2double(arr: np.ndarray) -> np.ndarray:
                """im2double."""
                a = np.asarray(arr)
                if np.issubdtype(a.dtype, np.floating):
                    return a.astype(np.float64)
                if np.issubdtype(a.dtype, np.integer):
                    info = np.iinfo(a.dtype)
                    if info.min < 0:
                        return (a.astype(np.float64) - info.min) / (info.max - info.min)
                    return a.astype(np.float64) / info.max
                if a.dtype == np.bool_:
                    return a.astype(np.float64)
                return a.astype(np.float64)

            print("Running Figure1249...")

            # Data
            img_path = dip_data("building-600by600.tif")
            IB = im2double(plt.imread(img_path))

            # Harris corners
            C1 = corner(IB)
            C2 = corner(IB, SensitivityFactor=0.249)
            C3 = corner(IB, SensitivityFactor=0.17, QualityLevel=0.05)
            C4 = corner(IB, QualityLevel=0.05)
            C5 = corner(IB, QualityLevel=0.07)

            # Display
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            axes = axes.ravel()

            for ax in axes:
                ax.imshow(IB, cmap="gray", vmin=0, vmax=1)
                ax.axis("off")

            axes[1].plot(C1[:, 0] - 1, C1[:, 1] - 1, "yo", markersize=4)
            axes[2].plot(C2[:, 0] - 1, C2[:, 1] - 1, "yo", markersize=4)
            axes[3].plot(C3[:, 0] - 1, C3[:, 1] - 1, "yo", markersize=4)
            axes[4].plot(C4[:, 0] - 1, C4[:, 1] - 1, "yo", markersize=4)
            axes[5].plot(C5[:, 0] - 1, C5[:, 1] - 1, "yo", markersize=4)

            fig.tight_layout()
            out_path = _os.path.join(_os.environ.get("DIP4E_OUTPUT_DIR", str(_Path(__file__).resolve().parents[2] / "output")), "Figure1249.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved {out_path}")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1250(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1250.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            """Figure 12.50 - Rotated building with Harris corners."""

            import os
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy import ndimage
            from helpers.libgeneral.corner import corner
            from helpers.libdipum.data_path import dip_data

            def im2double(arr: np.ndarray) -> np.ndarray:
                """im2double."""
                a = np.asarray(arr)
                if np.issubdtype(a.dtype, np.floating):
                    return a.astype(np.float64)
                if np.issubdtype(a.dtype, np.integer):
                    info = np.iinfo(a.dtype)
                    if info.min < 0:
                        return (a.astype(np.float64) - info.min) / (info.max - info.min)
                    return a.astype(np.float64) / info.max
                if a.dtype == np.bool_:
                    return a.astype(np.float64)
                return a.astype(np.float64)

            print("Running Figure1250...")

            # Data
            img_path = dip_data("building-600by600.tif")
            IB = im2double(plt.imread(img_path))

            # Rotation (uncropped / loose)
            IBR = ndimage.rotate(
                IB, 5.0, reshape=True, order=1, mode="constant", cval=0.0
            )

            # Cropping to remove black borders introduced by rotation.
            # MATLAB indices: IBR(55:596,53:591)
            IBRc = IBR[54:596, 52:591]

            # Corners with same settings as Figure 12.49 last panel.
            C = corner(IBRc, QualityLevel=0.07)

            # Display
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            for ax in axes:
                ax.imshow(IBRc, cmap="gray", vmin=0, vmax=1)
                ax.axis("off")

            axes[1].plot(C[:, 0] - 1, C[:, 1] - 1, "yo", markersize=4)

            fig.tight_layout()
            out_path = _os.path.join(_os.environ.get("DIP4E_OUTPUT_DIR", str(_Path(__file__).resolve().parents[2] / "output")), "Figure1250.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved {out_path}")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1252(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1252.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            """Figure 12.52 - MSER of head CT."""

            import os
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy import ndimage
            import imageio.v2 as iio

            from helpers.libgeneral.detectMSERFeatures import detectMSERFeatures
            from helpers.libdipum.data_path import dip_data

            def _to_gray(a: np.ndarray) -> np.ndarray:
                """_to_gray."""
                arr = np.asarray(a)
                if arr.ndim == 2:
                    return arr
                if arr.ndim == 3:
                    if arr.shape[2] == 1:
                        return arr[..., 0]
                    return (
                        0.2989 * arr[..., 0]
                        + 0.5870 * arr[..., 1]
                        + 0.1140 * arr[..., 2]
                    ).astype(arr.dtype)
                raise ValueError("Input image must be 2-D grayscale or 3-D color")

            def _linear_idx_to_mask(
                pixel_idx_1based: np.ndarray, shape_hw: tuple[int, int]
            ) -> np.ndarray:
                """Convert MATLAB 1-based column-major linear indices to uint8 mask."""
                H, W = shape_hw
                out = np.zeros((H, W), dtype=np.uint8)
                idx0 = np.asarray(pixel_idx_1based, dtype=np.int64).ravel() - 1
                idx0 = idx0[(idx0 >= 0) & (idx0 < H * W)]
                if idx0.size == 0:
                    return out
                rr, cc = np.unravel_index(idx0, (H, W), order="F")
                out[rr, cc] = 255
                return out

            print("Running Figure1252 (MSER of head CT)...")

            # Data
            img_path = dip_data("headCT.tif")
            I = _to_gray(iio.imread(img_path))

            # Kernel
            w = np.ones((15, 15), dtype=np.float64)
            w = w / np.sum(w)

            # Filtering (replicate boundary)
            Is = ndimage.convolve(I, w, mode="nearest")

            # Maximally Stable External Region
            R, RCC = detectMSERFeatures(
                Is,
                "ThresholdDelta",
                2.5,
                "RegionAreaRange",
                [10260, 34200],
            )

            CC = RCC["PixelIdxList"]
            NCC = len(CC)
            Iunion = np.zeros_like(I, dtype=np.uint8)

            fig = plt.figure(figsize=(10, 8))

            ax1 = fig.add_subplot(2, 2, 1)
            ax1.imshow(I, cmap="gray")
            ax1.set_title("I")
            ax1.axis("off")

            ax2 = fig.add_subplot(2, 2, 2)
            ax2.imshow(Is, cmap="gray")
            ax2.set_title("Is")
            ax2.axis("off")

            for K in range(NCC):
                Idisp = _linear_idx_to_mask(CC[K], I.shape)
                Iunion = np.bitwise_or(Iunion, Idisp)
                subplot_idx = 3 + K
                if subplot_idx > 4:
                    break
                ax = fig.add_subplot(2, 2, subplot_idx)
                ax.imshow(Idisp, cmap="gray")
                ax.set_title(f"Region {K + 1}")
                ax.axis("off")

            fig.tight_layout()
            out_path = _os.path.join(_os.environ.get("DIP4E_OUTPUT_DIR", str(_Path(__file__).resolve().parents[2] / "output")), "Figure1252.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Detected regions: {R.Count}")
            print(f"Saved {out_path}")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1253(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1253.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            """Figure 12.53 - MSER of building."""

            import os
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy import ndimage
            import imageio.v2 as iio

            from helpers.libgeneral.detectMSERFeatures import detectMSERFeatures
            from helpers.libdipum.data_path import dip_data

            def _to_gray(a: np.ndarray) -> np.ndarray:
                """_to_gray."""
                arr = np.asarray(a)
                if arr.ndim == 2:
                    return arr
                if arr.ndim == 3:
                    if arr.shape[2] == 1:
                        return arr[..., 0]
                    return (
                        0.2989 * arr[..., 0]
                        + 0.5870 * arr[..., 1]
                        + 0.1140 * arr[..., 2]
                    ).astype(arr.dtype)
                raise ValueError("Input image must be 2-D grayscale or 3-D color")

            def _linear_idx_to_mask(
                pixel_idx_1based: np.ndarray, shape_hw: tuple[int, int]
            ) -> np.ndarray:
                """Convert MATLAB 1-based column-major linear indices to uint8 mask."""
                H, W = shape_hw
                out = np.zeros((H, W), dtype=np.uint8)
                idx0 = np.asarray(pixel_idx_1based, dtype=np.int64).ravel() - 1
                idx0 = idx0[(idx0 >= 0) & (idx0 < H * W)]
                if idx0.size == 0:
                    return out
                rr, cc = np.unravel_index(idx0, (H, W), order="F")
                out[rr, cc] = 255
                return out

            print("Running Figure1253 (MSER of building)...")

            # Data
            img_path = dip_data("building-600by600.tif")
            I = _to_gray(iio.imread(img_path))

            # Kernel
            w = np.ones((5, 5), dtype=np.float64)
            w = w / np.sum(w)

            # Filtering (replicate boundary)
            Is = ndimage.convolve(I, w, mode="nearest")

            # MSER
            R, RCC = detectMSERFeatures(
                Is,
                "ThresholdDelta",
                0.4,
                "RegionAreaRange",
                [10000, 30000],
            )

            # Display all images.
            CC = RCC["PixelIdxList"]
            NCC = len(CC)
            Iunion = np.zeros_like(I, dtype=np.uint8)

            fig = plt.figure(figsize=(12, 8))
            for K in range(NCC):
                Idisp = _linear_idx_to_mask(CC[K], I.shape)
                Iunion = np.bitwise_or(Iunion, Idisp)

                subplot_idx = K + 1  # MATLAB: subplot(2,3,K)
                if subplot_idx > 6:
                    break
                ax = fig.add_subplot(2, 3, subplot_idx)
                ax.imshow(Idisp, cmap="gray")
                ax.set_title(f"Region {K + 1}")
                ax.axis("off")

            ax = fig.add_subplot(2, 3, 6)  # MATLAB: subplot(2,3,6)
            ax.imshow(Iunion, cmap="gray")
            ax.set_title("Union")
            ax.axis("off")

            fig.tight_layout()
            out_path = _os.path.join(_os.environ.get("DIP4E_OUTPUT_DIR", str(_Path(__file__).resolve().parents[2] / "output")), "Figure1253.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Detected regions: {R.Count}")
            print(f"Saved {out_path}")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1254(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1254.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            """Figure 12.54 - MSER of rotated building."""

            import os
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy import ndimage
            import imageio.v2 as iio

            from helpers.libgeneral.detectMSERFeatures import detectMSERFeatures
            from helpers.libdipum.data_path import dip_data

            def _to_gray(a: np.ndarray) -> np.ndarray:
                """_to_gray."""
                arr = np.asarray(a)
                if arr.ndim == 2:
                    return arr
                if arr.ndim == 3:
                    if arr.shape[2] == 1:
                        return arr[..., 0]
                    return (
                        0.2989 * arr[..., 0]
                        + 0.5870 * arr[..., 1]
                        + 0.1140 * arr[..., 2]
                    ).astype(arr.dtype)
                raise ValueError("Input image must be 2-D grayscale or 3-D color")

            def _to_uint8_graylevels(a: np.ndarray) -> np.ndarray:
                """_to_uint8_graylevels."""
                arr = np.asarray(a)
                if arr.dtype == np.uint8:
                    return arr
                out = arr.astype(np.float64)
                if out.size == 0:
                    return np.zeros_like(out, dtype=np.uint8)
                if np.max(out) <= 1.0:
                    out = out * 255.0
                return np.uint8(np.clip(np.round(out), 0.0, 255.0))

            def _linear_idx_to_mask(
                pixel_idx_1based: np.ndarray, shape_hw: tuple[int, int]
            ) -> np.ndarray:
                """Convert MATLAB 1-based column-major linear indices to uint8 mask."""
                H, W = shape_hw
                out = np.zeros((H, W), dtype=np.uint8)
                idx0 = np.asarray(pixel_idx_1based, dtype=np.int64).ravel() - 1
                idx0 = idx0[(idx0 >= 0) & (idx0 < H * W)]
                if idx0.size == 0:
                    return out
                rr, cc = np.unravel_index(idx0, (H, W), order="F")
                out[rr, cc] = 255
                return out

            print("Running Figure1254 (MSER of rotated building)...")

            # Data
            img_path = dip_data("building-600by600.tif")
            I = _to_gray(iio.imread(img_path))
            Ir = ndimage.rotate(
                I, 5.0, reshape=True, order=1, mode="constant", cval=0.0
            )

            # Crop (MATLAB: Ir(55:596,53:591))
            Irc = Ir[54:596, 52:591]

            # Kernel
            w = np.ones((5, 5), dtype=np.float64)
            w = w / np.sum(w)

            # Filtering
            Ircs = ndimage.convolve(Irc, w, mode="nearest")
            Ircs_u8 = _to_uint8_graylevels(Ircs)

            # MSER
            R, RCC = detectMSERFeatures(
                Ircs_u8,
                "ThresholdDelta",
                0.4,
                "RegionAreaRange",
                [10000, 30000],
            )

            # Display all the images.
            CC = RCC["PixelIdxList"]
            NCC = len(CC)
            Iunion = np.zeros_like(Ircs_u8, dtype=np.uint8)
            for K in range(NCC):
                Idisp = _linear_idx_to_mask(CC[K], Ircs_u8.shape)
                Iunion = np.bitwise_or(Iunion, Idisp)

            # Display
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(Ir, cmap="gray")
            axs[0].set_title("rotated")
            axs[0].axis("off")

            axs[1].imshow(Ircs_u8, cmap="gray")
            axs[1].set_title("cropped")
            axs[1].axis("off")

            axs[2].imshow(Iunion, cmap="gray")
            axs[2].set_title("MSER (union)")
            axs[2].axis("off")

            fig.tight_layout()
            out_path = _os.path.join(_os.environ.get("DIP4E_OUTPUT_DIR", str(_Path(__file__).resolve().parents[2] / "output")), "Figure1254.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Detected regions: {R.Count}")
            print(f"Saved {out_path}")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1255(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1255.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            """Figure 12.55 - MSER of half-size building image."""

            import os
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy import ndimage
            import imageio.v2 as iio

            from helpers.libgeneral.detectMSERFeatures import detectMSERFeatures
            from helpers.libdipum.data_path import dip_data

            def _to_gray(a: np.ndarray) -> np.ndarray:
                """_to_gray."""
                arr = np.asarray(a)
                if arr.ndim == 2:
                    return arr
                if arr.ndim == 3:
                    if arr.shape[2] == 1:
                        return arr[..., 0]
                    return (
                        0.2989 * arr[..., 0]
                        + 0.5870 * arr[..., 1]
                        + 0.1140 * arr[..., 2]
                    ).astype(arr.dtype)
                raise ValueError("Input image must be 2-D grayscale or 3-D color")

            def _to_uint8_graylevels(a: np.ndarray) -> np.ndarray:
                """_to_uint8_graylevels."""
                arr = np.asarray(a)
                if arr.dtype == np.uint8:
                    return arr
                out = arr.astype(np.float64)
                if out.size == 0:
                    return np.zeros_like(out, dtype=np.uint8)
                if np.max(out) <= 1.0:
                    out = out * 255.0
                return np.uint8(np.clip(np.round(out), 0.0, 255.0))

            def _linear_idx_to_mask(
                pixel_idx_1based: np.ndarray, shape_hw: tuple[int, int]
            ) -> np.ndarray:
                """Convert MATLAB 1-based column-major linear indices to uint8 mask."""
                H, W = shape_hw
                out = np.zeros((H, W), dtype=np.uint8)
                idx0 = np.asarray(pixel_idx_1based, dtype=np.int64).ravel() - 1
                idx0 = idx0[(idx0 >= 0) & (idx0 < H * W)]
                if idx0.size == 0:
                    return out
                rr, cc = np.unravel_index(idx0, (H, W), order="F")
                out[rr, cc] = 255
                return out

            print("Running Figure1255 (MSER of half-size building image)...")

            # Data
            img_path = dip_data("building-600by600.tif")
            I = _to_gray(iio.imread(img_path))

            # Quarter size in pixels (half width and height): I(1:2:end, 1:2:end)
            I = I[::2, ::2]

            # Kernel
            w = np.ones((3, 3), dtype=np.float64)
            w = w / np.sum(w)

            # Filtering
            Is = ndimage.convolve(I, w, mode="nearest")
            Is_u8 = _to_uint8_graylevels(Is)

            # MSER
            R, RCC = detectMSERFeatures(
                Is_u8,
                "ThresholdDelta",
                0.7,
                "RegionAreaRange",
                [2500, 7500],
            )

            CC = RCC["PixelIdxList"]
            NCC = len(CC)
            Iunion = np.zeros_like(Is_u8, dtype=np.uint8)
            for K in range(NCC):
                Idisp = _linear_idx_to_mask(CC[K], Is_u8.shape)
                Iunion = np.bitwise_or(Iunion, Idisp)

            # Display
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(I, cmap="gray")
            axs[0].set_title("I half size")
            axs[0].axis("off")

            axs[1].imshow(Is_u8, cmap="gray")
            axs[1].set_title("Is")
            axs[1].axis("off")

            axs[2].imshow(Iunion, cmap="gray")
            axs[2].set_title("MSER")
            axs[2].axis("off")

            fig.tight_layout()
            out_path = _os.path.join(_os.environ.get("DIP4E_OUTPUT_DIR", str(_Path(__file__).resolve().parents[2] / "output")), "Figure1255.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Detected regions: {R.Count}")
            print(f"Saved {out_path}")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1257(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1257.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            """Figure 12.57 - Gaussian octaves without white padded frames."""

            import os
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy import ndimage
            from helpers.libdip.gaussKernel4e import gaussKernel4e
            from helpers.libdipum.data_path import dip_data

            print("Running Figure1257...")

            # Parameters
            sigma_first_oct = np.sqrt(2.0) / 2.0
            sigma_second_oct = 2.0 * sigma_first_oct
            sigma_third_oct = 2.0 * sigma_second_oct
            k = np.sqrt(2.0)
            images_per_octave = 5

            # Data
            img_path = dip_data("building-600by600.tif")
            f = plt.imread(img_path)
            if f.ndim == 3:
                f = f[..., 0]
            f = np.asarray(f)

            nr, nc = f.shape

            # First octave
            size_kernel_first_oct = [5, 7, 9, 13, 17]
            first_octave = np.zeros((nr, nc, images_per_octave), dtype=np.float64)
            les_sigma_first_oct = np.zeros(images_per_octave, dtype=np.float64)

            for i in range(images_per_octave):
                if i == 0:
                    les_sigma_first_oct[i] = sigma_first_oct
                else:
                    les_sigma_first_oct[i] = les_sigma_first_oct[i - 1] * k

                kernel = gaussKernel4e(
                    size_kernel_first_oct[i], les_sigma_first_oct[i], 1
                )
                if i == 0:
                    first_octave[:, :, i] = ndimage.convolve(f, kernel, mode="reflect")
                else:
                    first_octave[:, :, i] = ndimage.convolve(
                        first_octave[:, :, i - 1], kernel, mode="reflect"
                    )

            # Second octave
            size_kernel_second_oct = [9, 13, 17, 25, 35]
            second_base = f[::2, ::2]
            second_octave = np.zeros(
                (second_base.shape[0], second_base.shape[1], images_per_octave),
                dtype=np.float64,
            )
            les_sigma_second_oct = np.zeros(images_per_octave, dtype=np.float64)

            for i in range(images_per_octave):
                if i == 0:
                    les_sigma_second_oct[i] = sigma_second_oct
                else:
                    les_sigma_second_oct[i] = les_sigma_second_oct[i - 1] * k

                kernel = gaussKernel4e(
                    size_kernel_second_oct[i], les_sigma_second_oct[i], 1
                )
                if i == 0:
                    second_octave[:, :, i] = ndimage.convolve(
                        second_base, kernel, mode="reflect"
                    )
                else:
                    second_octave[:, :, i] = ndimage.convolve(
                        second_octave[:, :, i - 1], kernel, mode="reflect"
                    )

            # Third octave
            size_kernel_third_oct = [17, 25, 35, 49, 67]
            third_base = f[::4, ::4]
            third_octave = np.zeros(
                (third_base.shape[0], third_base.shape[1], images_per_octave),
                dtype=np.float64,
            )
            les_sigma_third_oct = np.zeros(images_per_octave, dtype=np.float64)

            for i in range(images_per_octave):
                if i == 0:
                    les_sigma_third_oct[i] = sigma_third_oct
                else:
                    les_sigma_third_oct[i] = les_sigma_third_oct[i - 1] * k

                kernel = gaussKernel4e(
                    size_kernel_third_oct[i], les_sigma_third_oct[i], 1
                )
                if i == 0:
                    third_octave[:, :, i] = ndimage.convolve(
                        third_base, kernel, mode="reflect"
                    )
                else:
                    third_octave[:, :, i] = ndimage.convolve(
                        third_octave[:, :, i - 1], kernel, mode="reflect"
                    )

            # Display helper: place image at top-left in full-size coordinates
            # without padded white image data.
            def show_octave(
                ax: plt.Axes, img: np.ndarray, nr_: int, nc_: int, title: str
            ) -> None:
                """show_octave."""
                ax.imshow(
                    img,
                    cmap="gray",
                    interpolation="nearest",
                    origin="upper",
                    extent=(1, img.shape[1], img.shape[0], 1),
                )
                ax.set_xlim(1, nc_)
                ax.set_ylim(nr_, 1)
                ax.set_facecolor("black")
                ax.axis("off")
                ax.set_title(title)

            fig, axes = plt.subplots(3, images_per_octave, figsize=(18, 10))

            for i in range(images_per_octave):
                show_octave(
                    axes[0, i],
                    first_octave[:, :, i],
                    nr,
                    nc,
                    f"Oct = 1, $\\sigma$ = {les_sigma_first_oct[i]:.3g}",
                )

            for i in range(images_per_octave):
                show_octave(
                    axes[1, i],
                    second_octave[:, :, i],
                    nr,
                    nc,
                    f"Oct = 2, $\\sigma$ = {les_sigma_second_oct[i]:.3g}",
                )

            for i in range(images_per_octave):
                show_octave(
                    axes[2, i],
                    third_octave[:, :, i],
                    nr,
                    nc,
                    f"Oct = 3, $\\sigma$ = {les_sigma_third_oct[i]:.3g}",
                )

            fig.tight_layout()
            out_path = _os.path.join(_os.environ.get("DIP4E_OUTPUT_DIR", str(_Path(__file__).resolve().parents[2] / "output")), "Figure1257.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved {out_path}")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1258(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1258.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            """Figure 12.58 - Difference of Gaussians."""

            import os
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy import ndimage
            from helpers.libdipum.data_path import dip_data

            def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
                """gaussian_kernel."""
                size = int(size)
                if size < 1:
                    size = 1
                # Match fspecial-style centered kernel for odd/even sizes.
                r = (size - 1) / 2.0
                x = np.linspace(-r, r, size)
                y = np.linspace(-r, r, size)
                xx, yy = np.meshgrid(x, y)
                k = np.exp(-(xx**2 + yy**2) / (2.0 * sigma * sigma))
                s = k.sum()
                if s != 0:
                    k = k / s
                return k

            def im2uint8_from_double(a: np.ndarray) -> np.ndarray:
                """im2uint8_from_double."""
                # MATLAB im2uint8 behavior for double input: clamp to [0,1], then scale.
                return np.uint8(np.clip(a, 0.0, 1.0) * 255.0)

            def im2double(a: np.ndarray) -> np.ndarray:
                """im2double."""
                arr = np.asarray(a)
                if np.issubdtype(arr.dtype, np.floating):
                    out = arr.astype(np.float64)
                    # Defensive normalization for float images encoded in [0,255].
                    if out.size and out.max() > 1.0:
                        out = out / 255.0
                    return out
                if np.issubdtype(arr.dtype, np.integer):
                    info = np.iinfo(arr.dtype)
                    if info.min < 0:
                        return (arr.astype(np.float64) - info.min) / (
                            info.max - info.min
                        )
                    return arr.astype(np.float64) / info.max
                if arr.dtype == np.bool_:
                    return arr.astype(np.float64)
                return arr.astype(np.float64)

            print("Running Figure1258...")

            # Parameters
            k = np.sqrt(2.0)
            sdev = k / 2.0
            T = 3

            # Data
            img_path = dip_data("building-600by600.tif")
            f1 = plt.imread(img_path)
            if f1.ndim == 3:
                f1 = f1[..., 0]
            f1 = im2double(f1)

            nr, nc = f1.shape
            f2 = f1[::2, ::2]
            f3 = f2[::2, ::2]

            # Octave 1
            sig = np.zeros(5, dtype=np.float64)
            sig[0] = sdev
            sig[1] = k * sig[0]
            for i in range(2, 5):
                sig[i] = k * sig[i - 1]

            oct1 = np.zeros((f1.shape[0], f1.shape[1], 5), dtype=np.float64)
            for i in range(5):
                w = gaussian_kernel(int(np.ceil(6 * sig[i])), sig[i])
                oct1[:, :, i] = ndimage.convolve(f1, w, mode="nearest")

            # Octave 2
            sig[0] = 2 * sdev
            sig[1] = k * sig[0]
            for i in range(2, 5):
                sig[i] = k * sig[i - 1]

            oct2 = np.zeros((f2.shape[0], f2.shape[1], 5), dtype=np.float64)
            for i in range(5):
                w = gaussian_kernel(int(np.ceil(6 * sig[i])), sig[i])
                oct2[:, :, i] = ndimage.convolve(f2, w, mode="nearest")

            # Octave 3
            sig[0] = 4 * sdev
            sig[1] = k * sig[0]
            for i in range(2, 5):
                sig[i] = k * sig[i - 1]

            oct3 = np.zeros((f3.shape[0], f3.shape[1], 5), dtype=np.float64)
            for i in range(5):
                w = gaussian_kernel(int(np.ceil(3 * sig[i])), sig[i])
                oct3[:, :, i] = ndimage.convolve(f3, w, mode="nearest")

            # Difference of Gaussians
            DoG1 = oct1[:, :, 2] - oct1[:, :, 1]
            DoG2 = oct2[:, :, 2] - oct2[:, :, 1]
            DoG3 = oct3[:, :, 2] - oct3[:, :, 1]

            DoG18 = im2uint8_from_double(DoG1)
            DoG28 = im2uint8_from_double(DoG2)
            DoG38 = im2uint8_from_double(DoG3)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            # Keep the same order as MATLAB code.
            for ax, img in zip(axes, [DoG38 > T, DoG28 > T, DoG18 > T]):
                h, w = img.shape
                ax.imshow(
                    img,
                    cmap="gray",
                    interpolation="nearest",
                    origin="upper",
                    extent=(1, w, h, 1),
                )
                ax.set_xlim(1, nc)
                ax.set_ylim(nr, 1)
                ax.axis("off")

            fig.tight_layout()
            out_path = _os.path.join(_os.environ.get("DIP4E_OUTPUT_DIR", str(_Path(__file__).resolve().parents[2] / "output")), "Figure1258.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved {out_path}")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1260(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1260.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            """Figure 12.60 - Keypoints without orientation arrows."""

            import os
            import numpy as np
            import matplotlib.pyplot as plt
            from helpers.libdip.boundary2image4e import boundary2image4e
            from helpers.libdipum.sift import sift
            import ia870 as ia
            from helpers.libdipum.data_path import dip_data

            print("Running Figure1260New (keypoints without orientation arrows)...")

            # Data
            img_path_pgm = dip_data("building-600by600.pgm")

            # SIFT
            image, descrips, locs = sift(img_path_pgm)

            # Keep the same fixed-index behavior, clipped to available keypoints.
            k = min(643, locs.shape[0])
            rows = locs[:k, 0]
            cols = locs[:k, 1]
            b = np.column_stack((rows, cols))

            keypoints = boundary2image4e(b, 600, 600)

            # Add 4-neighbors to enlarge each keypoint.
            keypoints_Enlarged = ia.iadil(keypoints, ia.iasecross(2))

            # Display
            fig, axs = plt.subplots(1, 2, figsize=(12, 4))
            axs[0].imshow(keypoints, cmap="gray")
            axs[0].set_title(f"Keypoints (k = {k})")
            axs[0].axis("off")

            axs[1].imshow(keypoints_Enlarged, cmap="gray")
            axs[1].set_title(f"Enlarged keypoints (k = {k})")
            axs[1].axis("off")

            fig.tight_layout()
            out_path = _os.path.join(_os.environ.get("DIP4E_OUTPUT_DIR", str(_Path(__file__).resolve().parents[2] / "output")), "Figure1260.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1261(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1261.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            """Figure 12.61 - Keypoints with orientation."""

            import os
            import matplotlib.pyplot as plt

            from helpers.libdipum.sift import sift
            from helpers.libdipum.showkeys import showkeys
            from helpers.libdipum.data_path import dip_data

            print("Running Figure1261 (keypoints with orientation)...")

            # Data
            img_path_pgm = dip_data("building-600by600.pgm")

            # SIFT
            image, descrips, locs = sift(img_path_pgm)
            print(f"Detected keypoints: {locs.shape[0]}")

            # Display
            overlay = showkeys(image, locs)
            plt.figure()
            plt.imshow(overlay)
            plt.title("Keypoints with orientation")
            plt.axis("off")

            out_path = _os.path.join(_os.environ.get("DIP4E_OUTPUT_DIR", str(_Path(__file__).resolve().parents[2] / "output")), "Figure1261.png")
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved {out_path}")
            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1263(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1263.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            """Figure 12.63 - Matching of building corner."""

            import os
            import matplotlib.pyplot as plt

            from helpers.libdipum.sift import sift
            from helpers.libdipum.showkeys import showkeys
            from helpers.libdipum.match import match
            from helpers.libdipum.data_path import dip_data

            print("Running Figure1263 (matching of building corner)...")

            # Data
            img1 = dip_data("building-600by600.pgm")
            img2 = dip_data("building-corner.pgm")

            # Keypoints for building
            image, descrips, locs1 = sift(img1)
            overlay1 = showkeys(image, locs1)
            print(f"Building keypoints found: {locs1.shape[0]}")

            # Keypoints for building corner
            image, descrips, locs2 = sift(img2)
            overlay2 = showkeys(image, locs2)
            print(f"Building-corner keypoints found: {locs2.shape[0]}")

            # Match between the two
            num, overlay_match = match(img1, img2)
            print(f"Found {num} matches")

            # Display
            fig1 = plt.figure(1)
            plt.imshow(overlay1)
            plt.title(f"Building keypoints, {locs1.shape[0]}, keypoints")
            plt.axis("off")

            fig2 = plt.figure(2)
            plt.imshow(overlay2)
            plt.title(f"Building-corner keypoints, {locs2.shape[0]}, keypoints")
            plt.axis("off")

            fig3 = plt.figure(3)
            plt.imshow(overlay_match)
            plt.title(f"Matching, {num}, match")
            plt.axis("off")

            # Save (equivalent to MATLAB print -f1/-f2/-f3)
            out_dir = _os.environ.get("DIP4E_OUTPUT_DIR", str(_Path(__file__).resolve().parents[2] / "output"))
            out1 = os.path.join(out_dir, "Figure1263.png")
            out2 = os.path.join(out_dir, "Figure1263Bis.png")
            out3 = os.path.join(out_dir, "Figure1263Ter.png")
            fig1.savefig(out1, dpi=150, bbox_inches="tight")
            fig2.savefig(out2, dpi=150, bbox_inches="tight")
            fig3.savefig(out3, dpi=150, bbox_inches="tight")
            print(f"Saved {out1}")
            print(f"Saved {out2}")
            print(f"Saved {out3}")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1265(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1265.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            """Figure 12.65 - Matching of half-size building corner."""

            import os
            import matplotlib.pyplot as plt
            import imageio.v2 as iio
            from scipy import ndimage

            from helpers.libdipum.sift import sift
            from helpers.libdipum.showkeys import showkeys
            from helpers.libdipum.match import match
            from helpers.libdipum.data_path import dip_data

            print("Running Figure1265 (matching of half-size building corner)...")

            # Parameters
            ScaleFactor = 0.5

            # Data (rotated image) - kept for parity with MATLAB script
            I = iio.imread(dip_data("building-600by600.tif"))
            if I.ndim == 2:
                Ih = ndimage.zoom(I, ScaleFactor, order=1)
            else:
                Ih = ndimage.zoom(I, (ScaleFactor, ScaleFactor, 1), order=1)
            _ = Ih  # variable intentionally retained to mirror MATLAB flow

            # Keypoints for half-size building
            img1 = dip_data("building-halfsize.pgm")
            image, descrips, locs1 = sift(img1)
            overlay1 = showkeys(image, locs1)

            # Keypoints for half-size building corner
            img2 = dip_data("building-halfsize-corner.pgm")
            image, descrips, locs2 = sift(img2)
            overlay2 = showkeys(image, locs2)

            # Match between the two
            num, overlay_match = match(img1, img2)

            # Display
            fig1 = plt.figure(1)
            plt.imshow(overlay1)
            plt.title(f"Building keypoints, {locs1.shape[0]}, keypoints")
            plt.axis("off")

            fig2 = plt.figure(2)
            plt.imshow(overlay2)
            plt.title(f"Building-corner keypoints, {locs2.shape[0]}, keypoints")
            plt.axis("off")

            fig3 = plt.figure(3)
            plt.imshow(overlay_match)
            plt.title(f"Matching, {num}, match")
            plt.axis("off")

            # Save
            out_dir = _os.environ.get("DIP4E_OUTPUT_DIR", str(_Path(__file__).resolve().parents[2] / "output"))
            out1 = os.path.join(out_dir, "Figure1265.png")
            out2 = os.path.join(out_dir, "Figure1265Bis.png")
            out3 = os.path.join(out_dir, "Figure1265Ter.png")
            fig1.savefig(out1, dpi=150, bbox_inches="tight")
            fig2.savefig(out2, dpi=150, bbox_inches="tight")
            fig3.savefig(out3, dpi=150, bbox_inches="tight")
            print(f"Saved {out1}")
            print(f"Saved {out2}")
            print(f"Saved {out3}")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

    def figure1266(self, data_dir: str | None = None) -> dict[str, Any]:
        """Run Chapter12 script `Figure1266.py` with inlined code."""
        _ctx, pre_fig_nums, script_path = self._prepare_script_context(data_dir=data_dir)
        try:
            """Figure 12.66 - Match rotated and half-size corners against original building."""

            import os
            import matplotlib.pyplot as plt

            from helpers.libdipum.match import match
            from helpers.libdipum.data_path import dip_data

            print("Running Figure1266 (matching rotated and half-size corners)...")

            img_base = dip_data("building-600by600.pgm")
            img_rot_corner = dip_data("building-rot-corner.pgm")
            img_half_corner = dip_data("building-halfsize-corner.pgm")

            # Match rotated corner against original building
            num1, match1 = match(img_base, img_rot_corner)

            # Match half-size corner against original building
            num2, match2 = match(img_base, img_half_corner)

            # Display
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            axs[0].imshow(match1)
            axs[0].set_title(f"Matching, {num1}, match")
            axs[0].axis("off")

            axs[1].imshow(match2)
            axs[1].set_title(f"Matching, {num2}, match")
            axs[1].axis("off")

            fig.tight_layout()
            out_path = _os.path.join(_os.environ.get("DIP4E_OUTPUT_DIR", str(_Path(__file__).resolve().parents[2] / "output")), "Figure1266.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved {out_path}")

            plt.show()
        finally:
            self._restore_script_context(_ctx, data_dir=data_dir)
        return self._collect_new_figures(pre_fig_nums)

