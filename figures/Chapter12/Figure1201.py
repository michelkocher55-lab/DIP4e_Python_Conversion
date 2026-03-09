from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from General.N8 import N8
from General.mmshow import mmshow
from libDIPUM.freemanChainCode import freemanChainCode
import sys
from libDIPUM.data_path import dip_data


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


if __name__ == "__main__":
    # Check args
    if len(sys.argv) > 1:
        try:
            val = int(sys.argv[1])
            Figure1201(val)
        except:
            Figure1201()
    else:
        Figure1201()
