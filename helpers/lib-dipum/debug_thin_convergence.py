import numpy as np
from morphoHitmiss4e import morphoHitmiss4e


def debug_thin_convergence():
    """debug_thin_convergence."""
    # Setup kernels same as morphoThin4e Logic
    B = np.zeros((3, 3, 8), dtype=int)
    B[:, :, 0] = np.array([[0, 0, 0], [2, 1, 2], [1, 1, 1]])
    B[:, :, 1] = np.array([[2, 0, 0], [1, 1, 0], [1, 1, 2]])
    for k in range(2, 8):
        B[:, :, k] = np.rot90(B[:, :, k - 2], k=-1)  # Logic reused from mine

    # Create Image: 40x40. Filled Circle Radius 15.
    H, W = 40, 40
    y, x = np.ogrid[:H, :W]
    I = np.zeros((H, W))
    mask = (x - 20) ** 2 + (y - 20) ** 2 <= 15**2
    I[mask] = 1

    currentH = I.copy()
    max_iter = 100

    print(f"Start Area: {np.sum(currentH)}")

    for folder_iter in range(max_iter):
        change_in_folder = False

        # Cycle B
        for k in range(8):
            currentB = B[:, :, k]

            # HitMiss
            # We implemented HM separately
            HM = morphoHitmiss4e(currentH, currentB)

            # Remove hits
            # matches = (HM == 1)
            # if any match, change occurs
            # Logic: T = currentH & ~matches

            mask_rem = HM == 1
            rem_count = np.sum(mask_rem)

            if rem_count > 0:
                change_in_folder = True
                # Update
                currentH[mask_rem] = 0

        area = np.sum(currentH)
        print(f"Iter {folder_iter + 1}: Area {area}")

        if not change_in_folder:
            print("Converged.")
            break

    # Final check
    # Circle skeleton should be single point or empty?
    # Topological thinning preserves topology.
    # Circle without holes -> Point?
    # Ring -> Loop.
    # Filled Circle -> Point.

    if area > 10:
        print("FAIL: Stopped with large area.")
    else:
        print("PASS: Thinned to small skeleton.")


if __name__ == "__main__":
    debug_thin_convergence()
