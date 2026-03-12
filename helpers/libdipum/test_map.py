import numpy as np
import matplotlib.pyplot as plt
from helpers.libdip.snakeMap4e import snakeMap4e


def test_map():
    """test_map."""
    # Synthetic Image: 100x100.
    # Black background (0).
    # White Circle (1) radius 30 at 50,50.

    I = np.zeros((100, 100))
    y, x = np.ogrid[:100, :100]
    mask = (x - 50) ** 2 + (y - 50) ** 2 <= 30**2
    I[mask] = 1.0

    # Run snakeMap4e
    # T='auto' or T=0.5.
    # Note: Figure112 uses T=0.001.
    # But synthetic image is clean. T=0.5 is fine.
    # Let's try T=0.5 first.

    print("Running snakeMap4e on Circle...")
    emap = snakeMap4e(I, T=0.5, sig=1, nsig=3, order="both")

    print(f"Emap Stat: Min={emap.min()}, Max={emap.max()}, Mean={emap.mean()}")

    # Check Corner (0,0). Should be Black (0).
    print(f"Corner Value: {emap[0, 0]}")
    # Check Edge (Pixel near 50, 20). Should be White (1 or dispersed).
    # Check Center (50,50). Should be Black (0) usually (since gradient is 0 inside constant regions).
    print(f"Center Value: {emap[50, 50]}")

    # Save/Show
    plt.figure()
    plt.imshow(emap, cmap="gray")
    plt.title("Test Map Output")
    plt.show()  # In non-interactive this does nothing helpful for me except verification context if I could see it.

    # Logic verification
    if emap.mean() > 0.5:
        print("WARNING: Image appears mostly WHITE. (Inverted?)")
    else:
        print("OK: Image appears mostly DARK. (Correct)")


if __name__ == "__main__":
    test_map()
