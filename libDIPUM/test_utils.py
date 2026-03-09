import numpy as np
from imPad4e import imPad4e
from twodConv4e import twodConv4e


def test_utils():
    """test_utils."""
    # Test 1: imPad4e
    print("Testing imPad4e...")
    I = np.zeros((5, 5))
    I[2, 2] = 1  # Center point

    # Pad by 2
    # Replicate 0s
    P = imPad4e(I, 2, 2, "replicate", "both")
    print(f"P shape: {P.shape}")
    print(f"P mean: {P.mean()} (Should be small, not near 1)")
    print(P)

    if P.mean() > 0.5:
        print("FAIL: Padding created massive 1s?")
    else:
        print("PASS: Padding seems correct.")

    # Test 2: twodConv4e
    print("\nTesting twodConv4e...")
    # Convolve I with 3x3 ones (box blur)
    K = np.ones((3, 3)) / 9.0
    C = twodConv4e(I, K, param="ns")  # ns = no scaling

    print(f"C shape: {C.shape}")
    print(f"C center (2,2): {C[2, 2]}")
    # Should be 1/9 = 0.11

    if abs(C[2, 2] - 0.111) < 0.01:
        print("PASS: Convolution correct.")
    else:
        print(f"FAIL: Convolution value {C[2, 2]}")

    # Check polarity preservation
    if C.mean() < 0.5:
        print("PASS: Polarity preserved (Dark BG).")
    else:
        print("FAIL: Polarity inverted.")


if __name__ == "__main__":
    test_utils()
