from typing import Any
import matplotlib.pyplot as plt

from helpers.libdipum.twomodegauss import twomodegauss


def manualhist(*args: Any):
    """
    Generates a two-mode histogram.

    If called with args (m1, sig1, m2, sig2, A1, A2, k), returns the histogram p.
    If called without args, enters interactive mode (plotting and prompting).

    Parameters:
    args: Optional. (m1, sig1, m2, sig2, A1, A2, k)

    Returns:
    p (ndarray): 256-element histogram.
    """

    # Default values
    defaults = [0.15, 0.05, 0.75, 0.05, 1, 0.07, 0.002]

    if len(args) == 7:
        return twomodegauss(*args)
    elif len(args) > 0:
        print("Error: manualhist requires 0 or 7 arguments.")
        return None

    # Interactive mode
    p = twomodegauss(*defaults)

    print("Interactive Mode. Default values used initially.")
    print(f"Defaults: {defaults}")
    print("Enter 'x' to quit.")

    while True:
        # Plot current
        plt.figure()
        plt.plot(p)
        plt.xlim([0, 255])
        plt.title("Current Histogram")
        plt.show()

        user_input = input("Enter m1, sig1, m2, sig2, A1, A2, k OR x to quit: ")
        if user_input.strip().lower() == "x":
            break

        try:
            # Parse CSV or space separated
            parts = user_input.replace(",", " ").split()
            values = [float(v) for v in parts]

            if len(values) != 7:
                print("Incorrect number of inputs. Need 7.")
                continue

            p = twomodegauss(*values)

        except ValueError:
            print("Invalid input. Please enter numbers.")

    return p


if __name__ == "__main__":
    manualhist()
