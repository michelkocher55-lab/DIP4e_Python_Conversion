import matplotlib.pyplot as plt


def curve_display(x, y, style=None, ax=None):
    """
    Display of 2-D curve.

    curve_display(x, y, style) displays the coordinates (x, y) of a curve.
    To superimpose the curve on an image:
        plt.imshow(f, cmap='gray')
        curve_display(x, y, style)

    style: matplotlib-style string (e.g., 'ro-', 'g.', etc.).
    Default is black dots with no lines ('.k').
    """
    if style is None:
        style = '.k'

    if ax is None:
        ax = plt.gca()

    # MATLAB uses plot(y, x). Keep the same convention.
    ax.plot(y, x, style)
