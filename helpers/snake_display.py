from typing import Any
import matplotlib.pyplot as plt


def snake_display(x: Any, y: Any, style: Any = "-k"):
    """
    Displays the snake curve.

    Parameters:
        x, y: Coordinates.
        style: Plot style.
    """
    # Note: MATLAB plot(y, x) -> y horizontal, x vertical.
    # Matches our convention x=Row, y=Col.
    # Matplotlib plot(horizontal, vertical) => plot(y, x).

    # MATLAB description says "style" string.
    # Matplotlib accepts style strings like 'r-', 'bo', etc.
    # We pass it directly.
    plt.plot(y, x, style)
