from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from helpers.ginput import ginput
from helpers.interparc import interparc


def snake_manual_input(f: Any, np_points: Any, style: Any = "b."):
    """
    Manual input of initial snake.

    Parameters:
        f: Image.
        np_points: Number of points for the snake.
        style: Plot style (default 'b.' as in MATLAB code default logic description,
               though MATLAB code says 'b' color, '.' symbol).

    Returns:
        x, y: Coordinates of the snake (r, c).
    """
    print(" ")
    print("PRESS <ENTER> TO TERMINATE DATA ENTRY. SELECT POINTS WITH THE MOUSE.")
    print(" ")

    # Display image
    fig = plt.figure()
    plt.imshow(f, cmap="gray")
    plt.axis("off")
    plt.title("Select points for snake")

    print("Waiting for input...")
    # n=-1 means accumulate until Enter
    col, row = ginput(n=-1, show_clicks=True)
    plt.plot(col, row, marker="+", color="r", markersize=12, mew=2)
    plt.draw()

    if len(col) == 0:
        print("No points selected.")
        return np.array([]), np.array([])

    x = row  # Vertical (Row)
    y = col  # Horizontal (Col)

    # Add one more point to close the snake
    x = np.append(x, x[0])
    y = np.append(y, y[0])

    # Interpolate
    qy, qx = interparc(np_points, y, x)

    # Assign back
    x_out = qx
    y_out = qy

    # Close figure?
    plt.close(fig)

    # Display result
    plt.figure()
    plt.imshow(f, cmap="gray")
    plt.axis("off")
    plt.title("Initial Snake")

    # Plot
    plt.plot(y_out, x_out, style)
    plt.show()

    return x_out, y_out
