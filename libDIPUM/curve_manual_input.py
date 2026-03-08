import numpy as np
import matplotlib.pyplot as plt
from libDIPUM.interparc import interparc

def curve_manual_input(f, numpoints='auto', style=None):
    """
    Manual input of a closed polygonal curve using matplotlib ginput.

    [x, y, vx, vy] = curve_manual_input(f, numpoints, style)
    """
    print(" ")
    print("INSTRUCTIONS")
    print("  1) To BEGIN: Press any key except RETURN.")
    print("  2) LEFT-CLICK with the MOUSE to enter points")
    print("  3) When DONE, press RETURN")

    fig, ax = plt.subplots()
    ax.imshow(f, cmap='gray')
    ax.set_title('Click to input points, press Enter when done')
    plt.pause(0.01)
    input()  # wait for any key

    pts = plt.ginput(n=-1, timeout=0)
    plt.close(fig)

    if len(pts) == 0:
        raise ValueError("No points were selected.")

    # ginput returns (x, y) as (col, row)
    c = np.array([p[0] for p in pts])
    r = np.array([p[1] for p in pts])

    x = r.copy()
    y = c.copy()
    vx = r.copy()
    vy = c.copy()

    # Close the curve
    x = np.concatenate([x, [x[0]]])
    y = np.concatenate([y, [y[0]]])

    # Compute number of points
    if numpoints == 'auto':
        xd = x.reshape(-1, 1)
        yd = y.reshape(-1, 1)
        d = np.sqrt((xd - np.roll(xd, 1)) ** 2 + (yd - np.roll(yd, 1)) ** 2)
        np_points = int(np.ceil(2 * np.sum(d)))
    else:
        np_points = int(numpoints)

    if np_points < len(vx):
        raise ValueError("the number of points must be > the no. of input vertices")

    # Interpolate to get equally spaced points
    qx, qy = interparc(np_points, y, x)
    x = qy
    y = qx

    # If style provided, display curve
    if style is not None:
        fig2, ax2 = plt.subplots()
        ax2.imshow(f, cmap='gray')
        ax2.plot(y, x, style)
        plt.show()

    return x, y, vx, vy
