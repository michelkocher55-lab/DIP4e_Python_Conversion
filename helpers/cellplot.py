from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def cellplot(c: Any, lims: Any = None):
    """
    Display graphical depiction of nested list/tuple (MATLAB-like cellplot).
    If lims == 'legend', adds a legend colorbar.
    """
    legend = False
    if lims is not None:
        if isinstance(lims, str) and lims.lower() == "legend":
            legend = True
        else:
            raise ValueError("Invalid legend value.")

    if c is None:
        return []
    if not isinstance(c, (list, tuple)):
        raise TypeError("cellplot: input must be list/tuple.")

    fig, ax = plt.gcf(), plt.gca()
    ax.set_aspect("equal")
    ax.axis("off")

    # colors for: double, char, sparse, structure, other
    cmap = plt.get_cmap("prism", 5)
    handles = []

    def _type_code(obj: Any):
        """_type_code."""
        if isinstance(obj, str):
            return 2
        if isinstance(obj, dict):
            return 4
        if isinstance(obj, (int, float, np.number, np.ndarray)):
            return 1
        return 5

    def _draw_cell(obj: Any, x0: Any, y0: Any, w: Any, h: Any, depth: Any = 0):
        """_draw_cell."""
        if isinstance(obj, (list, tuple)):
            m = len(obj)
            if m == 0:
                return
            # Draw container outline
            pad = 0.04 * min(w, h)
            rect = Rectangle(
                (x0, y0 - h), w, h, facecolor="none", edgecolor="k", linewidth=1.0
            )
            ax.add_patch(rect)
            # Inner bounds for children
            cx0 = x0 + pad
            cy0 = y0 - pad
            cw = w - 2 * pad
            ch = h - 2 * pad
            # If this is a proper 2D list (all rows are list/tuple of same length)
            is_2d = all(isinstance(row, (list, tuple)) for row in obj)
            if is_2d:
                n = len(obj[0]) if len(obj[0]) > 0 else 1
                gapx = 0.03 * min(w, h)
                gapy = 0.03 * min(w, h)
                cell_w = (cw - gapx * (n - 1)) / n
                cell_h = (ch - gapy * (m - 1)) / m
                for i in range(m):
                    row = obj[i]
                    for j in range(n):
                        item = row[j] if j < len(row) else None
                        _draw_cell(
                            item,
                            cx0 + j * (cell_w + gapx),
                            cy0 - i * (cell_h + gapy),
                            cell_w,
                            cell_h,
                            depth + 1,
                        )
            else:
                # Treat 1-D cell arrays:
                # Top-level -> vertical stacking (Nx1), deeper levels -> horizontal (1xN)
                if depth == 0:
                    gap = 0.03 * min(w, h)
                    cell_h = (ch - gap * (m - 1)) / m
                    for i in range(m):
                        item = obj[i]
                        _draw_cell(
                            item, cx0, cy0 - i * (cell_h + gap), cw, cell_h, depth + 1
                        )
                else:
                    n = m
                    gap = 0.03 * min(w, h)
                    cell_w = (cw - gap * (n - 1)) / n
                    for j in range(n):
                        item = obj[j]
                        _draw_cell(
                            item, cx0 + j * (cell_w + gap), cy0, cell_w, ch, depth + 1
                        )
        else:
            code = _type_code(obj)
            color = cmap((code - 1) / 5.0)
            pad_leaf = 0.06 * min(w, h)
            rect = Rectangle(
                (x0 + pad_leaf, y0 - h + pad_leaf),
                w - 2 * pad_leaf,
                h - 2 * pad_leaf,
                facecolor=color,
                edgecolor="k",
                linewidth=0.5,
            )
            ax.add_patch(rect)
            handles.append(rect)
            # Add text for scalar numbers/short strings
            if isinstance(obj, (int, float, np.number)):
                ax.text(
                    x0 + w / 2,
                    y0 - h / 2,
                    str(int(obj)) if float(obj).is_integer() else str(obj),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="k",
                )
            elif isinstance(obj, str) and len(obj) <= 10:
                ax.text(
                    x0 + w / 2,
                    y0 - h / 2,
                    obj,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="k",
                )

    # Treat c as 2-D grid if it's a list of lists
    if len(c) > 0 and isinstance(c[0], (list, tuple)):
        rows = len(c)
        cols = len(c[0])
    else:
        rows = len(c)
        cols = 1

    _draw_cell(c, 0, rows, cols, rows)
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)

    if legend:
        cb = plt.colorbar(
            plt.cm.ScalarMappable(cmap=cmap), ax=ax, ticks=np.linspace(0.1, 0.9, 5)
        )
        cb.ax.set_yticklabels(["double", "char", "sparse", "structure", "other"])

    return handles
