from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import time
import ia870 as ia
from libDIPUM.data_path import dip_data

# %% Figure918
# Hole filling

# %% Data
f_img = np.array(Image.open(dip_data("region-filling-reflections.tif")))
if f_img.ndim == 3:
    f_img = f_img[..., 0]
f = f_img > 128

# %% SE
B = ia.iasecross(1)

# %% Manual hole filling (interactive seed)
fig = plt.figure(1, figsize=(6, 4))
ax_seed = plt.subplot(1, 1, 1)
ax_seed.imshow(f, cmap="gray")
ax_seed.set_title("Click one seed point")
ax_seed.axis("off")

# Click one point on figure 1 (no Enter required).
seed = {"pt": None}


def _on_click(event: Any):
    """_on_click."""
    if event.inaxes is ax_seed and event.xdata is not None and event.ydata is not None:
        seed["pt"] = (event.xdata, event.ydata)
        print(
            f"Seed selected at (row={int(round(event.ydata))}, col={int(round(event.xdata))})"
        )


cid = fig.canvas.mpl_connect("button_press_event", _on_click)
plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)
try:
    # Bring seed window to front (helps avoid the first click being just focus).
    fig.canvas.manager.window.raise_()
except Exception:
    pass

print(f"Matplotlib backend: {matplotlib.get_backend()}")
print("Click once in the image window to set the seed point.")

t0 = time.time()
timeout_s = 60.0
while (
    seed["pt"] is None
    and plt.fignum_exists(fig.number)
    and (time.time() - t0 < timeout_s)
):
    plt.pause(0.05)

fig.canvas.mpl_disconnect(cid)
plt.close(fig)

if seed["pt"] is None:
    # Fallback if click is not captured (backend/IDE issue).
    C = f.shape[1] // 2
    R = f.shape[0] // 2
    print(f"No click captured. Using fallback seed at (row={R}, col={C}).")
else:
    C, R = seed["pt"]

r0 = int(np.clip(np.round(R), 1, f.shape[0]))
c0 = int(np.clip(np.round(C), 1, f.shape[1]))

X_list = []
x0 = np.zeros_like(f, dtype=bool)
x0[r0 - 1, c0 - 1] = True
X_list.append(x0)

x1 = ia.iaintersec(ia.ianeg(f), ia.iadil(X_list[0], B))
X_list.append(x1)

while True:
    xk = ia.iaintersec(ia.ianeg(f), ia.iadil(X_list[-1], B))
    X_list.append(xk)
    if np.array_equal(X_list[-1], X_list[-2]):
        break

g = np.maximum(f, X_list[-1])

# %% Display
fig2 = plt.figure(2, figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(f, cmap="gray")
plt.title("f")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(X_list[-1], cmap="gray")
plt.title(f"X_{{{len(X_list)}}}, X_k = d(X_{{k-1}}) ∩ ~f, X_1 = d({r0}, {c0})")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(g, cmap="gray")
plt.title(f"g = X_{{{len(X_list)}}} ∪ f")
plt.axis("off")

plt.tight_layout()
fig2.savefig("Figure918.png", dpi=150, bbox_inches="tight")
print("Saved Figure918.png from figure 2.")
plt.show()
