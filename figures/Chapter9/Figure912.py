import numpy as np
import matplotlib.pyplot as plt
import ia870 as ia
from General.mmshow import mmshow

# %% Figure912

# %% Data
X = np.zeros((14, 17), dtype=bool)
X[1:10, 1:6] = True
X[8:13, 7:12] = True
X[4:7, 13:16] = True

# %% Interval
BBGImg = np.zeros((7, 7), dtype=bool)
BFGImg = np.zeros((7, 7), dtype=bool)
BFGImg[1:6, 1:6] = True
BBGImg = ia.iaframe(BBGImg)

BFG = ia.iaimg2se(BFGImg)
BBG = ia.iaimg2se(BBGImg)

I = ia.iase2hmt(BFG, BBG)

# %% HMT
Y_hmt = ia.iasupgen(X, I)

Y_eroFG = ia.iaero(X, BFG)
Y_eroBG = ia.iaero(ia.ianeg(X), BBG)

inter = ia.iaintersec(Y_eroFG, Y_eroBG)
ok = np.all(Y_hmt == inter)
print(f"OK = {ok}")

# %% Display
Grid = np.ones_like(BFGImg, dtype=bool)

fig = plt.figure(1, figsize=(14, 7))

plt.subplot(2, 4, 1)
plt.imshow(ia.iabshow(Grid, BFGImg))
plt.title("B_{FG}")
plt.axis("off")

plt.subplot(2, 4, 2)
plt.imshow(ia.iabshow(Grid, BBGImg))
plt.title("B_{BG}")
plt.axis("off")

plt.subplot(2, 4, 3)
plt.imshow(ia.iabshow(Grid, BFGImg, ia.ianeg(ia.iaunion(BFGImg, BBGImg))))
plt.title("B = (B_{FG}, B_{BG})")
plt.axis("off")

plt.subplot(2, 4, 4)
plt.imshow(X, cmap="gray")
plt.title("X")
plt.axis("off")

plt.subplot(2, 4, 5)
plt.imshow(Y_eroFG, cmap="gray")
plt.title(r"$\epsilon_{B_{FG}}(X)$")
plt.axis("off")

plt.subplot(2, 4, 6)
plt.imshow(ia.ianeg(X), cmap="gray")
plt.title(r"$X^C$")
plt.axis("off")

plt.subplot(2, 4, 7)
plt.imshow(Y_eroBG, cmap="gray")
plt.title(r"$\epsilon_{B_{BG}}(X^C)$")
plt.axis("off")

plt.subplot(2, 4, 8)
mmshow(X, inter)
plt.title(r"$\epsilon_{B_{FG}}(X) \cap \epsilon_{B_{BG}}(X^C)$")
plt.axis("off")

plt.tight_layout()
fig.savefig("Figure912.png", dpi=150, bbox_inches="tight")
plt.show()
