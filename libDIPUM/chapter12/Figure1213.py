import numpy as np
import matplotlib.pyplot as plt
import ia870 as ia

print("Running Figure1213 (Distance Transform with ia870)...")

# Data
# print("Generating X3...")
X3 = ~ia.iaframe(np.ones((200, 400), dtype=bool), 20, 20)

# 2. Distance Transform
DT3 = ia.iadist(X3, ia.iasebox(), "EUCLIDEAN")

# 3. Display
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(X3, cmap="gray")
axes[0].set_title("X")
axes[0].axis("off")

axes[1].imshow(DT3, cmap="gray")
axes[1].set_title("DT(X)")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("Figure1213.png")
print("Saved Figure1213.png")
plt.show()
