import matplotlib.pyplot as plt
from libDIPUM.logdogfilter import logdogfilter

# Figure 10.23
# Comparison of LoG and DoG

# LoG and DoG filter
_, _, PL1, PD1 = logdogfilter(511, 20, 1.75, "auto", 1)
_, _, PL2, PD2 = logdogfilter(511, 20, 1.6, "auto", 1)

# Display
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].plot(-PD1, "r-.")
axes[0].plot(-PL1, "g")
axes[0].set_box_aspect(1)
axes[0].autoscale(enable=True, axis="both", tight=True)

axes[1].plot(-PD2, "r-.")
axes[1].plot(-PL2, "g")
axes[1].set_box_aspect(1)
axes[1].autoscale(enable=True, axis="both", tight=True)

plt.tight_layout()
plt.savefig("Figure1023.png", dpi=300, bbox_inches="tight")
plt.show()
