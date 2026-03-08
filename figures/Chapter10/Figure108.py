
import sys
import os
import matplotlib.pyplot as plt
from libDIPUM.edgemodel import edgemodel

# fstep = edgemodel('step', 128, 565, 0, .9, 1);
fstep = edgemodel('step', 128, 565, 0.0, 0.9, 1)

# framp = edgemodel('ramp', 128, 565, 0, .9, 250);
framp = edgemodel('ramp', 128, 565, 0.0, 0.9, 250)

# froof = edgemodel('roof', 128, 565, 0, .9, 150);
froof = edgemodel('roof', 128, 565, 0.0, 0.9, 150)

# Display
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes = axes.flatten()

axes[0].imshow(fstep, cmap='gray')
axes[0].set_title('Step Edge')
axes[0].axis('off')

axes[1].imshow(framp, cmap='gray')
axes[1].set_title('Ramp Edge')
axes[1].axis('off')

axes[2].imshow(froof, cmap='gray')
axes[2].set_title('Roof Edge')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('Figure108.png')
print("Saved Figure108.png")
plt.show()