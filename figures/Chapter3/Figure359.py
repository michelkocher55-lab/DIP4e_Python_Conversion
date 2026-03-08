import matplotlib.pyplot as plt
from libDIPUM.zoneplate import zoneplate

# Data
f = zoneplate(8.2, 0.0275, 0)
print(f.shape)

# Display
plt.figure(figsize=(6, 6))
plt.imshow(f, cmap='gray', vmin=0, vmax=1)
plt.axis('off')
plt.tight_layout()

# Print to file
plt.savefig('Figure359.png')
print('Saved Figure359.png')
plt.show()
