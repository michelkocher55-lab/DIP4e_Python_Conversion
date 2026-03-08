import matplotlib.pyplot as plt

from libDIP.basisImage4e import basisImage4e
from libDIPUM.haarDWTbasisImage import haarDWTbasisImage

# Parameters
N = 8
P = 1

# Process
S_COMPOSITE, S_DISPLAY = basisImage4e('HAAR', N, P)

# Display
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(S_DISPLAY, cmap='gray', vmin=S_DISPLAY.min(), vmax=S_DISPLAY.max())
plt.title('HAAR basis functions')
plt.axis('off')

plt.subplot(1, 2, 2)
haarDWTbasisImage(3)
plt.title('Basis images 3 scale 8x8')
plt.axis('off')

# Print to file
plt.savefig('Figure631.png')
plt.show()
