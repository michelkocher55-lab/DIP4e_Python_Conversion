import matplotlib.pyplot as plt

from libDIP.basisImage4e import basisImage4e

# Process
S_COMPOSITEr, S_DISPLAYr = basisImage4e('DFTr', 8, 1)
S_COMPOSITEi, S_DISPLAYi = basisImage4e('DFTi', 8, 1)

# Display
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(S_DISPLAYr, cmap='gray', vmin=S_DISPLAYr.min(), vmax=S_DISPLAYr.max())
axes[0].axis('off')
axes[1].imshow(S_DISPLAYi, cmap='gray', vmin=S_DISPLAYi.min(), vmax=S_DISPLAYi.max())
axes[1].axis('off')

plt.tight_layout()
plt.savefig('Figure67.png')
plt.show()
