import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from libDIPUM.lpfilter import lpfilter

# Transfer function
H = lpfilter('ideal', 1000, 1000, 30)
M, N = H.shape

# Impulse response
h = np.fft.fftshift(np.fft.ifft2(H))
h = np.real(h)
profile = h[M // 2, :]

# Display
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(np.fft.fftshift(H), cmap='gray')
axes[0].set_title('H')
axes[0].axis('off')

axes[1].imshow(h, cmap='gray')
axes[1].set_title('h')
axes[1].axis('off')

axes[2].plot(profile)
axes[2].set_title('profile (h)')
#axes[2].set_aspect('equal', adjustable='box')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('Figure442.png')
plt.show()
