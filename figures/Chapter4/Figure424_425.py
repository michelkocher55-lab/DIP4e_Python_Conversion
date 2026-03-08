import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate


# Centered rectangle
f = np.zeros((512, 512), dtype=float)
f[207:304, 247:264] = 1  # MATLAB 208:304, 248:264

# Displaced rectangle
g = np.zeros((512, 512), dtype=float)
g[107:204, 347:364] = 1  # MATLAB 108:204, 348:364

# Rotated rectangle (bilinear, crop)
r = rotate(f, angle=-45, resize=False, order=1, mode='constant', cval=0.0, preserve_range=True)

# Fourier transform
F = np.fft.fft2(f)

G = np.fft.fft2(g)
SG = np.abs(G)
SG = np.fft.fftshift(SG)
SG = np.log10(1 + np.abs(SG))
SG = SG - SG.min()
SG = SG / SG.max()

R = np.fft.fft2(r)
SR = np.abs(R)
SR = np.fft.fftshift(SR)
SR = np.log10(1 + np.abs(SR))
SR = SR - SR.min()
SR = SR / SR.max()

# Phase angles
fphi = np.angle(F)
gphi = np.angle(G)
rphi = np.angle(R)

# Reconstruct
Cf = 1j * fphi
Cg = 1j * gphi

Freconst = np.abs(F) * np.exp(Cf)
frec = np.real(np.fft.ifft2(Freconst))

Greconst = np.abs(G) * np.exp(Cg)
grec = np.real(np.fft.ifft2(Greconst))

# Display figure 1
fig1, axes1 = plt.subplots(2, 2, figsize=(10, 10))
axes1[0, 0].imshow(f, cmap='gray')
axes1[0, 0].set_title('f')
axes1[0, 0].axis('off')

axes1[0, 1].imshow(frec, cmap='gray')
axes1[0, 1].set_title('f_rec')
axes1[0, 1].axis('off')

axes1[1, 0].imshow(g, cmap='gray')
axes1[1, 0].set_title('g')
axes1[1, 0].axis('off')

axes1[1, 1].imshow(grec, cmap='gray')
axes1[1, 1].set_title('g_rec')
axes1[1, 1].axis('off')

plt.tight_layout()
plt.savefig('Figure424.png')

# Display figure 2
fig2, axes2 = plt.subplots(2, 2, figsize=(10, 10))
axes2[0, 0].imshow(g, cmap='gray')
axes2[0, 0].set_title('f translated')
axes2[0, 0].axis('off')

axes2[0, 1].imshow(SG, cmap='gray')
axes2[0, 1].set_title('Fourier (f translated)')
axes2[0, 1].axis('off')

axes2[1, 0].imshow(r, cmap='gray')
axes2[1, 0].set_title('f rotated')
axes2[1, 0].axis('off')

axes2[1, 1].imshow(SR, cmap='gray')
axes2[1, 1].set_title('Fourier (f rotated)')
axes2[1, 1].axis('off')

plt.tight_layout()
plt.savefig('Figure424Bis.png')

# Display figure 3
fig3, axes3 = plt.subplots(1, 3, figsize=(12, 4))
axes3[0].imshow(fphi, cmap='gray')
axes3[0].axis('off')

axes3[1].imshow(gphi, cmap='gray')
axes3[1].axis('off')

axes3[2].imshow(rphi, cmap='gray')
axes3[2].axis('off')

plt.tight_layout()
plt.savefig('Figure425.png')
plt.show()
