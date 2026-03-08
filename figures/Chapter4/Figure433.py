import numpy as np
import matplotlib.pyplot as plt

# %% Figure 4.33
# 1-D example of ringing as a result of spatial padding.

# Data
HIdeal = np.zeros(256, dtype=float)
# MATLAB: HIdeal(125:131)=1; (1-based inclusive)
HIdeal[124:131] = 1.0

# Compute impulse response
Hm = np.fft.fftshift(HIdeal)

h = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(HIdeal)))
H = np.fft.fft(h)

# Embed h in zeros
hpad = np.zeros(512, dtype=float)
# MATLAB: hpad(129:384) = real(h(1:256));
hpad[128:384] = np.real(h[:256])
Hpad = np.fft.fft(hpad)

# Display
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.plot(HIdeal)
plt.title('H_ideal')
plt.xlim(0, len(HIdeal) - 1)

plt.subplot(2, 2, 2)
plt.plot(hpad)
plt.xlim(0, len(hpad) - 1)

plt.subplot(2, 2, 3)
plt.plot(np.real(h))
plt.xlim(0, len(h) - 1)

plt.subplot(2, 2, 4)
plt.plot(np.abs(np.fft.fftshift(Hpad)))
plt.xlim(0, len(Hpad) - 1)

plt.tight_layout()
plt.savefig('Figure433.png')
print('Saved Figure433.png')
plt.show()
