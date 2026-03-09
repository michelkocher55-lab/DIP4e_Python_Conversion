import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import filtfilt

# Init
np.random.seed(0)  # rng('default')


# Parameters
Choice = 1
SigmaNoise = 0.5

Num = np.array([0.1, 1.0, 0.1])
Den = np.array([1.0, 0.0, 0.0]) * np.sum(Num)

# Signature Data (10x11 matrix)
Signature = np.array(
    [
        [0, 2, -2, -0.3, -0.2, 0, 0, 1.8, -2, 0, 0],
        [0, 0, 1, 0, 1, -1, -2, 0, 0, 0, 0],
        [0, 0, 2, -1, -0.5, 1, -2, 0, 0, 0, 0],
        [0, 0, 1.3, 1, -2, -0.2, 0, -1, 0, 0, 0],
        [0, 0, 1.2, 0, -1, 0, 2, 0, -2.5, 0, 0],
        [0, 0, 1.5, -1, -0.2, -0.2, 1, -2, 0, 0, 0],
        [0, 0, 1, -0.6, 0.3, -0.5, 0, 2, -3, 0, 0],
        [0, 0, 1.2, -1, 1, -2, 0.5, -1, 0, 0, 0],
        [0, 0, 1.2, 1.2, -2, -0.2, 0, 2, -2, -2, 0],
        [0, 0, 2, -1, -1.8, 0, 0, 1, -2.5, 0, 0],
    ]
)

NSymbols, N = Signature.shape
k = np.arange(1, N + 1)

# Signatures
if Choice:
    SignatureBlurred = filtfilt(Num, Den, Signature, axis=0)
else:
    SignatureBlurred = Signature.copy()

# Classification Stats (Monte Carlo)
Nr = 100
ConfusionMatrix = np.zeros((NSymbols, NSymbols), dtype=int)
TotalErrors = 0

# Store variables from last run for plotting
SignatureNoise_Last = None
Distance_Last = None

for r in range(Nr):
    # Generate new noise
    SignatureNoise = SignatureBlurred + SigmaNoise * np.random.randn(NSymbols, N)

    # Calculate Distances
    Dist_r = np.linalg.norm(
        Signature[:, np.newaxis, :] - SignatureNoise[np.newaxis, :, :], axis=2
    )

    # Classify
    Predicted_r = np.argmin(Dist_r, axis=0)

    # Accumulate Confusion
    for j in range(NSymbols):  # True class j
        ConfusionMatrix[j, Predicted_r[j]] += 1

    # Accumulate Errors
    TotalErrors += np.sum(Predicted_r != np.arange(NSymbols))

    # Keep last run for display
    if r == Nr - 1:
        Distance_Last = Dist_r
        SignatureNoise_Last = SignatureNoise

# Average Error Rate
ErrorRate = 100 * TotalErrors / (Nr * NSymbols)

# Display (Using last run for signals/dist)
SymbolList = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# Figure 1: Signatures (Last Run)
plt.figure(figsize=(12, 8))
for i in range(NSymbols):
    plt.subplot(3, 4, i + 1)
    plt.plot(k, Signature[i], "r", label="Clean")
    plt.plot(k, SignatureBlurred[i], "g", label="Blurred")
    plt.plot(k, SignatureNoise_Last[i], "b", label="Noisy")
    plt.title(SymbolList[i])
    plt.grid(True)
    plt.ylim([np.min(SignatureNoise_Last), np.max(SignatureNoise_Last)])

plt.tight_layout()
plt.savefig("Figure1311.png")

# Figure 2: Distance stems (Last Run)
plt.figure(figsize=(12, 8))
for i in range(NSymbols):
    plt.subplot(3, 4, i + 1)
    plt.stem(range(NSymbols), Distance_Last[i])
    plt.title(f"Dist. From {SymbolList[i]}")
    plt.xlabel("to")
    plt.grid(True)

plt.tight_layout()
plt.savefig("Figure1311Bis.png")

# Figure 3: Distance Matrix (Gray) (Last Run)
plt.figure()
plt.imshow(Distance_Last, interpolation="nearest", cmap="gray")
plt.colorbar()
plt.title("Distance Matrix (Last Run)")
plt.xlabel("Noisy Sample Index")
plt.ylabel("Prototype Index")
plt.savefig("Figure1311Ter.png")

# Figure 4: Accumulated Confusion Matrix (Gray Tones!)
plt.figure()
plt.imshow(ConfusionMatrix, interpolation="nearest", cmap="Greys")
plt.colorbar(label="Count (out of 100)")
plt.title(f"Confusion Matrix ({Nr} runs, Mean Error = {ErrorRate:.1f}%)")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")

# Add grid/ticks for clarity
plt.xticks(np.arange(NSymbols), SymbolList)
plt.yticks(np.arange(NSymbols), SymbolList)

plt.savefig("Figure1311Quater.png")

print(f"Mean Error Rate ({Nr} runs): {ErrorRate:.1f}%")
print("Confusion Matrix saved to Figure1311Quater.png")

plt.show()
