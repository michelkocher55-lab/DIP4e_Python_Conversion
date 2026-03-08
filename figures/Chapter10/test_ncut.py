
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from libDIPUM.nCutSegmentation import nCutSegmentation


def test_ncut():
    # Create simple 32x32 image with two regions
    # Left half dark (0), right half light (1)
    I = np.zeros((32, 32))
    I[:, 16:] = 1.0
    
    # Add Gaussian noise
    rng = np.random.default_rng(42)
    I = I + rng.normal(0, 0.1, I.shape)
    
    print("Running nCutSegmentation...")
    # Using 2 segments
    S = nCutSegmentation(I, 2, sf=1.0)
    
    unique_lbls = np.unique(S)
    print("S unique values:", unique_lbls)
    
    plt.figure()
    plt.imshow(S)
    plt.title("Ncut Result (32x32)")
    plt.colorbar()
    
    plt.savefig("test_ncut_result.png")
    #print(f"Saved result to {output_file}")
    plt.show()

if __name__ == "__main__":
    test_ncut()
