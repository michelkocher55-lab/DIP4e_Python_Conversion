
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.feature import ORB, match_descriptors, plot_matches
from skimage.color import rgb2gray
import os
from libDIPUM.data_path import dip_data

def Figure1315SURF():
    # Init
    
    # Data
    path_f = dip_data('circuitboard.tif')
    path_pattern = dip_data('circuitboard-connector.tif')
    
    if not os.path.exists(path_f) or not os.path.exists(path_pattern):
        print("Warning: Images not found at hardcoded path.")
    
    # Load as grayscale
    f = imread(path_f, as_gray=True)
    pattern = imread(path_pattern, as_gray=True)
    
    # Feature Detection using ORB (Oriented FAST and Rotated BRIEF)
    # skimage 0.18.3 does not generally contain SIFT/SURF.
    # ORB is a good alternative for rotation invariant matching.
    
    descriptor_extractor = ORB(n_keypoints=1000)
    
    # Detect and Extract f
    descriptor_extractor.detect_and_extract(f)
    keypoints_f = descriptor_extractor.keypoints
    descriptors_f = descriptor_extractor.descriptors
    
    # Detect and Extract Pattern
    descriptor_extractor.detect_and_extract(pattern)
    keypoints_pattern = descriptor_extractor.keypoints
    descriptors_pattern = descriptor_extractor.descriptors
    
    print(f"Algorithm: ORB (skimage)")
    print(f"Points in f: {len(keypoints_f)}")
    print(f"Points in Pattern: {len(keypoints_pattern)}")
    
    # Matching
    # binary descriptors use hamming distance by default? match_descriptors handles this.
    matches12 = match_descriptors(descriptors_f, descriptors_pattern, cross_check=True)
    
    print(f"Matches found: {len(matches12)}")
    
    # Display
    
    # 1. Keypoints
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(f, cmap='gray')
    plt.title('f')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(pattern, cmap='gray')
    plt.title('Pattern')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(f, cmap='gray')
    plt.plot(keypoints_f[:, 1], keypoints_f[:, 0], '.r', markersize=2)
    plt.title(f'f, N = {len(keypoints_f)} (ORB)')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(pattern, cmap='gray')
    plt.plot(keypoints_pattern[:, 1], keypoints_pattern[:, 0], '.r', markersize=2)
    plt.title(f'Pattern, N = {len(keypoints_pattern)} (ORB)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('Figure1315SURF.png')
    
    # 2. Matches
    plt.figure(figsize=(14, 6))
    ax = plt.gca()
    
    # Show top 50 matches if too many? ORB usually produces many.
    # matches12 indices are sorted by distance? No guaranteed.
    # match_descriptors returns indices.
    
    plot_matches(ax, f, pattern, keypoints_f, keypoints_pattern, matches12, only_matches=True)
    plt.title(f'Candidate point matches ({len(matches12)} total)')
    plt.axis('off')
    plt.savefig('Figure1315SURFBis.png')
    
    plt.show()

if __name__ == "__main__":
    Figure1315SURF()
