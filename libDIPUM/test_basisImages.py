
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from basisImages import basisImages
except ImportError:
    try:
        from .basisImages import basisImages
    except ImportError:
        import basisImages
        
def test_basisImages():
    # Create an 8x8 DCT matrix
    # DCT matrix A is such that Y = A * X * A.T
    # scipy.fftpack.dct with norm='ortho' on Identity gives the matrix rows?
    N = 8
    # Construct DCT matrix
    # A(i, j) = c(i) * cos...
    # Or just use dct on identity
    eye = np.eye(N)
    A = dct(eye, axis=0, norm='ortho')
    # scipy dct is along last axis by default? No, define axis.
    # dct(x, axis=-1)
    # If we want transformation matrix A where Y = AX (1D)
    # columns of A^T are basis vectors?
    # A's rows are basis vectors.
    # Let's verify.
    A = dct(np.eye(N), axis=1, norm='ortho')
    
    # Run basisImages
    print("Generating basis images for 8x8 DCT...")
    img = basisImages(A, gray=0.5, space=2)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title('DCT Basis Images (8x8)')
    plt.savefig('test_basisImages_result.png', bbox_inches='tight')
    print("Saved test_basisImages_result.png")
    
    # Test random complex matrix
    print("Generating basis images for random complex matrix (4x4)...")
    Ar = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
    img_c = basisImages(Ar, gray=0.5, space=1)
    plt.figure(figsize=(10, 5))
    plt.imshow(img_c, cmap='gray')
    plt.axis('off')
    plt.title('Random Complex Basis Images (Real | Imag)')
    plt.savefig('test_basisImages_complex.png', bbox_inches='tight')
    print("Saved test_basisImages_complex.png")
    plt.show()

if __name__ == "__main__":
    test_basisImages()
