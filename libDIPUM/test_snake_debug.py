
import numpy as np
import matplotlib.pyplot as plt
from snakeMap4e import snakeMap4e
from morphoThin4e import morphoThin4e
from morphoHitmiss4e import morphoHitmiss4e

def test_morphoThin():
    print("Testing morphoThin4e on a square...")
    # Create 9x9 image with 5x5 square
    I = np.zeros((11, 11))
    I[3:8, 3:8] = 1
    
    # Thinning should produce a skeleton (single point or line?)
    # For a square, it usually reduces to center dot or small shape depending on iteration.
    # Standard thinning reduces to skeleton.
    
    T = morphoThin4e(I)
    
    # Display
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(I, cmap='gray')
    ax[0].set_title('Original Square')
    ax[1].imshow(T, cmap='gray')
    ax[1].set_title('Thinned')
    plt.savefig('test_thin_square.png')
    plt.close()
    
    # Check if T is empty
    if np.sum(T) == 0:
        print("ERROR: Thinned image is empty.")
    else:
        print(f"Thinned image has {np.sum(T)} pixels.")
        print("Thinned Array:")
        print(T)
        
    # Check B seq
    B = np.zeros((3, 3, 8), dtype=int)
    temp_B1 = np.array([[0, 0, 0], [2, 1, 2], [1, 1, 1]])
    B[:, :, 0] = temp_B1
    print("B1 kernel:")
    print(B[:, :, 0])
    
    # Test hitmiss with B1 on I
    from morphoHitmiss4e import morphoHitmiss4e
    hm = morphoHitmiss4e(I, B[:, :, 0])
    print("Hitmiss match count for B1:", np.sum(hm))

def test_snakeMap():
    print("Testing snakeMap4e on step edge...")
    # Create step edge image
    I = np.zeros((50, 50))
    I[:, 25:] = 1.0
    
    # Add noise?
    # No, clean edge first.
    
    # Map
    # T='auto'
    emap = snakeMap4e(I, T='auto')
    
    plt.figure()
    plt.imshow(emap, cmap='gray')
    plt.title('SnakeMap Step Edge')
    plt.savefig('test_snakemap_edge.png')
    plt.close()
    
    if np.sum(emap) == 0:
        print("ERROR: SnakeMap is empty.")
    else:
        print(f"SnakeMap has {np.sum(emap)} pixels.")
        
    # Check values
    print("Unique values in emap:", np.unique(emap))

if __name__ == "__main__":
    test_morphoThin()
    test_snakeMap()
