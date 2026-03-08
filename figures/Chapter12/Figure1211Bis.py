
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import ia870 as ia
from libDIPUM.bwboundaries import bwboundaries
from libDIPUM.signature import signature
from libDIPUM.bound2im import bound2im

# Helper for robust image reading
def read_image_robust(path):
    # Try skimage first
    try:
        img = imread(path)
        return img
    except Exception as e:
        # Try PIL
        try:
            from PIL import Image
            img_pil = Image.open(path)
            # PIL opens as object, convert to numpy
            return np.array(img_pil)
        except ImportError:
            pass
        except Exception as e_pil:
             pass
             
        # Try OpenCV
        try:
            import cv2
            img_cv = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img_cv is not None:
                return img_cv
        except ImportError:
            pass
            
        # Re-raise original error if all fail
        print(f"Failed to read image {path} with skimage, PIL, and OpenCV.")
        raise e

def Figure1211Bis():
    print("Figure 12.11 Bis (Signatures with ia870)")
    
    # 1. Input Choice
    try:
        choix = int(input('Artificial data (1) or airplane (2) : '))
    except ValueError:
        print("Invalid input.")
        return

    f = None
    B1 = None
    title_str = ""
    
    if choix == 1:
        # Artificial
        f = np.zeros((100, 100), dtype=bool)
        f[50, 50] = True
        f_uint8 = f.astype(np.uint8)
        
        try:
            choix1 = int(input('Diamond (1), Square (2) or Disk (3) : '))
        except ValueError:
            print("Invalid input.")
            return
            
        if choix1 == 1:
            B0 = ia.iasecross(20)
            title_str = "Diamond"
        elif choix1 == 2:
            B0 = ia.iasebox(20)
            title_str = "Square"
        elif choix1 == 3:
            B0 = ia.iasedisk(20)
            title_str = "Disk"
        else:
            print("Plouc (Invalid choice)")
            return
            
        # Dilate
        f_dil = ia.iadil(f_uint8, B0)
        f_binary = f_dil > 0
        B1 = ia.iasedisk(1)
        
    elif choix == 2:
        # Airplane
        try:
            choix1 = int(input('Plane number from 1 to 4 : '))
        except ValueError:
            print("Invalid input.")
            return
            
        filename = f"Plane{choix1}.tif"
        # Search for file
        base_paths = [
            '/Users/michelkocher/michel/Data/DIP3E/DIP3E_Original_Images_CH12',
            '/Users/michelkocher/michel/Data/DIP4E',
             '.'
        ]
        path = None
        for bp in base_paths:
            p = os.path.join(bp, filename)
            if os.path.exists(p):
                path = p
                break
                
        if not path:
            print(f"{filename} not found.")
            return
            
        print(f"Loading {path}...")
        try:
            f_in = read_image_robust(path)
        except Exception as e:
            print(f"Error loading image: {e}")
            return
            
        if f_in.ndim == 3:
             f_in = f_in[:,:,0]
             
        # Process
        f_neg = ia.ianeg(f_in)
        f_opened = ia.iaareaopen(f_neg, 4)
        f_binary = f_opened > 0
        B1 = ia.iasedisk(4)
        title_str = f"Plane {choix1}"
        
    else:
        print("Plouc (Invalid choice)")
        return
        
    # Boundary Extraction
    boundaries = bwboundaries(f_binary, conn=8)
    if not boundaries:
        print("No boundaries found.")
        return
    B = boundaries[0] # Take first
    
    # Signature
    try:
        dist, angle = signature(B)
    except Exception as e:
        print(f"Error computing signature: {e}")
        return
        
    # Centroid
    y0 = np.mean(B[:, 0])
    x0 = np.mean(B[:, 1])
    
    # Display
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # 1. Image with Marker
    # Mask logic from MATLAB: mmshow(f, mmdil(Mask, B1))
    # We can overlay.
    axes[0, 0].imshow(f_binary, cmap='gray')
    axes[0, 0].plot(x0, y0, 'rx') # Simple centroid marker
    axes[0, 0].set_title(f'{title_str}, GC')
    axes[0, 0].axis('off')
    
    # 2. Angle
    axes[0, 1].plot(angle, 'b-')
    axes[0, 1].set_title('Angle')
    axes[0, 1].set_xlabel('Contour Sample')
    axes[0, 1].axis('tight')
    
    # 3. Distance (ST)
    axes[1, 0].plot(dist, 'k-')
    axes[1, 0].set_title('Distance to GC')
    axes[1, 0].set_xlabel('Contour Sample')
    axes[1, 0].axis('tight')
    
    # 4. Signature (Dist vs Angle)
    axes[1, 1].plot(angle, dist, 'k-')
    axes[1, 1].set_title('Distance to GC (Signature)')
    axes[1, 1].set_xlabel('Angle')
    axes[1, 1].axis('tight')
    
    plt.tight_layout()
    out_name = f'Figure1211Bis_{title_str.replace(" ","")}.png'
    plt.savefig(out_name)
    print(f"Saved {out_name}")
    plt.show()

if __name__ == "__main__":
    Figure1211Bis()
