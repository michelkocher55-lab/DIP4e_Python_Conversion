
import skimage
import inspect
from skimage.morphology import thin

print(f"Skimage Version: {skimage.__version__}")
sig = inspect.signature(thin)
print(f"ci Signature: {sig}")
