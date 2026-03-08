
# Figure13.37 and 13.39

import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.io import imread
from libDIPUM.data_path import dip_data

def sigmoid(x):
    # Clip to avoid overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

class SimpleMLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001, seed=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(0)
            
        # Weights
        self.W1 = np.random.randn(hidden_size, input_size) * 0.1
        self.b1 = np.random.randn(hidden_size, 1) * 0.1
        self.W2 = np.random.randn(output_size, hidden_size) * 0.1
        self.b2 = np.random.randn(output_size, 1) * 0.1
        
    def forward(self, X):
        self.z1 = np.dot(self.W1, X) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.W2, self.a1) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2
        
    def train(self, X, Target, epochs=3000):
        costs = []
        N_samples = X.shape[1]
        
        for epoch in range(epochs):
            perm = np.random.permutation(N_samples)
            X_shuffled = X[:, perm]
            T_shuffled = Target[:, perm]
            
            # Batch or Stochastic? MATLAB's neuralNet4e usually does online/stochastic
            # But plain Python loops for 4000 samples * 3000 epochs is SLOW.
            # We will use Mini-Batch (e.g., 32) or Full Batch for speed in Python.
            # Given "neuralNet4e" iterates patterns, it's stochastic.
            # To be fast in Python, we'll vectorise.
            
            # Forward Full Batch
            A2 = self.forward(X_shuffled)
            
            # Cost
            error = T_shuffled - A2
            MSE = np.mean(np.sum(error**2, axis=0))
            costs.append(MSE)
            
            # Backward (Batch)
            # Delta2
            delta2 = error * (A2 * (1 - A2)) # (3, N)
            
            # Delta1
            delta1 = np.dot(self.W2.T, delta2) * (self.a1 * (1 - self.a1)) # (3, N)
            
            # Updates
            # Mean gradient over batch
            self.W2 += self.learning_rate * np.dot(delta2, self.a1.T) 
            self.b2 += self.learning_rate * np.sum(delta2, axis=1, keepdims=True) 
            self.W1 += self.learning_rate * np.dot(delta1, X_shuffled.T) 
            self.b1 += self.learning_rate * np.sum(delta1, axis=1, keepdims=True)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}: MSE={MSE:.4f}")
                
        return costs

# 1. Load Data (fixed absolute paths)
paths = {
    'VisibleBlue': dip_data('WashingtonDC-Band1-Blue-512.tif'),
    'VisibleGreen': dip_data('WashingtonDC-Band2-Green-512.tif'),
    'VisibleRed': dip_data('WashingtonDC-Band3-Red-512.tif'),
    'NearInfraRed': dip_data('WashingtonDC-Band4-NearInfrared-512.tif'),
    'WaterMask': dip_data('WashingtonDC-mask-water-512.tif'),
    'UrbanMask': dip_data('WashingtonDC-mask-urban-512.tif'),
    'VegetationMask': dip_data('WashingtonDC-mask-vegetation-512.tif'),
}

imgs = {}
for k, p in paths.items():
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing required image: {p}")
    imgs[k] = imread(p)

# Normalize images
VB = imgs['VisibleBlue'].astype(float) / 255.0
VG = imgs['VisibleGreen'].astype(float) / 255.0
VR = imgs['VisibleRed'].astype(float) / 255.0
NIR = imgs['NearInfraRed'].astype(float) / 255.0

# Extract Pixels
IxWater = np.where(imgs['WaterMask'].flatten())[0]
IxUrban = np.where(imgs['UrbanMask'].flatten())[0]
IxVegetation = np.where(imgs['VegetationMask'].flatten())[0]

def get_features(indices):
    return np.vstack([
        VB.flat[indices],
        VG.flat[indices],
        VR.flat[indices],
        NIR.flat[indices]
    ])

Water = get_features(IxWater) # (4, N)
Urban = get_features(IxUrban)
Vegetation = get_features(IxVegetation)

# Split Train/Test (50/50)
def split(arr):
    n = arr.shape[1] // 2
    return arr[:, :n], arr[:, n:]

Ref_Water, Test_Water = split(Water)
Ref_Urban, Test_Urban = split(Urban)
Ref_Vegetation, Test_Vegetation = split(Vegetation)

# Prepare Training Data
# X: (4, N_total)
Train_X = np.hstack([Ref_Water, Ref_Urban, Ref_Vegetation])

# R: One-hot (3, N_total)
# Class 1: Water, 2: Urban, 3: Veg
N_W = Ref_Water.shape[1]
N_U = Ref_Urban.shape[1]
N_V = Ref_Vegetation.shape[1]

Train_R = np.zeros((3, Train_X.shape[1]))
Train_R[0, :N_W] = 1
Train_R[1, N_W:N_W+N_U] = 1
Train_R[2, N_W+N_U:] = 1

# Train MLP
# Nodes: [4, 3, 3]
# Learning Rate Alpha=0.001
mlp = SimpleMLP(input_size=4, hidden_size=3, output_size=3, learning_rate=0.001, seed=1)

print("Training MLP (WashingtonDC)...")
costs = mlp.train(Train_X, Train_R, epochs=3000)

# Predict Test Data
Test_X = np.hstack([Test_Water, Test_Urban, Test_Vegetation])

N_Wt = Test_Water.shape[1]
N_Ut = Test_Urban.shape[1]
N_Vt = Test_Vegetation.shape[1]

# True Labels (for confusion matrix)
True_Labels = np.zeros(Test_X.shape[1], dtype=int)
True_Labels[:N_Wt] = 0 # Water
True_Labels[N_Wt:N_Wt+N_Ut] = 1 # Urban
True_Labels[N_Wt+N_Ut:] = 2 # Veg

# Forward
Output_Test = mlp.forward(Test_X)

# Convert to Class Index (Argmax)
Pred_Labels = np.argmax(Output_Test, axis=0)

# Confusion Matrix
# CM = confusion_matrix(True_Labels, Pred_Labels)
CM = np.zeros((3, 3), dtype=int)
for t, p in zip(True_Labels, Pred_Labels):
    CM[t, p] += 1
print("Confusion Matrix:")
print(CM)

# Display
# Figure 1: Original Data + MSE
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1); plt.imshow(imgs['VisibleBlue'], cmap='gray'); plt.title('Visible Blue')
plt.subplot(2, 3, 2); plt.imshow(imgs['VisibleGreen'], cmap='gray'); plt.title('Visible Green')
plt.subplot(2, 3, 3); plt.imshow(imgs['VisibleRed'], cmap='gray'); plt.title('Visible Red')
plt.subplot(2, 3, 4); plt.imshow(imgs['NearInfraRed'], cmap='gray'); plt.title('NIR')

# Mask Composite
MaskRGB = np.zeros_like(VB)
MaskCombined = np.dstack([imgs['UrbanMask'], imgs['VegetationMask'], imgs['WaterMask']]) * 255 # R=U, G=V, B=W (Approx)
# Check MATLAB text: Water(b), urban(r), veg(g)
# So R=Urban, G=Veg, B=Water
MaskColor = np.zeros((VB.shape[0], VB.shape[1], 3), dtype=float)
MaskColor[..., 0] = imgs['UrbanMask'] # R
MaskColor[..., 1] = imgs['VegetationMask'] # G
MaskColor[..., 2] = imgs['WaterMask'] # B

plt.subplot(2, 3, 5); plt.imshow(MaskColor); plt.title('Masks (U=R, V=G, W=B)')

out_dir = os.path.dirname(__file__)
plt.savefig(os.path.join(out_dir, 'Figure1337.png'))

# Figure 2: Confusion Matrix
plt.figure(figsize=(6, 6))
plt.imshow(CM, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Test Set)')
plt.colorbar()
tick_marks = np.arange(3)
plt.xticks(tick_marks, ['Water', 'Urban', 'Veg'])
plt.yticks(tick_marks, ['Water', 'Urban', 'Veg'])

thresh = CM.max() / 2.
for i in range(CM.shape[0]):
    for j in range(CM.shape[1]):
        plt.text(j, i, format(CM[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if CM[i, j] > thresh else "black")

plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'Figure1337Bis.png'))

# MSE
plt.figure(figsize=(6, 6))
plt.semilogy(costs)
plt.xlabel('Iteration')
plt.title('Train MSE')
plt.savefig(os.path.join(out_dir, 'Figure1339.png'))

plt.show()