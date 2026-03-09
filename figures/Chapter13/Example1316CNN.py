import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from libDIPUM.cnnsetup import cnnsetup
from libDIPUM.cnntrain import cnntrain
from libDIPUM.cnntest import cnntest

print("Running Example1316CNN...")

# 1. Generate Training and Testing Images
num_classes = 3
num_images_per_class = 3

# Raw Data Definition
raw_data = np.array(
    [
        # Class 1
        [255, 0, 255, 255, 0, 255, 255, 0, 255],
        [198, 5, 213, 244, 14, 245, 241, 8, 231],
        [246, 8, 222, 225, 40, 237, 228, 5, 235],
        # Class 2
        [255, 255, 255, 255, 0, 255, 255, 255, 255],
        [234, 255, 205, 251, 0, 251, 238, 253, 240],
        [232, 255, 231, 247, 38, 246, 190, 236, 250],
        # Class 3
        [255, 255, 255, 0, 0, 0, 255, 255, 255],
        [245, 225, 205, 1, 0, 5, 238, 253, 240],
        [225, 235, 231, 7, 8, 4, 190, 236, 250],
    ]
)

images = raw_data.T  # (9, 9)

# Labels
labels = np.array(
    [
        [1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1],
    ]
)

# Format Data
train_x = images.reshape(3, 3, -1, order="F") / 255.0

# Resize to 6x6
train_x = ndimage.zoom(train_x, (2, 2, 1), order=1)
train_y = labels

# Test Generation
test_x = images.reshape(3, 3, -1, order="F") / 255.0
test_x = ndimage.zoom(test_x, (2, 2, 1), order=1)

# Add Gaussian Noise
sigma = np.sqrt(0.01)
noise = np.random.normal(0, sigma, test_x.shape)
test_x = test_x + noise

# Clip to 0-1
test_x = np.clip(test_x, 0.0, 1.0)
test_y = labels

# Setup CNN
# input (6x6) -> Conv (3x3, 2 kernels) -> 4x4x2
# -> Sub (2x2) -> 2x2x2
# -> Flatten -> 8
# -> FC -> 3
np.random.seed(0)

cnn = {}
cnn["layers"] = [
    {"type": "i"},
    {"type": "c", "outputmaps": 2, "kernelsize": 3},
    {"type": "s", "scale": 2},
]

cnn = cnnsetup(cnn, train_x, train_y)

# Train
opts = {"alpha": 1.0, "batchsize": 9, "numepochs": 400}

print("Training...")
cnn = cnntrain(cnn, train_x, train_y, opts)

# Test
err_train, cnn, bad_train, est_train = cnntest(cnn, train_x, train_y)
print(f"Error rate (training) = {err_train * 100:.2f}%")

err_test, cnn, bad_test, est_test = cnntest(cnn, test_x, test_y)
print(f"Error rate (testing) = {err_test * 100:.2f}%")

# Class-wise Accuracy (Testing)
# est_test: indices of predictions.
# truth: indices of truth.
truth_labels = np.argmax(test_y, axis=0)

# Accuracy
classes = np.unique(truth_labels)
num_classes = len(classes)

h1 = np.zeros(num_classes)  # Total
h2 = np.zeros(num_classes)  # Correct

for i in range(len(truth_labels)):
    cls = truth_labels[i]
    h1[cls] += 1
    if cls == est_test[i]:
        h2[cls] += 1
accuracy = np.divide(h2, h1, out=np.zeros_like(h1), where=h1 != 0)

# Display
plt.figure("Training Images", figsize=(6, 6))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(train_x[:, :, i], cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.title(f"Train / Class {i // 3 + 1}")
plt.tight_layout()
plt.savefig("Example1316CNN_Training_Images.png")

plt.figure("Testing Images", figsize=(6, 6))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_x[:, :, i], cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.title(f"Test {i + 1}")
plt.tight_layout()
plt.savefig("Example1316CNN_Testing_Images.png")

plt.figure("MSE")
plt.plot(cnn["rL"], linewidth=2)
plt.xlabel("Batches")
plt.ylabel("Mean Squared Error")
plt.title("Training MSE")
plt.grid(True)
plt.savefig("Example1316CNN_MSE.png")

plt.figure("Class Accuracy")
x = np.arange(num_classes) + 1
width = 0.35
plt.bar(x - width / 2, h1, width, label="Total Samples")
plt.bar(x + width / 2, h2, width, label="Correct Samples")
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Class-wise Accuracy")
plt.xticks(x)
plt.legend()

for i, acc in enumerate(accuracy):
    plt.text(x[i], h1[i] + 0.1, f"{acc:.2f}", ha="center")

plt.savefig("Example1316CNN_Accuracy.png")

# Visualize Kernels
# Layer 1 is 'c'
layer1 = cnn["layers"][1]
if layer1["type"] == "c":
    kernels = layer1["k"]  # list of lists
    num_input = len(kernels)
    num_output = len(kernels[0])

    plt.figure("Kernels")
    idx = 1
    for j in range(num_input):
        for k in range(num_output):
            plt.subplot(num_input, num_output, idx)
            plt.imshow(kernels[j][k], cmap="gray")
            plt.axis("off")
            idx += 1
    plt.suptitle(f"Kernels: {num_input} in -> {num_output} out")
    plt.savefig("Example1316CNN_Kernels.png")

# Feedforward Example
# Plot maps for Sample 1
sample_idx = 0
fig_ff = plt.figure("FeedForward Example", figsize=(10, 6))

# Input
plt.subplot(2, 4, 1)
plt.imshow(cnn["layers"][0]["a"][0][:, :, sample_idx], cmap="gray")
plt.title("Input")
plt.axis("off")

# Conv Maps (2 maps)
# 4x4
for k in range(2):
    plt.subplot(2, 4, 2 + k)
    plt.imshow(cnn["layers"][1]["a"][k][:, :, sample_idx], cmap="gray")
    plt.title(f"Conv {k + 1}")
    plt.axis("off")

# Sub Maps (2 maps)
# 2x2
for k in range(2):
    plt.subplot(2, 4, 5 + k)
    plt.imshow(cnn["layers"][2]["a"][k][:, :, sample_idx], cmap="gray")
    plt.title(f"Sub {k + 1}")
    plt.axis("off")

# Output
plt.subplot(2, 4, 8)
out_probs = cnn["o"][:, sample_idx]
plt.bar(range(1, 4), out_probs)
plt.title("Output Probs")

plt.tight_layout()
plt.savefig("Example1316CNN_FeedForward.png")

print("Example1316CNN Completed. Figures saved.")
plt.show()
