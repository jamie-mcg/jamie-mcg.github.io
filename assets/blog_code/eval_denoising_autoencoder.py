# Evaluate the model
import os
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from train_denoising_autoencoder import DenoisingAutoencoder, add_noise

DATA_PATH = os.environ.get("DATA_PATH")

model = DenoisingAutoencoder()

# Load the FashionMNIST dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

test_dataset = datasets.MNIST(
    root=DATA_PATH, train=False, download=True, transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

# Get some test images
dataiter = iter(test_loader)
images, labels = next(dataiter)
noisy_images = add_noise(images)

model.load_state_dict(torch.load("./denoising_autoencoder.pth"))

# Reconstruct the images
model.eval()
with torch.no_grad():
    reconstructed = model(noisy_images)

# Plot the original, noisy, and reconstructed images
fig, axes = plt.subplots(3, 10, figsize=(15, 5))
for i in range(10):
    axes[0, i].imshow(images[i].numpy().squeeze(), cmap="gray")
    axes[0, i].axis("off")
    axes[1, i].imshow(noisy_images[i].numpy().squeeze(), cmap="gray")
    axes[1, i].axis("off")
    axes[2, i].imshow(reconstructed[i].numpy().squeeze(), cmap="gray")
    axes[2, i].axis("off")

# Add titles to the left of each row using text
fig.text(0.05, 0.75, "Original", va="center", ha="center", fontsize=12)
fig.text(0.05, 0.5, "Noisy", va="center", ha="center", fontsize=12)
fig.text(0.05, 0.25, "Reconstructed", va="center", ha="center", fontsize=12)
plt.savefig("mnist_ae.png")
