import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DATA_PATH = os.environ.get("DATA_PATH")


# Define the autoencoder model
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                32, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Add noise to the images
def add_noise(img, noise_factor=0.5):
    noisy_img = img + noise_factor * torch.randn(*img.shape).to(img.device)
    noisy_img = torch.clip(noisy_img, 0.0, 1.0)
    return noisy_img


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:3")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    # Load the MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        root=DATA_PATH, train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize the model, loss function, and optimizer
    model = DenoisingAutoencoder()
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        for n, data in enumerate(train_loader):
            img, _ = data
            img = img.to(device)
            noisy_img = add_noise(img)

            # Forward pass
            output = model(noisy_img)
            loss = criterion(output, img)

            if n % 100 == 0:
                print("epoch: ", epoch, "iter: ", n, "loss: ", loss.item())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "denoising_autoencoder.pth")
