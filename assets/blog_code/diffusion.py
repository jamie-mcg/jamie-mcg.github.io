import numpy as np
import matplotlib.pyplot as plt
import imageio
from skimage import data
from numpy.fft import fft2, fftshift
import os


# FIGURE 1: Diffusion process
def add_noise(image, noise_level):
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)


def diffusion_process(image, steps=10):
    images = []
    for i in range(steps):
        noise_level = i / steps
        noisy_image = add_noise(image, noise_level)
        images.append(noisy_image)
    return images


# Load a sample image
image = data.astronaut() / 255.0

# Convert to grayscale
image = np.mean(image, axis=2)

# Generate diffusion process images
diffusion_images = diffusion_process(image, steps=10)

# Save as GIF
imageio.mimsave(
    "../img/blogs/diffusion/diffusion_process.gif",
    [np.uint8(img * 255) for img in diffusion_images],
    duration=0.5,
)

# Display the GIF
# from IPython.display import Image

# Image(filename="diffusion_process.gif")


# FIGURE 2: Reverse diffusion process
def reverse_diffusion_process(image, steps=10):
    images = []
    for i in range(steps):
        noise_level = (steps - i) / steps
        noisy_image = add_noise(image, noise_level)
        images.append(noisy_image)
    return images


# Generate reverse diffusion process images
reverse_diffusion_images = reverse_diffusion_process(image, steps=10)

# Save as GIF
imageio.mimsave(
    "../img/blogs/diffusion/reverse_diffusion_process.gif",
    [np.uint8(img * 255) for img in reverse_diffusion_images],
    duration=0.5,
)

# # Display the GIF
# Image(filename="../img/blogs/diffusion/reverse_diffusion_process.gif")


# FIGURE 3: Power spectrum
def compute_power_spectrum(image):
    f_transform = fft2(image)
    f_transform_shifted = fftshift(f_transform)
    power_spectrum = np.abs(f_transform_shifted) ** 2
    return power_spectrum


def plot_power_spectrum(image, steps=10):
    diffusion_images = diffusion_process(image, steps)
    plt.figure(figsize=(15, 5))

    for i, img in enumerate(diffusion_images):
        power_spectrum = compute_power_spectrum(img)
        plt.subplot(2, steps // 2, i + 1)
        plt.imshow(np.log1p(power_spectrum), cmap="gray")
        plt.title(f"Step {i}")
        plt.axis("off")

    plt.suptitle("Power Spectrum at Each Step of the Diffusion Process")
    plt.tight_layout()
    plt.savefig("../img/blogs/diffusion/power_spectrum_diffusion.png")
    plt.show()


# Load a sample image
image = data.astronaut() / 255.0

# Convert to grayscale
image = np.mean(image, axis=2)

# Plot and save the power spectrum
plot_power_spectrum(image, steps=10)


# FIGURE 4: Probability space changes


def generate_complex_distribution(n_samples=1000):
    # Generate data points from a complex distribution (e.g., a mixture of Gaussians)
    mean1 = [2, 2]
    cov1 = [[0.1, 0], [0, 0.1]]
    mean2 = [-2, -2]
    cov2 = [[0.1, 0], [0, 0.1]]

    data1 = np.random.multivariate_normal(mean1, cov1, n_samples // 2)
    data2 = np.random.multivariate_normal(mean2, cov2, n_samples // 2)

    data = np.vstack((data1, data2))
    return data


def add_noise(data, noise_level):
    noise = np.random.normal(0, noise_level, data.shape)
    noisy_data = data + noise
    return noisy_data


def forward_diffusion(data, steps=10):
    trajectories = [data]
    for i in range(steps):
        noise_level = (i + 1) / steps
        noisy_data = add_noise(data, noise_level)
        trajectories.append(noisy_data)
    return trajectories


def reverse_diffusion(data, steps=10):
    trajectories = [data]
    for i in range(steps):
        noise_level = (steps - i - 1) / steps
        noisy_data = add_noise(data, noise_level)
        trajectories.append(noisy_data)
    return trajectories


def create_gif(trajectories, title, filename):
    images = []
    for i, data in enumerate(trajectories):
        plt.figure(figsize=(5, 5))
        plt.scatter(data[:, 0], data[:, 1], alpha=0.5, s=10)
        plt.title(f"{title} - Step {i}")
        plt.axis("equal")
        plt.axis("off")

        # Save the current plot as an image
        plt.savefig(f"temp_{i}.png")
        plt.close()

        # Read the image and append to the list
        images.append(imageio.imread(f"temp_{i}.png"))

    # Create a GIF from the images
    imageio.mimsave(filename, images, duration=0.5)

    # Remove temporary images
    for i in range(len(trajectories)):
        os.remove(f"temp_{i}.png")


# Generate complex distribution data
complex_data = generate_complex_distribution()

# Forward diffusion process
forward_trajectories = forward_diffusion(complex_data, steps=10)

# Generate simple Gaussian distribution data
simple_data = np.random.normal(0, 1, (1000, 2))

# Reverse diffusion process
reverse_trajectories = reverse_diffusion(simple_data, steps=10)

# Combine forward and reverse trajectories
combined_trajectories = forward_trajectories + reverse_trajectories[::-1]

# Create and save the GIF
create_gif(
    combined_trajectories,
    "Diffusion Process",
    "../img/blogs/diffusion/diffusion_process.gif",
)

# Display the GIF
# from IPython.display import Image

# Image(filename="diffusion_process.gif")

# FIGURE 5: Noise schedule comparison


def linear_schedule(step, total_steps):
    return step / total_steps


def cosine_schedule(step, total_steps):
    return 0.5 * (1 - np.cos(np.pi * step / total_steps))


def exponential_schedule(step, total_steps):
    return (np.exp(step / total_steps) - 1) / (np.e - 1)


def plot_noise_schedules(image, steps=10):
    schedules = [linear_schedule, cosine_schedule, exponential_schedule]
    schedule_names = ["Linear", "Cosine", "Exponential"]

    plt.figure(figsize=(15, 10))

    for i, schedule in enumerate(schedules):
        diffusion_images = diffusion_process(image, steps, schedule)
        for j, img in enumerate(diffusion_images):
            plt.subplot(len(schedules), steps, i * steps + j + 1)
            plt.imshow(img, cmap="gray")
            if i == 0:
                plt.title(f"Step {j}")
            if j == 0:
                plt.ylabel(schedule_names[i])
            plt.axis("off")

    plt.suptitle("Comparison of Different Noise Schedules in Forward Diffusion Process")
    plt.tight_layout()
    plt.savefig("noise_schedules_comparison.png")
    plt.show()


# Load a sample image
image = data.astronaut() / 255.0

# Convert to grayscale
image = np.mean(image, axis=2)

# Plot and save the comparison of noise schedules
plot_noise_schedules(image, steps=10)


# FIGURE 6:


def plot_generated_samples(image_shape, steps=10):
    noise_schedule = linear_schedule
    reverse_images = reverse_diffusion(image_shape, steps, noise_schedule)

    plt.figure(figsize=(15, 5))

    for i, img in enumerate(reverse_images):
        plt.subplot(1, steps + 1, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"Step {i}")
        plt.axis("off")

    plt.suptitle("Generated Samples in Reverse Diffusion Process")
    plt.tight_layout()
    plt.savefig("generated_samples.png")
    plt.show()


# Plot and save the generated samples
plot_generated_samples(image.shape, steps=10)
