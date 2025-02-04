import numpy as np
import matplotlib.pyplot as plt
import imageio
from skimage import data
from numpy.fft import fft2, fftshift
import os
import seaborn as sns

from pysteps.utils import spectral


# FIGURE 1: Diffusion process
def add_noise(data, noise_level):
    noise = np.random.normal(0, 1.0, data.shape)
    noisy_data = data + noise_level * noise
    return noisy_data


def diffusion_process(data, steps=10, noise_scale=100):
    datas = []
    for i in range(steps + 1):
        noise_level = np.log10(noise_scale) * i / (steps - 1)
        noisy_image = add_noise(data, noise_level)
        datas.append(noisy_image)
    return datas


# FIGURE 2: Reverse diffusion process
def reverse_diffusion_process(data, steps=10, noise_scale=100):
    datas = []
    for i in range(steps + 1):
        noise_level = np.log10(noise_scale) * (steps - i) / steps
        noisy_image = add_noise(data, noise_level)
        datas.append(noisy_image)
    return datas


# Load a sample image
image = data.astronaut() / 255.0

# Convert to grayscale
image = np.mean(image, axis=2)

# Generate diffusion process images
diffusion_images = diffusion_process(image, steps=20, noise_scale=10)

# Generate reverse diffusion process images
reverse_diffusion_images = diffusion_images[::-1]

diffusion_images.extend(reverse_diffusion_images)

# Save as GIF
imageio.mimsave(
    "../img/blogs/diffusion/diffusion_process.gif",
    [np.uint8(img * 255) for img in diffusion_images],
    duration=2.0,
)

# # Display the GIF
# Image(filename="../img/blogs/diffusion/reverse_diffusion_process.gif")


# FIGURE 3: Power spectrum
def compute_power_spectrum(image):
    f_transform = fft2(image)
    f_transform_shifted = fftshift(f_transform)
    power_spectrum = np.abs(f_transform_shifted) ** 2
    return power_spectrum


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


def plot_power_spectrum(image, steps=10, noise_scale=200):
    images = []

    diffusion_images = diffusion_process(image, steps, noise_scale)

    for i, img in enumerate(diffusion_images):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        power_spectrum = compute_power_spectrum(img)
        rapsd, frequencies = spectral.rapsd(img, fft_method=np.fft, return_freq=True)
        plt.plot(
            frequencies[1:], rapsd[1:], c="red", marker="o", markersize=3
        )  # Chop off the DC component.
        plt.xscale("log")
        plt.yscale("log")
        plt.ylim(0.0003, 1000)
        plt.title(f"Step {i}")

        plt.subplot(1, 2, 2)
        plt.imshow(img, cmap="gray")
        plt.axis("off")

        # Save the current plot as an image
        plt.savefig(f"temp_{i}.png")
        plt.close()

        # Read the image and append to the list
        images.append(imageio.imread(f"temp_{i}.png"))

    # Remove temporary images
    for i in range(len(diffusion_images)):
        os.remove(f"temp_{i}.png")

    diffusion_images = reverse_diffusion_process(image, steps, noise_scale)

    for i, img in enumerate(diffusion_images):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        power_spectrum = compute_power_spectrum(img)
        rapsd, frequencies = spectral.rapsd(img, fft_method=np.fft, return_freq=True)
        plt.plot(
            frequencies[1:], rapsd[1:], c="red", marker="o", markersize=3
        )  # Chop off the DC component.
        plt.xscale("log")
        plt.yscale("log")
        plt.ylim(0.0003, 1000)
        plt.title(f"Step {steps - i}")

        plt.subplot(1, 2, 2)
        plt.imshow(img, cmap="gray")
        plt.axis("off")

        # Save the current plot as an image
        plt.savefig(f"temp_{i}.png")
        plt.close()

        # Read the image and append to the list
        images.append(imageio.imread(f"temp_{i}.png"))

    # Remove temporary images
    for i in range(len(diffusion_images)):
        os.remove(f"temp_{i}.png")

    # Create a GIF from the images
    imageio.mimsave(
        "../img/blogs/diffusion/power_spectrum_diffusion.gif", images, duration=1.0
    )


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


def create_joint_plot_gif(trajectories, title, filename):
    images = []
    for i, data in enumerate(trajectories):
        fig, axs = plt.subplots(
            2,
            2,
            figsize=(8, 6),
            gridspec_kw={
                "hspace": 0,
                "wspace": 0,
                "width_ratios": [5, 1],
                "height_ratios": [1, 5],
            },
        )
        # Upper part charts
        sns.distplot(data[:, 0], bins=20, ax=axs[0, 0], color="LightBlue")

        axs[0, 0].axis("off")
        axs[0, 1].axis("off")
        axs[1, 1].axis("off")

        # Right part charts
        sns.distplot(
            data[:, 1], bins=20, ax=axs[1, 1], color="LightBlue", vertical=True
        )

        # KDE middle part
        sns.kdeplot(
            x=data[:, 0],
            y=data[:, 1],
            fill=True,
            thresh=0.05,
            cmap="Blues",
            ax=axs[1, 0],
        )

        # Save the current plot as an image
        plt.savefig(f"temp_{i}.png")
        plt.close()

        # Read the image and append to the list
        images.append(imageio.imread(f"temp_{i}.png"))

    # Create a GIF from the images
    imageio.mimsave(filename, images, duration=1.0)

    # Remove temporary images
    for i in range(len(trajectories)):
        os.remove(f"temp_{i}.png")


# Generate complex distribution data
complex_data = generate_complex_distribution()

# Forward diffusion process
forward_trajectories = diffusion_process(complex_data, steps=20, noise_scale=200)

# Combine forward and reverse trajectories
combined_trajectories = forward_trajectories + forward_trajectories[::-1][1:]

# Create and save the GIF
create_joint_plot_gif(
    combined_trajectories,
    "Diffusion Process",
    "../img/blogs/diffusion/diffusion_process_probability.gif",
)

# Display the GIF
# from IPython.display import Image

# Image(filename="diffusion_process.gif")


# FIGURE 5: Noise schedule comparison
def linear_schedule(step, total_steps):
    return step / total_steps


# def cosine_schedule(step, total_steps):
#     return 0.5 * (1 - np.cos(np.pi * step / total_steps))


# def exponential_schedule(step, total_steps):
#     return (np.exp(step / total_steps) - 1) / (np.e - 1)


# def plot_noise_schedules(image, steps=10):
#     schedules = [linear_schedule, cosine_schedule, exponential_schedule]
#     schedule_names = ["Linear", "Cosine", "Exponential"]

#     plt.figure(figsize=(15, 10))

#     for i, schedule in enumerate(schedules):
#         diffusion_images = diffusion_process(image, steps, schedule)
#         for j, img in enumerate(diffusion_images):
#             plt.subplot(len(schedules), steps, i * steps + j + 1)
#             plt.imshow(img, cmap="gray")
#             if i == 0:
#                 plt.title(f"Step {j}")
#             if j == 0:
#                 plt.ylabel(schedule_names[i])
#             plt.axis("off")

#     plt.suptitle("Comparison of Different Noise Schedules in Forward Diffusion Process")
#     plt.tight_layout()
#     plt.savefig("noise_schedules_comparison.png")
#     plt.show()


# # Load a sample image
# image = data.astronaut() / 255.0

# # Convert to grayscale
# image = np.mean(image, axis=2)

# # Plot and save the comparison of noise schedules
# plot_noise_schedules(image, steps=10)


# # FIGURE 6:
def plot_generated_samples(image, steps=10):
    noise_schedule = linear_schedule
    reverse_images = reverse_diffusion_process(image, steps, 200)

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
plot_generated_samples(image, steps=10)
