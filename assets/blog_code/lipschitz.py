import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Apply a style for consistent and professional-looking plots
mpl.rcParams.update(
    {
        "font.family": "serif",
        "text.latex.preamble": "\\usepackage{times} ",
        "figure.figsize": (3.25*2.1, 2.0086104634371584*2),
        "figure.constrained_layout.use": True,
        "figure.autolayout": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.015,
        "font.size": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.titlesize": 8,
    }
)


# Define the domain for the functions
x = np.linspace(-10, 10, 400)

# Define the functions
def linear_function(x):
    return 2 * x

def cosine_function(x):
    return np.cos(x)

def cusp_function(x):
    return np.abs(x)

def quadratic_function(x):
    return x**2

# Plot each function
plt.figure()

# Linear function plot
plt.subplot(2, 2, 1)
plt.plot(x, linear_function(x), color="green", label='Linear Function')
plt.title('Lipschitz Continuous')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()

# Sine function plot
plt.subplot(2, 2, 2)
plt.plot(x, cosine_function(x), color="green", label='Cosine Function')
plt.title('Lipschitz Continuous')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()

# Cusp function plot
plt.subplot(2, 2, 3)
plt.plot(x, cusp_function(x), color="purple", label='Cusp Function')
plt.title('Not Lipschitz Continuous')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()

# Quadratic function plot
plt.subplot(2, 2, 4)
plt.plot(x, quadratic_function(x), color="blue", label='Quadratic Function')
plt.title('Lipschitz Continuous on Bounded Intervals')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()

# Adjust layout and show plot
# plt.tight_layout()
plt.savefig("lipschitz_curves.png", dpi=200)


# x_local = np.linspace(-2, 2, 400)

# # Plot each function
# plt.figure()

# # Linear function plot
# plt.subplot(2, 2, 1)
# plt.plot(x_local, linear_function(x_local), color="green", label='Linear Function')
# plt.title('Locally Lipschitz Continuous')
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.legend()

# # Sine function plot
# plt.subplot(2, 2, 2)
# plt.plot(x_local, cosine_function(x_local), color="green", label='Cosine Function')
# plt.title('Locally Lipschitz Continuous')
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.legend()

# # Cusp function plot
# plt.subplot(2, 2, 3)
# plt.plot(x_local, cusp_function(x_local), color="purple", label='Cusp Function')
# plt.title('Locally Lipschitz Continuous (except at 0)')
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.legend()

# # Quadratic function plot
# plt.subplot(2, 2, 4)
# plt.plot(x_local, quadratic_function(x_local), color="blue", label='Quadratic Function')
# plt.title('Locally Lipschitz Continuous')
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.legend()

# # Adjust layout and show plot
# # plt.tight_layout()
# plt.savefig("lipschitz_curves.png", dpi=200)