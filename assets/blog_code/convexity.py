import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Apply a style for consistent and professional-looking plots
mpl.rcParams.update(
    {
        "font.family": "serif",
        "text.latex.preamble": "\\usepackage{times} ",
        "figure.figsize": (6.25, 4.0086104634371584),
        "figure.constrained_layout.use": True,
        "figure.autolayout": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.015,
        "font.size": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 10,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.titlesize": 8,
    }
)

# Define the range for the x-axis
x = np.linspace(-2, 2, 100)

# Define the functions
def strongly_convex(x, mu=2):
    return 0.5 * mu * (x - 0.5)**2 + 0.25

def strictly_convex(x):
    return x**4

def convex(x):
    return x**2

# # Plot each function
# plt.figure()

# # Strongly convex
# plt.plot(x, strongly_convex(x), label='Strongly Convex ($(x-0.5)^2 + 0.25$)', color='blue', linewidth=2)

# # Strictly convex
# plt.plot(x, strictly_convex(x), label='Strictly Convex ($x^{4}$ for $x \\neq 0$)', color='green', linewidth=2)

# # Convex
# plt.plot(x, convex(x), label='Convex ($x^2$)', color='red', linestyle='--', linewidth=2)

# # Styling the plot
# plt.title('Examples of Convex Functions', fontsize=16)
# plt.xlabel('x', fontsize=14)
# plt.ylabel('f(x)', fontsize=14)
# plt.axhline(0, color='black',linewidth=0.5)
# plt.axvline(0, color='black',linewidth=0.5)
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.legend()
# # plt.tight_layout()

# # Show the plot
# plt.show()
# plt.savefig("strong_and_strict.png")


# Define the range for the x-axis
x = np.linspace(-1, 1, 100)

# Choose two points and their weights
x1, x2 = -0.5, 0.8
lambda1, lambda2 = 0.6, 0.4

# Calculate the weighted average of the points
x_avg = lambda1 * x1 + lambda2 * x2

# Calculate the function values at the points and their weighted average
f_x1 = convex(x1)
f_x2 = convex(x2)
f_x_avg = convex(x_avg)

# Calculate the right-hand side of Jensen's Inequality
rhs_jensen = lambda1 * f_x1 + lambda2 * f_x2

# Plot the convex function
plt.figure()
plt.plot(x, convex(x), label='Convex Function $f(x) = x^2$', color='blue')

# Plot the points and their weighted average
plt.scatter([x1, x2], [f_x1, f_x2], color='red', zorder=5)
plt.scatter(x_avg, f_x_avg, color='green', zorder=5)

# Plot the line segment between the points
plt.plot([x1, x2], [f_x1, f_x2], 'red', linestyle='--', label='Line Segment between $f(x_1)$ and $f(x_2)$')

# Plot the vertical line for the weighted average
plt.vlines(x_avg, min(f_x_avg, rhs_jensen), max(f_x_avg, rhs_jensen), color='green', linestyle=':', label="Jensen's Inequality")

# Annotate the points
plt.annotate('$x_1$', (x1, f_x1), textcoords="offset points", xytext=(-15,-10), ha='center')
plt.annotate('$x_2$', (x2, f_x2), textcoords="offset points", xytext=(-15,-10), ha='center')
plt.annotate('Weighted Average', (x_avg, f_x_avg), textcoords="offset points", xytext=(0,10), ha='center')

# Styling the plot
plt.title("Visualization of Jensen's Inequality")
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
# plt.tight_layout()

plt.savefig("jensens.png")

from matplotlib.animation import FuncAnimation

# Apply a style for consistent and professional-looking plots
mpl.rcParams.update(
    {
        "font.family": "serif",
        "text.latex.preamble": "\\usepackage{times} ",
        "figure.figsize": (6.25, 4.0086104634371584),
        "figure.constrained_layout.use": True,
        "figure.autolayout": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.015,
        "font.size": 8,
        "axes.labelsize": 10,
        "legend.fontsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.titlesize": 10,
    }
)

# Initialize the figure
fig, ax = plt.subplots()

# Plot the convex function
ax.plot(x, convex(x), label='Convex Function $f(x) = x^2$', color='blue')
ax.set_title("Visualization of Jensen's Inequality")
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Initialize the points, line, and annotations
points, = ax.plot([], [], 'ro', zorder=5)
line, = ax.plot([], [], 'r--', zorder=5)
weighted_point, = ax.plot([], [], 'go', zorder=5)
vline = ax.axvline(x=0, color='green', linestyle=':', zorder=5)

# Annotation for weighted average
weighted_annotation = ax.annotate('', xy=(0.7, 0.1), xytext=(0,10), textcoords="offset points", ha='center', color='green')

# Animation update function
def update(frame):
    # Generate x1 and x2 using sine and cosine functions for smooth transitions
    x1 = np.sin(frame * np.pi / 50)  # Period of 100 frames
    x2 = np.cos(frame * np.pi / 50)  # Period of 100 frames
    lambda1, lambda2 = 0.6, 0.4  # Fixed weights for simplicity
    x_avg = lambda1 * x1 + lambda2 * x2
    f_x1, f_x2, f_x_avg = convex(x1), convex(x2), convex(x_avg)
    rhs_jensen = lambda1 * f_x1 + lambda2 * f_x2
    
    # Update points and line
    points.set_data([x1, x2], [f_x1, f_x2])
    line.set_data([x1, x2], [f_x1, f_x2])
    weighted_point.set_data(x_avg, f_x_avg)
    
    # Update vertical line for Jensen's Inequality
    vline.set_xdata(x_avg)
    
    # Update annotation
    weighted_annotation.set_position((x_avg, f_x_avg))
    weighted_annotation.set_text(f'Weighted Avg\n({x_avg:.2f}, {f_x_avg:.2f})')
    
    return points, line, weighted_point, vline, weighted_annotation


# Create the animation
ani = FuncAnimation(fig, update, frames=100, interval=100, blit=True)

# Save the animation as a GIF
ani.save('jensens_inequality.gif', writer='imagemagick')

# Close the plot display
plt.close(fig)