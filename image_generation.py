import numpy as np
import matplotlib.pyplot as plt

from skimage.data import shepp_logan_phantom
from skimage.transform import resize

# 7(a) Population generation
# Define parameters for grid, signal, and background
grid_size = 64  # 64x64 pixels
signal_center = (0, 0) # Center of the signal (0, 0)
sigma_s = 1 / 4  # Signal radius (σ_s)
A_s = 3  # Signal amplitude
N = 10  # Number of lumps
sigma_n = sigma_s  # Lump width
a_n = 1  # Lump amplitude
M = 32 * 16 # Total number of sensitivity functions


def generate_fov_mask(grid_size, radius=0.5):
    """
    Generate a circular FOV mask for the image.

    Parameters:
    - grid_size (int): Size of the square grid (e.g., 64x64).
    - radius (float): Radius of the circular FOV, in normalized units (0 to 0.5).

    Returns:
    - mask (numpy.ndarray): A binary mask of shape (grid_size, grid_size).
    """
    x = np.linspace(-0.5, 0.5, grid_size)
    y = np.linspace(-0.5, 0.5, grid_size)
    r_x, r_y = np.meshgrid(x, y)
    mask = (r_x**2 + r_y**2) <= radius**2
    return mask


def circ(r, center_coords, sigma_s):
    r_x, r_y = r
    c_x, c_y = center_coords
    return (r_x - c_x) ** 2 + (r_y-c_y) ** 2 <= sigma_s ** 2

def rect(x):
    return np.abs(x) < 0.5

def pixelated(r_x, r_y):
    return np.logical_and(rect(r_x), rect(r_y))

def generate_signal(grid_size, signal_center, sigma_s = 1/4, A_s = 3):
    x = np.linspace(-1, 1, grid_size) # rect(r)
    y = np.linspace(-1, 1, grid_size)
    r_x, r_y = np.meshgrid(x, y)
    signal = circ(r=(r_x, r_y), center_coords=(signal_center[0], signal_center[1]), sigma_s=sigma_s)
    pixel = pixelated(r_x, r_y)
    signal = A_s * pixel * signal
    # r = np.sqrt((r_x - signal_center[0])**2 + (r_y - signal_center[1])**2)
    # signal = np.zeros_like(r)
    # signal[r <= sigma_s] = A_s
    return signal

def generate_background(grid_size, N=10, sigma_n=1/4, a_n=1, num=500):
    background = np.zeros((grid_size, grid_size))
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    r_x, r_y = np.meshgrid(x, y)
    pixel = pixelated(r_x, r_y)

    for _ in range(N):
        lump_center_x = np.random.uniform(-0.5, 0.5)
        lump_center_y = np.random.uniform(-0.5, 0.5)
        coeff = a_n / (2 * np.pi * sigma_n ** 2)
        lump = coeff * np.exp(-((r_x - lump_center_x)**2 + (r_y - lump_center_y)**2) / (2 * sigma_n**2))
        background += lump

    return pixel * background

def generate_signal_present(signal, background, A_s=3):
    """
    Merge the signal and background into a single image.
    Scale the signal amplitude and add it to the background.
    """
    signal_present = background + A_s * signal
    return signal_present

signal = generate_signal(grid_size, signal_center)
background = generate_background(grid_size)
signal_present = generate_signal_present(signal, background)
signal_absent = background  # Signal-absent

# Generate objects with size J
def generate_multiple_objects(signal_present, signal_absent, grid_size, J):
    """
    Generate J realizations of signal-present and signal-absent objects
    """
    signal_present_objects = np.zeros((J, grid_size, grid_size))
    signal_absent_objects = np.zeros((J, grid_size, grid_size))

    for j in range(J):
        # signal_present, signal_absent, _, _ = generate_object_realization(size)
        signal_present_objects[j] = signal_present
        signal_absent_objects[j] = signal_absent

    return signal_present_objects, signal_absent_objects

# Plot the signal, background, and combined object
def plot_generated_objects(signal, background, signal_present):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plot signal
    axs[0].imshow(signal, extent=[-1, 1, -1, 1], cmap="viridis", origin="lower")
    axs[0].set_title("Signal (fs)")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].axis('off')

    # Plot background
    axs[1].imshow(background, extent=[-1, 1, -1, 1], cmap="viridis", origin="lower")
    axs[1].set_title("Background (fb)")
    axs[1].set_xlabel("x")
    axs[1].axis('off')

    # Plot combined object
    axs[2].imshow(signal_present, extent=[-1, 1, -1, 1], cmap="viridis", origin="lower")
    axs[2].set_title("Object with Signal (f = fs + fb)")
    axs[2].set_xlabel("x")
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()
    # plt.close()

plot_generated_objects(signal, background, signal_present)


# Part 7(b) Image generation
# Initialize lists to store results for different pixel sizes
# pixel_sizes = [64, 32, 8]  # Dimensions for discretization
pixel_sizes = 32 # If focus on one size


def downsample(signal_present, signal_absent, factor=2):
    """Downsample the image by averaging over non-overlapping blocks."""
    # Get the shape of the downsampled image
    new_size = signal_present.shape[0] // factor
    # Reshape and compute the mean
    signal_present = signal_present.reshape(new_size, factor, new_size, factor).mean(axis=(1, 3))
    signal_absent = signal_absent.reshape(new_size, factor, new_size, factor).mean(axis=(1, 3))
    return signal_present, signal_absent


def generate_system_matrix(M, pixel_size, fov_mask=None):
    """
    Generate a system matrix with optional FOV masking.
    
    Parameters:
    - M (int): Total number of sensitivity functions.
    - pixel_size (int): Size of the image grid (e.g., 32x32).
    - fov_mask (numpy.ndarray): Optional FOV mask to apply (same shape as the image grid).

    Returns:
    - H (numpy.ndarray): System matrix of shape (M, pixel_size^2).
    """
    num_strips = 32  # Number of horizontal strips
    num_rotations = 16
    H = np.zeros((M, pixel_size**2))
    x = np.linspace(-0.5, 0.5, pixel_size)
    y = np.linspace(-0.5, 0.5, pixel_size)
    xx, yy = np.meshgrid(x, y)

    if fov_mask is not None:
        fov_mask_resized = resize(fov_mask.astype(float), (pixel_size, pixel_size), mode='constant', anti_aliasing=False)
        fov_mask_resized = fov_mask_resized > 0.5  # Convert back to binary
        H *= fov_mask_resized.flatten()
    else:
        fov_mask_resized = np.ones((pixel_size, pixel_size))
        
    for r in range(num_rotations):
        rotation_angle = r * (np.pi / num_rotations)
        rotation_matrix = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)]
        ])
        rotated_coordinates = np.dot(rotation_matrix, np.vstack([xx.ravel(), yy.ravel()]))
        r_y = rotated_coordinates[1, :].reshape(xx.shape)
        sensitivity = np.zeros_like(r_y)

        for m in range(num_strips):
            y_min = -0.5 + m / 32
            y_max = -0.5 + (m + 1) / 32
            strip_index = ((y_min < r_y) & (r_y <= y_max))
            sensitivity[strip_index] = 1

            # Apply the FOV mask
            sensitivity *= fov_mask_resized
            matrix_index = r * num_strips + m
            H[matrix_index, :] = sensitivity.ravel()

    return H


def generate_pseudoinverse(size, H, reg_param=1e-2):
    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    S_reg_inv = np.diag([1/s if s > reg_param else 0 for s in S])
    H_MP_reg = Vt.T @ S_reg_inv @ U.T
    return H_MP_reg


def add_noise_levels(projection_data, noise_levels):
    """
    Add Poisson noise to the projection data for multiple noise levels.

    Parameters:
    - projection_data (numpy.ndarray): Clean projection data (ḡ).
    - noise_levels (dict): Noise levels with total intensity sums.

    Returns:
    - noisy_projections (dict): Projection data with noise added for each level.
    """
    noisy_projections = {}
    for level, total_sum in noise_levels.items():
        # Scale projection data to the desired total intensity
        scaled_data = projection_data * (total_sum / np.sum(projection_data, axis=1, keepdims=True))
        # Add Poisson noise
        noisy_projections[level] = np.random.poisson(scaled_data)
    return noisy_projections

# Add Poisson noise at different levels
def generate_reconstruction(projection_data, H_MP):
    noise_levels = {
        "Low": 50000,
        "Medium": 25000,
        "High": 10000,
        "Very_High": 2500
    }
    noisy_projections = {}

    for level, scale in noise_levels.items():
        scaled_projections = projection_data * (scale / np.sum(projection_data, axis=1, keepdims=True))
        noisy_projections[level] = np.random.poisson(scaled_projections)

    reconstructed_images = {}

    for level, noisy_data in noisy_projections.items():
        reconstructed_images[level] = np.dot(H_MP, noisy_data.T).T

    return reconstructed_images

# Add Poisson noise at different levels
def generate_noise(projection_data):
    noisy_projections = {}

    for level, scale in noise_levels.items():
        scaled_projections = projection_data * (scale / np.sum(projection_data, axis=1, keepdims=True))
        noisy_projections[level] = np.random.poisson(scaled_projections)

    return noisy_projections


# Visualize a sample reconstruction for each noise level
def plot_reconstruction(num_realizations, reconstructed_images, original_image, selected_size):
    sample_index = np.random.randint(num_realizations)
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    axs[0].imshow(original_image[sample_index].reshape(selected_size, selected_size), cmap='viridis')
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    for i, (level, recon_images) in enumerate(reconstructed_images.items()):
        axs[i + 1].imshow(recon_images[sample_index].reshape(selected_size, selected_size), cmap='viridis')
        axs[i + 1].set_title(f'Reconstruction ({level} Noise)')

    plt.tight_layout()
    plt.show()
    # plt.close()

# Generate FOV mask
fov_mask = generate_fov_mask(grid_size, radius=0.5)

# Generate signal, background, and signal-present
signal = generate_signal(grid_size, signal_center, sigma_s=1/4, A_s=5)
background = generate_background(grid_size)
signal_present = generate_signal_present(signal, background)

# Apply FOV mask to all images
signal_present = signal_present * fov_mask
signal_absent = background * fov_mask
background = background * fov_mask  # Ensure consistency

num_realizations = 500
selected_size = 32
noise_levels = {
    "Low": 50000,
    "Medium": 25000,
    "High": 10000,
    "Very_High": 2500
}

H = generate_system_matrix(M, pixel_sizes, fov_mask=fov_mask)
H_MP = generate_pseudoinverse(selected_size, H) # Print singular values and vectors or not?

# Downsampling a 64x64 image to 32x32
downsampled_present, downsampled_absent = downsample(signal_present, signal_absent)  # Downsample by 2
# downsampled_absent = downsample(signal_absent)
print("Original shape:", signal_present.shape)
print("Downsampled signal present:", downsampled_present.shape)
print("Downsampled signal absent:", downsampled_absent.shape)

# signal_present_objects, signal_absent_objects = generate_multiple_objects(signal_present, signal_absent, grid_size = selected_size, J = 500)
signal_present_objects, signal_absent_objects = generate_multiple_objects(downsampled_present, downsampled_absent, grid_size = selected_size, J = 500)
signal_present_realizations = signal_present_objects.reshape(500, -1)
signal_absent_realizations = signal_absent_objects.reshape(500, -1)

projection_data_present = np.dot(H, signal_present_realizations.T).T  # g = Hf, (512, 32x32) * (500, 32x32)^T = (512, 500)
projection_data_absent = np.dot(H, signal_absent_realizations.T).T # (512, 500)^T = (500, 512)

# Apply noise levels and visualize projections
for noise_level, noise_scale in noise_levels.items():
    # Scale the projection data
    scaled_projections = projection_data_present * (noise_scale / np.sum(projection_data_present, axis=1, keepdims=True))
    noisy_projections = np.random.poisson(scaled_projections)

    # # Visualize the noisy projection
    # plt.imshow(noisy_projections[0].reshape(16, 32), cmap="viridis")
    # plt.title(f"Noisy Projection ({noise_level})")
    # plt.show()
    
reconstructed_present = generate_reconstruction(projection_data_present, H_MP)
reconstructed_absent = generate_reconstruction(projection_data_absent, H_MP)

reconstructed__present_graph = plot_reconstruction(num_realizations, reconstructed_present, signal_present_objects, selected_size)
reconstructed__absent_graph = plot_reconstruction(num_realizations, reconstructed_absent, signal_absent_objects, selected_size)

scaled_projections = projection_data_present * (noise_scale / np.sum(projection_data_present, axis=1, keepdims=True))
noisy_projections = np.random.poisson(scaled_projections)
