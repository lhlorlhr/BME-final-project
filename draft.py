import numpy as np
import matplotlib.pyplot as plt

from skimage.data import shepp_logan_phantom
from skimage.transform import resize


# Define parameters for the grid and object
grid_size = 32  # 64x64 pixels
signal_center = (0, 0) # Center of the signal (0, 0)
sigma_s = 1 / 4  # Signal radius (σ_s)
A_s = 3  # Signal amplitude

def generate_grid(grid_size):
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    return X, Y

X, Y = generate_grid(grid_size)

def generate_signal(X, Y, signal_center, sigma_s, A_s):
    # Parameters for the signal (circular model)
    r = np.sqrt((X - signal_center[0])**2 + (Y - signal_center[1])**2)
    signal = np.zeros_like(r)
    signal[r <= sigma_s] = A_s
    return signal


# Parameters for the background (lumpy model)
N = 10  # Number of lumps
sigma_n = sigma_s  # Lump width
a_n = 1  # Lump amplitude


def generate_background(grid_size, X, Y):
    # Initialize background
    background = np.zeros((grid_size, grid_size))

    # Generate the Gaussian lumps
    for _ in range(N):
        lump_center_x = np.random.uniform(-0.5, 0.5)
        lump_center_y = np.random.uniform(-0.5, 0.5)
        lump = a_n * np.exp(-((X - lump_center_x)**2 + (Y - lump_center_y)**2) / (2 * sigma_n**2))
        background += lump

    return background

signal = generate_signal(X, Y, signal_center, sigma_s, A_s)
background = generate_background(grid_size, X, Y)
signal_present = signal + background
signal_absent = background  # Signal-absent object


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


# # Plot the signal, background, and combined object
# fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# # Plot signal
# axs[0].imshow(signal, extent=[-1, 1, -1, 1], cmap="hot", origin="lower")
# axs[0].set_title("Signal (fs)")
# axs[0].set_xlabel("x")
# axs[0].set_ylabel("y")

# # Plot background
# axs[1].imshow(background, extent=[-1, 1, -1, 1], cmap="hot", origin="lower")
# axs[1].set_title("Background (fb)")
# axs[1].set_xlabel("x")

# # Plot combined object
# axs[2].imshow(signal_present, extent=[-1, 1, -1, 1], cmap="hot", origin="lower")
# axs[2].set_title("Object with Signal (f = fs + fb)")
# axs[2].set_xlabel("x")

# plt.tight_layout()
# plt.show()


# Part 7(b) for image generation
# Initialize lists to store results for different pixel sizes
# pixel_sizes = [64, 32, 8]  # Dimensions for discretization
pixel_sizes = [32]
system_matrices = {}  # To store system matrices for each pixel size
svd_results = {}  # To store SVD results (singular values and vectors)
pseudoinverses = {}  # To store pseudoinverses

for size in pixel_sizes:
    # Define grid and corresponding object representation
    x = np.linspace(-0.5, 0.5, size)
    y = np.linspace(-0.5, 0.5, size)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)

    # System matrix simulation (for System 1)
    system_matrix = []
    num_strips = size  # Number of horizontal strips
    strip_width = 1 / num_strips  # Width of each strip

    for m in range(num_strips):
        strip = ((-0.5 + m * strip_width <= yy) & (yy < -0.5 + (m + 1) * strip_width)).astype(float)
        system_matrix.append(strip.flatten())

    # Rotate strips to simulate sensitivity functions
    full_system_matrix = []
    for rotation in range(size):
        rotated_matrix = np.roll(system_matrix, shift=rotation, axis=0)
        full_system_matrix.append(rotated_matrix)

    # Convert system matrix to numpy array
    H = np.vstack(full_system_matrix)
    system_matrices[size] = H

    # Compute SVD of the system matrix
    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    svd_results[size] = {'U': U, 'S': S, 'Vt': Vt}

    # Compute the pseudoinverse of the system operator
    S_inv = np.diag(1 / S)
    pseudoinverse = Vt.T @ S_inv @ U.T
    pseudoinverses[size] = pseudoinverse

# # Visualize singular value spectra for each pixel size
# plt.figure(figsize=(12, 6))
# for size in pixel_sizes:
#     plt.plot(svd_results[size]['S'], label=f'{size}x{size} Pixels')
# plt.yscale('log')
# plt.xlabel('Index')
# plt.ylabel('Singular Value (log scale)')
# plt.title('Singular Value Spectra for Different Pixel Sizes')
# plt.legend()
# plt.show()

# # Save the results for future use
# np.savez('image_generation_results.npz', system_matrices=system_matrices, svd_results=svd_results, pseudoinverses=pseudoinverses)

# Load previously computed pseudoinverses and system matrices
num_realizations = 500
selected_size = grid_size # Example pixel size
H = system_matrices[selected_size]  # System matrix for the selected pixel size
H_pseudo = pseudoinverses[selected_size]  # Pseudoinverse for reconstruction

# Simulated object realizations (signal-present objects)
# signal_present_realizations = np.random.rand(num_realizations, selected_size**2)  # Mock for demonstration
signal_present_objects, signal_absent_objects = generate_multiple_objects(signal_present, signal_absent, grid_size = selected_size, J = 500)
signal_present_realizations = signal_present_objects.reshape(500, -1)
signal_absent_realizations = signal_absent_objects.reshape(500, -1)

# Step iv: Generate projection data
projection_data_present = np.dot(H, signal_present_realizations.T).T  # g = Hf, (500, grid x grid)
projection_data_absent = np.dot(H, signal_absent_realizations.T).T
projection_data = [projection_data_present, projection_data_absent]
projection_data = projection_data[1]

# # Step iv example: Generate projection data of shepp_logan_phantom
# phantom = shepp_logan_phantom()
# phantom_resized = resize(phantom, (32, 32), mode='reflect', anti_aliasing=True)

# my_image = phantom_resized
# f_input = my_image.flatten()
# g_input = np.dot(H, f_input)
# projection_data = g_input

# Step v: Add Poisson noise at different levels
noise_levels = {
    "Low": 50000,
    "Medium": 25000,
    "High": 10000,
    "Very High": 2500
}
noisy_projections = {}

for level, scale in noise_levels.items():
    scaled_projections = projection_data * (scale / np.sum(projection_data, axis=1, keepdims=True))
    noisy_projections[level] = np.random.poisson(scaled_projections)

# Step vi: Reconstruct images using the pseudoinverse
reconstructed_images = {}

for level, noisy_data in noisy_projections.items():
    reconstructed_images[level] = np.dot(H_pseudo, noisy_data.T).T


# print(reconstructed_images['Low'].shape) # (500, grid x grid)

# Visualize a sample reconstruction for each noise level
sample_index = np.random.randint(num_realizations)
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
for i, (level, recon_images) in enumerate(reconstructed_images.items()):
    axs[i].imshow(recon_images[sample_index].reshape(selected_size, selected_size), cmap='viridis')
    axs[i].set_title(f'Reconstruction ({level} Noise)')
plt.tight_layout()
plt.show()

# # Save the results for further use
# np.savez('reconstruction_results.npz', projection_data=projection_data,
#          noisy_projections=noisy_projections, reconstructed_images=reconstructed_images)


# # 7(c) for CNN training
# # Shuffle the datasets
# np.random.seed(42)  # For reproducibility
# indices = np.random.permutation(500)

# # Split indices for training, validation, and testing
# train_indices = indices[:300]
# val_indices = indices[300:400]
# test_indices = indices[400:]