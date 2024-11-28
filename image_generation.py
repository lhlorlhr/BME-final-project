import numpy as np
import matplotlib.pyplot as plt

from skimage.data import shepp_logan_phantom
from skimage.transform import resize

# 7(a) Population generation
# Define parameters for grid, signal, and background
grid_size = 32  # 64x64 pixels
signal_center = (0, 0) # Center of the signal (0, 0)
sigma_s = 1 / 4  # Signal radius (σ_s)
A_s = 3  # Signal amplitude
N = 10  # Number of lumps
sigma_n = sigma_s  # Lump width
a_n = 1  # Lump amplitude
M = 32 * 16 # Total number of sensitivity functions


def generate_grid(grid_size):
    x = np.linspace(-1/2, 1/2, grid_size) # rect(r)
    y = np.linspace(-1/2, 1/2, grid_size)
    r_x, r_y = np.meshgrid(x, y)
    return r_x, r_y


X, Y = generate_grid(grid_size)


def generate_signal(X, Y, signal_center, sigma_s, A_s):
    r = np.sqrt((X - signal_center[0])**2 + (Y - signal_center[1])**2)
    signal = np.zeros_like(r)
    signal[r <= sigma_s] = A_s
    return signal


def generate_background(grid_size, X, Y):
    background = np.zeros((grid_size, grid_size))

    for _ in range(N):
        lump_center_x = np.random.uniform(-0.5, 0.5)
        lump_center_y = np.random.uniform(-0.5, 0.5)
        coeff = a_n / (2 * np.pi * sigma_n ** 2)
        lump = coeff * np.exp(-((X - lump_center_x)**2 + (Y - lump_center_y)**2) / (2 * sigma_n**2))
        background += lump

    return background

signal = generate_signal(X, Y, signal_center, sigma_s, A_s)
background = generate_background(grid_size, X, Y)
signal_present = signal + background # Signal-present
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
    axs[0].imshow(signal, extent=[-1, 1, -1, 1], cmap="hot", origin="lower")
    axs[0].set_title("Signal (fs)")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")

    # Plot background
    axs[1].imshow(background, extent=[-1, 1, -1, 1], cmap="hot", origin="lower")
    axs[1].set_title("Background (fb)")
    axs[1].set_xlabel("x")

    # Plot combined object
    axs[2].imshow(signal_present, extent=[-1, 1, -1, 1], cmap="hot", origin="lower")
    axs[2].set_title("Object with Signal (f = fs + fb)")
    axs[2].set_xlabel("x")

    plt.tight_layout()
    plt.show()
    plt.close()


# Part 7(b) Image generation
# Initialize lists to store results for different pixel sizes
# pixel_sizes = [64, 32, 8]  # Dimensions for discretization
pixel_sizes = [32] # If focus on one size


def generate_system_matrix(M, pixel_sizes):
    system_matrix = []
    num_strips = 32  # Number of horizontal strips = Number of sensitivity functions
    strip_width = 1 / 16  # Width is fixed
    num_rotations = 16
    for size in pixel_sizes:
        H = np.zeros((M, size**2))
        x = np.linspace(-0.5, 0.5, size)
        y = np.linspace(-0.5, 0.5, size)
        xx, yy = np.meshgrid(x, y)
        # fov_mask = (xx**2 + yy**2) <= 1

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
                y_min = -1 + m / 16
                y_max = -1 + (m + 1) / 16
                strip_index = ((y_min < r_y) & (r_y <= y_max))
                sensitivity[strip_index] = 1
                matrix_index = r * num_strips + m
                H[matrix_index, :] = sensitivity.ravel()

        system_matrix.append(H)


            # rotated_strip = np.zeros_like(strip_index)
            # for i in range(size):
            #     for j in range(size):
            #         x_rot = cos_angle * xx[i, j] - sin_angle * yy[i, j]
            #         y_rot = sin_angle * xx[i, j] + cos_angle * yy[i, j]
            #         if y_min <= y_rot < y_max:
            #             rotated_strip[i, j] = 1
            # rotated_strip *= fov_mask
            # system_matrix.append(rotated_strip.flatten())
                # Rotate strips to simulate sensitivity functions
    # full_system_matrix = []
    # for rotation in range(size):
    #     rotated_matrix = np.roll(system_matrix, shift=rotation, axis=0)
    #     full_system_matrix.append(rotated_matrix)

    # # Convert system matrix to numpy array
    # H = np.vstack(full_system_matrix)
    # system_matrices[size] = H
    return system_matrix


def generate_pseudoinverse(size, H):
    svd_results = {}  # Store SVD results
    pseudoinverses = {}  # Store pseudoinverses
    # Compute SVD of the system matrix
    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    svd_results[size] = {'U': U, 'S': S, 'Vt': Vt}

    # Compute the pseudoinverse of the system operator
    S_inv = np.diag(1 / S)
    pseudoinverse = Vt.T @ S_inv @ U.T
    pseudoinverses[size] = pseudoinverse
    return svd_results, pseudoinverse


# Add Poisson noise at different levels
def generate_reconstruction(projection_data, H_MP):
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

    reconstructed_images = {}

    for level, noisy_data in noisy_projections.items():
        reconstructed_images[level] = np.dot(H_MP, noisy_data.T).T

    return reconstructed_images

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





num_realizations = 500
selected_size = pixel_sizes[0] # Example pixel size
# system_matrices = {}  # Store system matrices for each pixel size
system_matrices = generate_system_matrix(M, pixel_sizes)
H = system_matrices[0]
_, H_MP = generate_pseudoinverse(selected_size, H) # Print singular values and vectors or not?

signal_present_objects, signal_absent_objects = generate_multiple_objects(signal_present, signal_absent, grid_size = selected_size, J = 500)
signal_present_realizations = signal_present_objects.reshape(500, -1)
signal_absent_realizations = signal_absent_objects.reshape(500, -1)

projection_data_present = np.dot(H, signal_present_realizations.T).T  # g = Hf, (500, grid x grid)
projection_data_absent = np.dot(H, signal_absent_realizations.T).T
projection_data = [projection_data_present, projection_data_absent]

reconstructed_present = generate_reconstruction(projection_data[0], H_MP)
reconstructed_absent = generate_reconstruction(projection_data[1], H_MP)

reconstructed_present_graph = plot_reconstruction(num_realizations, reconstructed_present, signal_present_objects, selected_size)
reconstructed_absent_graph = plot_reconstruction(num_realizations, reconstructed_absent, signal_absent_objects, selected_size)

# print(reconstructed_present.shape)












# H = system_matrices[selected_size]  # System matrix for the selected pixel size
# H_pseudo = pseudoinverses[selected_size]  # Pseudoinverse for reconstruction

# Simulated object realizations (signal-present objects)
# signal_present_realizations = np.random.rand(num_realizations, selected_size**2)  # Mock for demonstration

# Step iv: Generate projection data

# # Step iv example: Generate projection data of shepp_logan_phantom
# phantom = shepp_logan_phantom()
# phantom_resized = resize(phantom, (32, 32), mode='reflect', anti_aliasing=True)

# my_image = phantom_resized
# f_input = my_image.flatten()
# g_input = np.dot(H, f_input)
# projection_data = g_input


# Step vi: Reconstruct images using the pseudoinverse


# print(reconstructed_images['Low'].shape) # (500, grid x grid)


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