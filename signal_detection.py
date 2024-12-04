import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def compute_auc(signal_present, signal_absent):
    """
    Compute the AUC for signal detection.
    Args:
        signal_present (array): Signal-present images.
        signal_absent (array): Signal-absent images.
    Returns:
        float: AUC score.
    """
    # Flatten and prepare data
    signal_present_flat = signal_present.reshape(len(signal_present), -1)
    signal_absent_flat = signal_absent.reshape(len(signal_absent), -1)

    # Create labels and predictions
    labels = np.concatenate([np.ones(len(signal_present_flat)), np.zeros(len(signal_absent_flat))])
    predictions = np.concatenate([signal_present_flat.mean(axis=1), signal_absent_flat.mean(axis=1)])  # Mean intensity

    return roc_auc_score(labels, predictions)

def evaluate_signal_detection(signal_present_objects, signal_absent_objects, noise_levels):
    """
    Evaluate signal detection for all noise levels and plot AUC.
    Args:
        signal_present_objects (dict): Signal-present images for each noise level.
        signal_absent_objects (dict): Signal-absent images for each noise level.
        noise_levels (list): List of noise levels.
    """
    auc_results = []

    for level in noise_levels:
        # Extract signal-present and signal-absent data for the current noise level
        signal_present = signal_present_objects[level][:200]
        signal_absent = signal_absent_objects[level][:200]

        # Compute AUC
        auc = compute_auc(signal_present, signal_absent)
        auc_results.append(auc)

        print(f"AUC for {level} noise: {auc}")

    # Plot AUC results
    plt.figure()
    plt.plot(noise_levels, auc_results, marker="o")
    plt.xlabel("Noise Level")
    plt.ylabel("AUC")
    plt.title("Signal Detection Task Performance")
    plt.show()

if __name__ == "__main__":
    # Import necessary functions from image_generation
    from image_generation import (
        generate_system_matrix,
        generate_pseudoinverse,
        generate_signal,
        generate_background,
        generate_fov_mask,
        generate_multiple_objects,
        downsample,
    )

    # Generate the data needed for signal detection
    grid_size = 64
    selected_size = 32
    num_realizations = 500
    M = 32 * 16
    fov_mask = generate_fov_mask(grid_size, radius=0.5)
    noise_levels = {
        "Low": 50000,
        "Medium": 25000,
        "High": 10000,
        "Very_High": 5000
    }

    # Generate signal, background, and signal-present images
    signal_center = (0, 0)
    signal = generate_signal(grid_size, signal_center, sigma_s=1/4, A_s=5)
    background = generate_background(grid_size)
    signal_present = signal + background

    # Apply FOV mask to all images
    signal_present = signal_present * fov_mask
    signal_absent = background * fov_mask

    # Downsample images to selected size
    factor = grid_size // selected_size  # Assuming grid_size is divisible by selected_size
    downsampled_present, downsampled_absent = downsample(signal_present, signal_absent, factor=factor)

    # Generate multiple realizations
    signal_present_objects, signal_absent_objects = generate_multiple_objects(
        downsampled_present, downsampled_absent, grid_size=selected_size, J=num_realizations
    )

    # Reshape images for processing
    signal_present_realizations = signal_present_objects.reshape(num_realizations, -1)
    signal_absent_realizations = signal_absent_objects.reshape(num_realizations, -1)

    # Generate system matrix and pseudoinverse
    H = generate_system_matrix(M, selected_size, fov_mask=None)  # Assuming FOV mask is already applied
    H_MP = generate_pseudoinverse(selected_size, H)

    # Generate projection data
    projection_data_present = np.dot(H, signal_present_realizations.T).T
    projection_data_absent = np.dot(H, signal_absent_realizations.T).T

    # Generate reconstructions for each noise level
    reconstructed_present = {}
    reconstructed_absent = {}
    for level, counts in noise_levels.items():
        # Scale projection data
        scaled_projections_present = projection_data_present * (counts / np.sum(projection_data_present, axis=1, keepdims=True))
        scaled_projections_absent = projection_data_absent * (counts / np.sum(projection_data_absent, axis=1, keepdims=True))

        # Add Poisson noise
        noisy_projections_present = np.random.poisson(scaled_projections_present)
        noisy_projections_absent = np.random.poisson(scaled_projections_absent)

        # Reconstruct images
        recon_images_present = np.dot(H_MP, noisy_projections_present.T).T
        recon_images_absent = np.dot(H_MP, noisy_projections_absent.T).T

        # Reshape images back to 2D
        recon_images_present = recon_images_present.reshape(num_realizations, selected_size, selected_size)
        recon_images_absent = recon_images_absent.reshape(num_realizations, selected_size, selected_size)

        # Store the reconstructed images
        reconstructed_present[level] = recon_images_present
        reconstructed_absent[level] = recon_images_absent

    # Define noise levels in order
    noise_levels_list = ["Medium", "High", "Very_High"]

    # Evaluate signal detection
    evaluate_signal_detection(reconstructed_present, reconstructed_absent, noise_levels_list)
