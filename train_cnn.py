import os
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from DL_denoiser import get_cnn
import image_generation
# from image_generation import *


# # Import necessary functions and variables from image generation code
# from image_generation import (
#     generate_system_matrix,
#     generate_pseudoinverse,
#     generate_signal,
#     generate_background,
#     generate_fov_mask,
# )

# def get_cnn(input_size):
#     """
#     Define the CNN model architecture.
#     """
#     from keras.models import Sequential
#     from keras.layers import Conv2D, MaxPooling2D, UpSampling2D

#     model = Sequential()
#     model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(input_size, input_size, 1)))
#     model.add(MaxPooling2D((2, 2), padding='same'))
#     model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
#     model.add(MaxPooling2D((2, 2), padding='same'))
#     model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
#     model.add(UpSampling2D((2, 2)))
#     model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
#     model.add(UpSampling2D((2, 2)))
#     # Use linear activation in the output layer
#     model.add(Conv2D(1, (3, 3), activation='linear', padding='same'))
#     return model

# def generate_multiple_objects(grid_size, signal_center, sigma_s=1/4, A_s=3, N=10, sigma_n=1/4, a_n=1, J=500, fov_mask=None):
#     """
#     Generate J realizations of signal-present and signal-absent objects.
#     """
#     signal_present_objects = np.zeros((J, grid_size, grid_size))
#     signal_absent_objects = np.zeros((J, grid_size, grid_size))

#     for j in range(J):
#         signal = generate_signal(grid_size, signal_center, sigma_s=sigma_s, A_s=A_s)
#         background = generate_background(grid_size, N=N, sigma_n=sigma_n, a_n=a_n)
#         if fov_mask is not None:
#             signal *= fov_mask
#             background *= fov_mask
#         signal_present = signal + background
#         signal_absent = background

#         signal_present_objects[j] = signal_present
#         signal_absent_objects[j] = signal_absent

#     return signal_present_objects, signal_absent_objects

# def generate_reconstruction(projection_data, H_MP, noise_levels):
#     """
#     Generate reconstructions for multiple noise levels.
#     """
#     reconstructed_images = {}
#     for level, total_counts in noise_levels.items():
#         # Scale the projection data
#         scaled_projections = projection_data * (total_counts / np.sum(projection_data, axis=1, keepdims=True))
#         # Add Poisson noise
#         noisy_projections = np.random.poisson(scaled_projections)
#         # Reconstruct images
#         reconstructed_images[level] = np.dot(H_MP, noisy_projections.T).T
#     return reconstructed_images

def prepare_datasets():
    """
    Generate and save datasets for training, validation, and testing.
    """
    # # Parameters
    # M = 32 * 16
    selected_size = 32
    # num_realizations = 500
    # fov_mask = generate_fov_mask(selected_size, radius=0.5)
    # noise_levels = {
    #     "Low": 50000,
    #     "Medium": 25000,
    #     "High": 10000,
    #     "Very_High": 5000
    # }

    # # Generate signal-present and signal-absent objects
    # signal_center = (0, 0)
    # signal_present_objects, signal_absent_objects = generate_multiple_objects(
    #     grid_size=selected_size,
    #     signal_center=signal_center,
    #     sigma_s=1/4,
    #     A_s=5,
    #     N=10,
    #     sigma_n=1/4,
    #     a_n=1,
    #     J=num_realizations,
    #     fov_mask=fov_mask
    # )

    # # Reshape the objects for processing
    # signal_present_realizations = signal_present_objects.reshape(num_realizations, -1)
    # signal_absent_realizations = signal_absent_objects.reshape(num_realizations, -1)

    # # Generate system matrix and pseudoinverse
    # H = generate_system_matrix(M, selected_size, fov_mask=fov_mask)
    # H_MP = generate_pseudoinverse(selected_size, H)

    # # Projection data generation
    # projection_data_present = np.dot(H, signal_present_realizations.T).T
    # projection_data_absent = np.dot(H, signal_absent_realizations.T).T

    # # Combine signal-present and signal-absent projections
    # projection_data = np.concatenate([projection_data_present, projection_data_absent], axis=0)
    # original_images = np.concatenate([signal_present_realizations, signal_absent_realizations], axis=0)

    # # Reconstruct noisy images
    # reconstructed_images = generate_reconstruction(projection_data, H_MP, noise_levels)

    noise_levels = {
        # "Low": 50000,
        "Medium": 25000,
        "High": 10000,
        "Very_High": 2500
    }

    reconstructed_present = image_generation.reconstructed_present #'Low' = (500, 1024)
    reconstructed_absent = image_generation.reconstructed_absent
    # projection_data_present = image_generation.projection_data_present
    # projection_data_absent = image_generation.projection_data_absent
    # reconstructed_images = np.concatenate([reconstructed_present, reconstructed_absent], axis=0)
    # original_images = np.concatenate([projection_data_present, projection_data_absent], axis=0)


    # Prepare datasets for each noise level
    for noise_level in noise_levels.keys():
        print(f"Processing noise level: {noise_level}")
        noisy_images = np.concatenate([reconstructed_present[noise_level], reconstructed_absent[noise_level]],\
                                       axis = 0)
        # noisy_images = reconstructed_images[noise_level]
        # Use original images as labels
        low_noise_images = np.concatenate([reconstructed_present['Low'], reconstructed_absent['Low']],\
                                       axis = 0)
        # low_noise_images = reconstructed_images['Low']

        # Handle NaN and infinite values
        reconstructed_present[noise_level] = np.nan_to_num(reconstructed_present[noise_level], nan=0.0, posinf=0.0, neginf=0.0)
        reconstructed_absent[noise_level] = np.nan_to_num(reconstructed_absent[noise_level], nan=0.0, posinf=0.0, neginf=0.0)
        reconstructed_present['Low'] = np.nan_to_num(reconstructed_present['Low'], nan=0.0, posinf=0.0, neginf=0.0)
        reconstructed_absent['Low'] = np.nan_to_num(reconstructed_absent['Low'], nan=0.0, posinf=0.0, neginf=0.0)
        # # noisy_images = np.nan_to_num(noisy_images, nan=0.0, posinf=0.0, neginf=0.0)
        # # low_noise_images = np.nan_to_num(low_noise_images, nan=0.0, posinf=0.0, neginf=0.0)

        # # Normalize both datasets using the same scaling factor
        # # max_value = max(np.max(np.abs(noisy_images)), np.max(np.abs(low_noise_images)))
        # # if max_value != 0:
        # #     noisy_images = noisy_images / max_value
        # #     low_noise_images = low_noise_images / max_value
        max_value = max(np.max(np.abs(reconstructed_present[noise_level])), np.max(np.abs(reconstructed_absent[noise_level])),\
                        np.max(np.abs(reconstructed_present['Low'])), np.max(np.abs(reconstructed_absent['Low'])))
        if max_value != 0:
            # noisy_images = noisy_images / max_value
            # low_noise_images = low_noise_images / max_value
            reconstructed_present[noise_level] = reconstructed_present[noise_level].astype('float32') / max_value
            reconstructed_absent[noise_level] = reconstructed_absent[noise_level].astype('float32') / max_value
            reconstructed_present['Low'] = reconstructed_present['Low'].astype('float32') / max_value
            reconstructed_absent['Low'] = reconstructed_absent['Low'].astype('float32') / max_value

        # # Convert data types
        # noisy_images = noisy_images.astype('float32')
        # low_noise_images = low_noise_images.astype('float32')

        # Reshape images to (num_samples, selected_size, selected_size, 1)
        num_samples = reconstructed_present[noise_level].shape[0]
        noisy_images_present = reconstructed_present[noise_level].reshape(num_samples, selected_size, selected_size, 1) #(500,32,32,1)
        noisy_images_absent = reconstructed_absent[noise_level].reshape(num_samples, selected_size, selected_size, 1) #(1000,32,32,1)
        low_noise_images_present = reconstructed_present['Low'].reshape(num_samples, selected_size, selected_size, 1)
        low_noise_images_absent = reconstructed_absent['Low'].reshape(num_samples, selected_size, selected_size, 1)
        # noisy_images = noisy_images.reshape(num_samples, selected_size, selected_size, 1) #(1000,32,32,1)
        # low_noise_images = low_noise_images.reshape(num_samples, selected_size, selected_size, 1)

        # Shuffle and split data
        indices = np.random.permutation(num_samples)
        train_idx = indices[:300]
        val_idx = indices[300:400]
        test_idx = indices[400:500]
        # train_idx = indices[:600]
        # val_idx = indices[600:800]
        # test_idx = indices[800:1000]

        # Create directories and save files
        directory = f'./{noise_level}'
        os.makedirs(directory, exist_ok=True)
        noisy_images1 = np.concatenate([noisy_images_present[train_idx], noisy_images_absent[train_idx]], axis=0)
        noisy_images2 = np.concatenate([noisy_images_present[val_idx], noisy_images_absent[val_idx]], axis=0)
        noisy_images3 = np.concatenate([noisy_images_present[test_idx], noisy_images_absent[test_idx]], axis=0)
        np.save(f'{directory}/train_data.npy', noisy_images1)
        np.save(f'{directory}/val_data.npy', noisy_images2)
        np.save(f'{directory}/test_data.npy', noisy_images3)
        # np.save(f'{directory}/train_data.npy', noisy_images[train_idx])
        # np.save(f'{directory}/val_data.npy', noisy_images[val_idx])
        # np.save(f'{directory}/test_data.npy', noisy_images[test_idx])
        np.save(f'{directory}/test_data_present.npy', noisy_images_present[test_idx])
        np.save(f'{directory}/test_data_absent.npy', noisy_images_absent[test_idx])

        # Save ground truth labels
        low_noise_images1 = np.concatenate([low_noise_images_present[train_idx], low_noise_images_absent[train_idx]], axis=0)
        low_noise_images2 = np.concatenate([low_noise_images_present[val_idx], low_noise_images_absent[val_idx]], axis=0)
        low_noise_images3 = np.concatenate([low_noise_images_present[test_idx], low_noise_images_absent[test_idx]], axis=0)
        np.save(f'{directory}/train_labels.npy', low_noise_images1)
        np.save(f'{directory}/val_labels.npy', low_noise_images2)
        np.save(f'{directory}/test_labels.npy', low_noise_images3)
        # np.save(f'{directory}/train_labels.npy', low_noise_images[train_idx])
        # np.save(f'{directory}/val_labels.npy', low_noise_images[val_idx])
        # np.save(f'{directory}/test_labels.npy', low_noise_images[test_idx])
        np.save(f'{directory}/test_labels_present.npy', low_noise_images_present[test_idx])
        np.save(f'{directory}/test_labels_absent.npy', low_noise_images_absent[test_idx])
        print(f"Saved datasets for noise level {noise_level} in {directory}")

def load_data(noise_level):
    """
    Load training, validation, and test datasets for a given noise level.
    """
    directory = f'./{noise_level}'
    train_data = np.load(f'{directory}/train_data.npy')
    val_data = np.load(f'{directory}/val_data.npy')
    test_data = np.load(f'{directory}/test_data.npy')
    train_labels = np.load(f'{directory}/train_labels.npy')
    val_labels = np.load(f'{directory}/val_labels.npy')
    test_labels = np.load(f'{directory}/test_labels.npy')
    return train_data, train_labels, val_data, val_labels, test_data, test_labels

def train_cnn(noise_level, epochs=100, batch_size=16):
    """
    Train the CNN for a specific noise level.
    """
    train_data, train_labels, val_data, val_labels, _, _ = load_data(noise_level)
    # X = np.zeros((train_data.shape[0], train_data.shape[1], train_data.shape[2], 1))  #
    # Y = np.zeros((train_labels.shape[0], train_labels.shape[1], train_labels.shape[2], 1))  #

    # for i in np.arange(train_data.shape[0]):
    #     X[i, :, :, 0] = train_data[i, :, :, 0] # Change this path if needed
    #     Y[i, :, :, 0] = train_labels[i, :, :, 0]  # Change this path if needed

    # model = get_cnn(train_data.shape[1])  # Use image size from data
    print(f'train_data.shape is: {train_data.shape}')
    model = get_cnn(train_data.shape[1])

    # Use an optimizer with a lower learning rate
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    # model.summary()

    # early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(
        train_data, train_labels,
        validation_data=(val_data, val_labels),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[early_stop]
    )
    directory = f'./{noise_level}'
    model.save(f'{directory}/cnn_denoising_model.h5')
    # Save training history
    np.save(f'{directory}/train_loss_epochs{epochs}.npy', history.history['loss'])
    np.save(f'{directory}/val_loss_epochs{epochs}.npy', history.history['val_loss'])

def evaluate_model(noise_level, batch_size=16):
    """
    Evaluate the trained CNN model.
    """
    _, _, _, _, test_data, test_labels = load_data(noise_level)
    directory = f'./{noise_level}'
    model = load_model(f'{directory}/cnn_denoising_model.h5')
    print(f'test_data.shape is: {test_data.shape}')
    denoised_images = model.predict(test_data, batch_size=batch_size)
    denoised_images_present = model.predict(test_data[0:100], batch_size=batch_size)
    denoised_images_absent = model.predict(test_data[100:200], batch_size=batch_size)

    directory = f'./{noise_level}'
    np.save(f'{directory}/predicted_images_present.npy', denoised_images_present)
    np.save(f'{directory}/predicted_images_absent.npy', denoised_images_absent)

    # low_noise_images = np.asarray(low_noise_images)
    # denoised_images = np.asarray(denoised_images)

    # # Check if shapes match
    # if low_noise_image.shape != denoised_image.shape:
    #     raise ValueError("Images must have the same dimensions.")

    # Compute RMSE
    rmse = []
    for i in range(test_data.shape[0]):
        rmse.append(np.sqrt(np.mean((test_labels[i].flatten() - denoised_images[i].flatten()) ** 2)))

    mse = np.mean((test_labels.flatten() - denoised_images.flatten()) ** 2)
    rmse_mean = np.sqrt(mse)

    # rmse = np.sqrt(mean_squared_error(test_labels.flatten(), denoised_images.flatten()))
    print(f"RMSE for {noise_level} noise level: {rmse_mean}")
    return denoised_images, rmse_mean, test_data, test_labels, rmse

def visualize_denoising(noisy_image, denoised_image, low_noise_image):
    """Display noisy, denoised, and low-noise images side by side."""
    # Normalize images for display
    def normalize_image(img):
        img_min = img.min()
        img_max = img.max()
        if img_max - img_min == 0:
            return np.zeros_like(img)
        else:
            return (img - img_min) / (img_max - img_min)

    noisy_image_disp = normalize_image(noisy_image)
    denoised_image_disp = normalize_image(denoised_image)
    low_noise_image_disp = normalize_image(low_noise_image)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot noisy image
    axes[0].imshow(noisy_image_disp, cmap='viridis', origin='lower')
    axes[0].set_title("Noisy Image")
    axes[0].axis('off')

    # Plot denoised image
    axes[1].imshow(denoised_image_disp, cmap='viridis', origin='lower')
    axes[1].set_title("Denoised Image")
    axes[1].axis('off')

    # Plot low-noise image
    axes[2].imshow(low_noise_image_disp, cmap='viridis', origin='lower')
    axes[2].set_title("Original Image (Low Noise)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

def plot_loss_curves(noise_levels, epochs=100):
    """
    Plot training and validation loss curves for each noise level.
    """
    for noise_level in noise_levels:
        directory = f'./{noise_level}'
        # Load loss histories
        train_loss = np.load(f'{directory}/train_loss_epochs{epochs}.npy')
        val_loss = np.load(f'{directory}/val_loss_epochs{epochs}.npy')

        epochs_list = list(range(1, len(train_loss) + 1))
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_list, train_loss, label='Training Loss')
        plt.plot(epochs_list, val_loss, label='Validation Loss')
        plt.title(f"Loss Curves for {noise_level} Noise Level")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    prepare_datasets()
    metrics = {}
    rmse_boxplot = {}

    noise_levels = ['Medium', 'High', 'Very_High']
    for noise in noise_levels:
        print(f"Training CNN for noise level: {noise}")
        train_cnn(noise)

    # for noise in noise_levels:
        print(f"Evaluating CNN for noise level: {noise}")
        denoised_images, rmse_mean, test_data, test_labels, rmse = evaluate_model(noise)
        metrics[noise] = rmse_mean
        rmse_boxplot[noise] = rmse
        # Visualize the denoising result for a random sample
        idx = np.random.randint(0, test_data.shape[0])
        visualize_denoising(
            test_data[idx].reshape(test_data.shape[1], test_data.shape[2]),
            denoised_images[idx].reshape(denoised_images.shape[1], denoised_images.shape[2]),
            test_labels[idx].reshape(test_labels.shape[1], test_labels.shape[2])
        )
    print(metrics)
    print(test_data[idx].reshape(test_data.shape[1], test_data.shape[2]))
    print(denoised_images[idx].reshape(denoised_images.shape[1], denoised_images.shape[2]))
    print(test_labels[idx].reshape(test_labels.shape[1], test_labels.shape[2]))
    # Plot training and validation loss curves
    plot_loss_curves(noise_levels)

    # Plot RMSE metrics
    plt.figure(figsize=(8, 6))
    noise_labels = list(metrics.keys())
    rmse_values = [metrics[noise] for noise in noise_labels]
    plt.bar(noise_labels, rmse_values, color=['blue', 'orange', 'green'])
    plt.title('RMSE Metrics for Different Noise Levels')
    plt.xlabel('Noise Level')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.show()

    print("RMSE Metrics:", metrics)

    data = []
    for noise in noise_levels:
        data.append(rmse_boxplot[noise])
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_axes([0, 0, 1, 1])
    fig.axis('on')
    bp = ax.boxplot(data)
    plt.grid(True)
    plt.legend()
    plt.show()
