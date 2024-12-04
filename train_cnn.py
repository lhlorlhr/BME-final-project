import os
import numpy as np
from keras.models import load_model
from keras.callbacks import EarlyStopping
from DL_denoiser import get_cnn
from image_generation import *
from sklearn.metrics import mean_squared_error
import tensorflow as tf

# Debugging function
def debug_noisy_images(reconstructed_present, reconstructed_absent):
    for noise_level in reconstructed_present.keys():
        diff = np.mean(reconstructed_present[noise_level] - reconstructed_absent[noise_level])
        print(f"Noise Level: {noise_level}, Present-Absent Mean Difference: {diff}")

def prepare_datasets():
    """
    Generate and save datasets for training, validation, and testing.
    """
    # Projection data generation
    projection_data_present = np.dot(H, signal_present_realizations.T).T
    projection_data_absent = np.dot(H, signal_absent_realizations.T).T

    # Reconstruct noisy images
    reconstructed_present = generate_reconstruction(projection_data_present, H_MP)
    reconstructed_absent = generate_reconstruction(projection_data_absent, H_MP)

    for noise_level in reconstructed_present.keys():
        print(f"Processing noise level: {noise_level}")
        noisy_images = np.concatenate([reconstructed_present[noise_level], reconstructed_absent[noise_level]], axis=0)
        low_noise_images = np.concatenate([reconstructed_present["Low"], reconstructed_absent["Low"]], axis=0)

        # Debugging: Check array shapes
        print(f"Noise Level: {noise_level}, Noisy Images Shape: {noisy_images.shape}")
        print(f"Low Noise Images Shape (Labels): {low_noise_images.shape}")

        # Normalize and convert data types
        max_noisy = np.max(np.abs(noisy_images))
        noisy_images = noisy_images / max_noisy if max_noisy != 0 else 1
        max_low_noise = np.max(np.abs(low_noise_images))
        low_noise_images = low_noise_images / max_low_noise if max_low_noise != 0 else 1

        # Shuffle and split data
        indices = np.random.permutation(noisy_images.shape[0])
        train_idx = indices[:300]
        val_idx = indices[300:400]
        test_idx = indices[400:500]

        # Create directories and save files
        directory = f'./{noise_level}'
        os.makedirs(directory, exist_ok=True)
        np.save(f'{directory}/train_data.npy', noisy_images[train_idx].reshape(-1, 32, 32, 1))
        np.save(f'{directory}/val_data.npy', noisy_images[val_idx].reshape(-1, 32, 32, 1))
        np.save(f'{directory}/test_data.npy', noisy_images[test_idx].reshape(-1, 32, 32, 1))

        # Save ground truth labels
        np.save(f'{directory}/train_labels.npy', low_noise_images[train_idx].reshape(-1, 32, 32, 1))
        np.save(f'{directory}/val_labels.npy', low_noise_images[val_idx].reshape(-1, 32, 32, 1))
        np.save(f'{directory}/test_labels.npy', low_noise_images[test_idx].reshape(-1, 32, 32, 1))
        print(f"Saved train_labels.npy for noise level {noise_level} in {directory}")

def load_data(noise_level):
    """
    Load training, validation, and test datasets for a given noise level.
    """
    train_data = np.load(f'./{noise_level}/train_data.npy')
    val_data = np.load(f'./{noise_level}/val_data.npy')
    test_data = np.load(f'./{noise_level}/test_data.npy')
    train_labels = np.load(f'./{noise_level}/train_labels.npy')
    val_labels = np.load(f'./{noise_level}/val_labels.npy')
    test_labels = np.load(f'./{noise_level}/test_labels.npy')
    return train_data, train_labels, val_data, val_labels, test_data, test_labels

def train_cnn(noise_level, epochs=100, batch_size=16):
    """
    Train the CNN for a specific noise level.
    """
    train_data, train_labels, val_data, val_labels, _, _ = load_data(noise_level)

    model = get_cnn(32)
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        train_data, train_labels,
        validation_data=(val_data, val_labels),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[early_stop]
    )
    model.save(f'./{noise_level}/cnn_denoising_model.h5')

def evaluate_model(noise_level, batch_size=16):
    """
    Evaluate the trained CNN model.
    """
    _, _, _, _, test_data, test_labels = load_data(noise_level)
    model = load_model(f'./{noise_level}/cnn_denoising_model.h5')
    denoised_images = model.predict(test_data, batch_size=batch_size)

    rmse = np.sqrt(mean_squared_error(test_labels.flatten(), denoised_images.flatten()))
    print(f"RMSE for {noise_level} noise level: {rmse}")
    return denoised_images, rmse

if __name__ == "__main__":
    prepare_datasets()
    metrics = {}

    noise_levels = ['Medium', 'High', 'Very_High']
    for noise in noise_levels:
        train_cnn(noise)

    for noise in noise_levels:
        _, rmse = evaluate_model(noise)
        metrics[noise] = rmse

    print(metrics)
