import os
import numpy as np
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras import optimizers
from DL_denoiser import get_cnn
from image_generation import *
import tensorflow as tf
# from image_generation import generate_reconstruction, signal_present_objects, signal_absent_objects, H, H_MP
from sklearn.metrics import mean_squared_error


def prepare_datasets():
    """
    Generate and save datasets for training, validation, and testing.
    """
    num_realizations = signal_present_objects.shape[0]
    # train_size = 300
    # val_size = 100
    # test_size = 100

    # Generate projection data
    projection_data_present = np.dot(H, signal_present_objects.reshape(num_realizations, -1).T).T
    projection_data_absent = np.dot(H, signal_absent_objects.reshape(num_realizations, -1).T).T

    projection_data_present = np.dot(H, signal_present_realizations.T).T
    projection_data_absent = np.dot(H, signal_absent_realizations.T).T

    # Reconstruct noisy images
    reconstructed_present = generate_reconstruction(projection_data_present, H_MP)
    reconstructed_absent = generate_reconstruction(projection_data_absent, H_MP)
    # noisy_projection_present = generate_noise(projection_data_present)
    # noisy_projection_absent = generate_noise(projection_data_absent)
    # reconstructed_present = noisy_projection_present
    # reconstructed_absent = noisy_projection_absent

    for noise_level in reconstructed_present.keys():
        print(f"Processing noise level: {noise_level}")  # Debugging print

        noisy_images = np.concatenate([reconstructed_present[noise_level], reconstructed_absent[noise_level]], axis=0)
        low_noise_images = np.concatenate([reconstructed_present["Low"], reconstructed_absent["Low"]], axis=0)

        print(f"Noisy images shape before reshaping: {noisy_images.shape}") # (1000, 1024)
        print(f"Low noise images shape before reshaping: {low_noise_images.shape}") # (1000, 1024)

        # Check for NaN and infinite values
        if np.isnan(noisy_images).any() or np.isnan(low_noise_images).any():
            print(f"NaN detected in dataset generation for {noise_level}!")
            noisy_images = np.nan_to_num(noisy_images, nan=0.0)
            low_noise_images = np.nan_to_num(low_noise_images, nan=0.0)
        if not np.isfinite(noisy_images).all() or not np.isfinite(low_noise_images).all():
            print(f"Infinite values detected in dataset generation for {noise_level}!")
            noisy_images = np.nan_to_num(noisy_images, posinf=0.0, neginf=0.0)
            low_noise_images = np.nan_to_num(low_noise_images, posinf=0.0, neginf=0.0)

        # Normalize the data
        max_noisy = np.max(np.abs(noisy_images))
        max_low_noise = np.max(np.abs(low_noise_images))

        if max_noisy == 0:
            max_noisy = 1
        if max_low_noise == 0:
            max_low_noise = 1

        noisy_images = noisy_images / max_noisy
        low_noise_images = low_noise_images / max_low_noise

        # Convert data types to float32
        noisy_images = noisy_images.astype(np.float32)
        low_noise_images = low_noise_images.astype(np.float32)

        # Shuffle and split data
        indices = np.random.permutation(noisy_images.shape[0])
        train_idx = indices[:300]
        val_idx = indices[300:400]
        test_idx = indices[400:500]
        print(f'training indices are: {train_idx}')
        print(f'length of training indices: {len(train_idx)}')

        # Reshape and save datasets
        directory = f'./{noise_level}'
        os.makedirs(directory, exist_ok=True)
        np.save(f'{directory}/train_data.npy', noisy_images[train_idx].reshape(-1, 32, 32, 1)) # (1000, 1024) --> (N, 32, 32, 1)
        np.save(f'{directory}/val_data.npy', noisy_images[val_idx].reshape(-1, 32, 32, 1))
        np.save(f'{directory}/test_data.npy', noisy_images[test_idx].reshape(-1, 32, 32, 1))
        # np.save(f'train_data_{noise_level}.npy', noisy_images[train_idx].reshape(-1, 32, 32, 1))
        # np.save(f'val_data_{noise_level}.npy', noisy_images[val_idx].reshape(-1, 32, 32, 1))
        # np.save(f'test_data_{noise_level}.npy', noisy_images[test_idx].reshape(-1, 32, 32, 1))

        np.save('train_labels.npy', low_noise_images[train_idx].reshape(-1, 32, 32, 1)) # ground truth
        np.save('val_labels.npy', low_noise_images[val_idx].reshape(-1, 32, 32, 1))
        np.save('test_labels.npy', low_noise_images[test_idx].reshape(-1, 32, 32, 1))

# def loss_fn(y_true, y_pred):
#     # mean square error
#     data_fidelity = tf.reshape(y_true, shape=[-1]) - tf.reshape(y_pred, shape=[-1])
#     data_fidelity = tf.reduce_mean(tf.square(data_fidelity))
#     return data_fidelity

# Load datasets
def load_data(noise_level):
    """
    Load training, validation, and test datasets for a given noise level.
    """
    # train_data = np.load(f'train_data_{noise_level}.npy')
    # val_data = np.load(f'val_data_{noise_level}.npy')
    # test_data = np.load(f'test_data_{noise_level}.npy')
    train_data = np.load(f'./{noise_level}/train_data.npy')
    val_data = np.load(f'./{noise_level}/val_data.npy')
    test_data = np.load(f'./{noise_level}/test_data.npy')

    train_labels = np.load('train_labels.npy')
    val_labels = np.load('val_labels.npy')
    test_labels = np.load('test_labels.npy')

    print(f'training data shape: {train_data.shape}')
    print(f'validation data shape: {val_data.shape}')
    print(f'test data shape: {test_data.shape}')

    return train_data, train_labels, val_data, val_labels, test_data, test_labels


# Train CNN
def train_cnn(noise_level, epochs=100, batch_size=16, img_dim=32): # epochs=50
    """
    Train the CNN for a specific noise level.
    """
    train_data, train_labels, val_data, val_labels, _, _ = load_data(noise_level)

    # Define model
    model = get_cnn(img_dim)
    # model.summary()
    # optimizer = optimizers.Adam(learning_rate=1e-3, )
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train model
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        train_data, train_labels,
        validation_data=(val_data, val_labels),
        epochs=epochs,
        batch_size=batch_size,
        shuffle = True,
        callbacks=[early_stop]
    )
    print('Training finished.')
    train_loss = history.history['loss'] # training loss
    val_loss = history.history['val_loss'] # validation loss
    np.save(f'./{noise_level}/train_loss_epochs{epochs}.npy',train_loss) # training loss saved in npy file. Use provided readNPY() MATLAB function to read.
    np.save(f'./{noise_level}/val_loss_epochs{epochs}.npy',val_loss)# validation loss saved in npy file. Use provided readNPY() MATLAB function to read.

    # Save model
    # model.save(f'cnn_denoising_{noise_level}.h5')
    model.save(f'./{noise_level}/cnn_denoising_model.h5')

    return history


# Evaluate model
def evaluate_model(noise_level, img_dim=32, batch_size=16, num_test=500):
    """
    Evaluate the trained CNN model.
    """
    _, _, _, _, test_data, test_labels = load_data(noise_level)

    # Load trained model
    # model = load_model(f'cnn_denoising_{noise_level}.h5')
    model = load_model(f'./{noise_level}/cnn_denoising_model.h5')

    # Predict denoised images
    denoised_images = model.predict(test_data,
                                    batch_size=batch_size)
    print(f'test_images has {test_data.shape}')
    print(f'denoised_images has {denoised_images.shape}')

    print('Saving the predicted data...')
    for i in np.arange(100):
        current_pred = denoised_images[i]
        directory = f'./{noise_level}/prediction'
        os.makedirs(directory, exist_ok=True)
        np.save(f'{directory}/image{i}.npy',current_pred)
        # To read them, use provided readNPY() MATLAB function;

    # Check for NaN values
    if np.isnan(test_labels).any():
        print("NaN found in test_labels!")
        test_labels = np.nan_to_num(test_labels, nan=0.0)

    if np.isnan(denoised_images).any():
        print("NaN found in denoised_images!")
        denoised_images = np.nan_to_num(denoised_images, nan=0.0)

    # Compute RMSE
    rmse = np.sqrt(mean_squared_error(test_labels.flatten(), denoised_images.flatten()))
    print(f"RMSE for {noise_level} noise level: {rmse}")

    return denoised_images, rmse


# Main Execution
if __name__ == "__main__":
    print("Preparing datasets...")
    prepare_datasets()
    metrics = {'rmse': {}}

    noise_levels = ['Medium', 'High', 'Very_High']
    for noise in noise_levels:
        print(f"Training CNN for {noise} noise level...")
        train_cnn(noise)

    for noise in noise_levels:
        print(f"Evaluating CNN for {noise} noise level...")
        denoised_images, metric = evaluate_model(noise)
        metrics[noise] = metric

    print(metrics)