import numpy as np
import matplotlib.pyplot as plt


def plot_train_loss(noise_levels, epochs=100):
  train_loss = []
  for noise_level in noise_levels:
    with open(f'./{noise_level}/train_loss_epochs{epochs}.npy', 'rb') as f:
      loss1 = np.load(f)
    train_loss.append(loss1)

  plt.figure(figsize=(10, 6))
  for i, noise_level in enumerate(noise_levels):
    epochs_list = list(range(1, train_loss[i].shape[0]+1))
    plt.plot(epochs_list, train_loss[i], label=f'{noise_level}', marker='o')
  # plt.plot(epochs_list, train_loss[0], label=f'{noise_levels[0]}', marker='o')
  # plt.plot(epochs_list, train_loss[1], label=f'{noise_levels[1]}', marker='--')
  # plt.plot(epochs_list, train_loss[2], label=f'{noise_levels[2]}', marker='.')
  plt.title("Training Loss Function")
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.grid(True)
  plt.legend()
  plt.show()
  # plt.savefig("../cache/stats.jpg")


def plot_val_loss(noise_levels, epochs=100):
  val_loss = []
  for noise_level in noise_levels:
    with open(f'./{noise_level}/val_loss_epochs{epochs}.npy', "rb") as f:
      loss2 = np.load(f)
    val_loss.append(loss2)

  plt.figure(figsize=(10, 6))
  for i, noise_level in enumerate(noise_levels):
    epochs_list = list(range(1, val_loss[i].shape[0]+1))
    plt.plot(epochs_list, val_loss[i], label=f'{noise_level}', marker='o')
  # plt.plot(epochs_list, train_loss[0], label=f'{noise_levels[0]}', marker='o')
  # plt.plot(epochs_list, train_loss[1], label=f'{noise_levels[1]}', marker='--')
  # plt.plot(epochs_list, train_loss[2], label=f'{noise_levels[2]}', marker='.')
  plt.title("Validation Loss Function")
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.grid(True)
  plt.legend()
  plt.show()


noise_levels = ['Medium', 'High', 'Very_High']
plot_train_loss(noise_levels=noise_levels)
# plot_val_loss(noise_levels=noise_levels)









# import numpy as np
# from keras.models import load_model
# from keras.callbacks import EarlyStopping
# from DL_denoiser import get_cnn
# from image_generation import generate_reconstruction, signal_present_objects, signal_absent_objects, H, H_MP
# from sklearn.metrics import mean_squared_error

# def prepare_datasets():
#     """
#     Generate and save datasets for training, validation, and testing.
#     """
#     num_realizations = signal_present_objects.shape[0]
#     train_size = 300
#     val_size = 100
#     test_size = 100

#     # Generate projection data
#     projection_data_present = np.dot(H, signal_present_objects.reshape(num_realizations, -1).T).T
#     projection_data_absent = np.dot(H, signal_absent_objects.reshape(num_realizations, -1).T).T

#     # Reconstruct noisy images
#     reconstructed_present = generate_reconstruction(projection_data_present, H_MP)
#     reconstructed_absent = generate_reconstruction(projection_data_absent, H_MP)

#     for noise_level in reconstructed_present.keys():
#         print(f"Processing noise level: {noise_level}")  # Debugging print

#         noisy_images = np.concatenate([reconstructed_present[noise_level], reconstructed_absent[noise_level]], axis=0)
#         low_noise_images = np.concatenate([reconstructed_present["Low"], reconstructed_absent["Low"]], axis=0)

#         print(f"Noisy images shape before reshaping: {noisy_images.shape}")
#         print(f"Low noise images shape before reshaping: {low_noise_images.shape}")

#         # Check for NaN and infinite values
#         if np.isnan(noisy_images).any() or np.isnan(low_noise_images).any():
#             print(f"NaN detected in dataset generation for {noise_level}!")
#             noisy_images = np.nan_to_num(noisy_images, nan=0.0)
#             low_noise_images = np.nan_to_num(low_noise_images, nan=0.0)
#         if not np.isfinite(noisy_images).all() or not np.isfinite(low_noise_images).all():
#             print(f"Infinite values detected in dataset generation for {noise_level}!")
#             noisy_images = np.nan_to_num(noisy_images, posinf=0.0, neginf=0.0)
#             low_noise_images = np.nan_to_num(low_noise_images, posinf=0.0, neginf=0.0)

#         # Normalize the data
#         max_noisy = np.max(np.abs(noisy_images))
#         max_low_noise = np.max(np.abs(low_noise_images))

#         if max_noisy == 0:
#             max_noisy = 1
#         if max_low_noise == 0:
#             max_low_noise = 1

#         noisy_images = noisy_images / max_noisy
#         low_noise_images = low_noise_images / max_low_noise

#         # Convert data types to float32
#         noisy_images = noisy_images.astype(np.float32)
#         low_noise_images = low_noise_images.astype(np.float32)

#         # Shuffle and split data
#         indices = np.random.permutation(noisy_images.shape[0])
#         train_idx = indices[:train_size]
#         val_idx = indices[train_size:train_size + val_size]
#         test_idx = indices[train_size + val_size:]

#         # Reshape and save datasets
#         np.save(f'train_data_{noise_level}.npy', noisy_images[train_idx].reshape(-1, 32, 32, 1))
#         np.save(f'val_data_{noise_level}.npy', noisy_images[val_idx].reshape(-1, 32, 32, 1))
#         np.save(f'test_data_{noise_level}.npy', noisy_images[test_idx].reshape(-1, 32, 32, 1))

#         np.save('train_labels.npy', low_noise_images[train_idx].reshape(-1, 32, 32, 1))
#         np.save('val_labels.npy', low_noise_images[val_idx].reshape(-1, 32, 32, 1))
#         np.save('test_labels.npy', low_noise_images[test_idx].reshape(-1, 32, 32, 1))


# # Load datasets
# def load_data(noise_level):
#     """
#     Load training, validation, and test datasets for a given noise level.
#     """
#     train_data = np.load(f'train_data_{noise_level}.npy')
#     val_data = np.load(f'val_data_{noise_level}.npy')
#     test_data = np.load(f'test_data_{noise_level}.npy')

#     train_labels = np.load('train_labels.npy')
#     val_labels = np.load('val_labels.npy')
#     test_labels = np.load('test_labels.npy')

#     return train_data, train_labels, val_data, val_labels, test_data, test_labels


# # Train CNN
# def train_cnn(noise_level, epochs=50, batch_size=16, img_dim=32):
#     """
#     Train the CNN for a specific noise level.
#     """
#     train_data, train_labels, val_data, val_labels, _, _ = load_data(noise_level)

#     # Define model
#     model = get_cnn(img_dim)
#     model.compile(optimizer='adam', loss='mean_squared_error')

#     # Train model
#     early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#     history = model.fit(
#         train_data, train_labels,
#         validation_data=(val_data, val_labels),
#         epochs=epochs,
#         batch_size=batch_size,
#         callbacks=[early_stop]
#     )

#     # Save model
#     model.save(f'cnn_denoising_{noise_level}.h5')

#     return history


# # Evaluate model
# def evaluate_model(noise_level, img_dim=32):
#     """
#     Evaluate the trained CNN model.
#     """
#     _, _, _, _, test_data, test_labels = load_data(noise_level)

#     # Load trained model
#     model = load_model(f'cnn_denoising_{noise_level}.h5')

#     # Predict denoised images
#     denoised_images = model.predict(test_data)

#     # Check for NaN values
#     if np.isnan(test_labels).any():
#         print("NaN found in test_labels!")
#         test_labels = np.nan_to_num(test_labels, nan=0.0)

#     if np.isnan(denoised_images).any():
#         print("NaN found in denoised_images!")
#         denoised_images = np.nan_to_num(denoised_images, nan=0.0)

#     # Compute RMSE
#     rmse = np.sqrt(mean_squared_error(test_labels.flatten(), denoised_images.flatten()))
#     print(f"RMSE for {noise_level} noise level: {rmse}")

#     return denoised_images


# # Main Execution
# if __name__ == "__main__":
#     print("Preparing datasets...")
#     prepare_datasets()

#     noise_levels = ['Medium', 'High', 'Very_High']
#     for noise in noise_levels:
#         print(f"Training CNN for {noise} noise level...")
#         train_cnn(noise)

#     for noise in noise_levels:
#         print(f"Evaluating CNN for {noise} noise level...")
#         evaluate_model(noise)
