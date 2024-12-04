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
        epochs_list = list(range(1, train_loss[i].shape[0] + 1))
        plt.plot(epochs_list, train_loss[i], label=f'{noise_level}', marker='o')
    plt.title("Training Loss Function")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_val_loss(noise_levels, epochs=100):
    val_loss = []
    for noise_level in noise_levels:
        with open(f'./{noise_level}/val_loss_epochs{epochs}.npy', "rb") as f:
            loss2 = np.load(f)
        val_loss.append(loss2)

    plt.figure(figsize=(10, 6))
    for i, noise_level in enumerate(noise_levels):
        epochs_list = list(range(1, val_loss[i].shape[0] + 1))
        plt.plot(epochs_list, val_loss[i], label=f'{noise_level}', marker='o')
    plt.title("Validation Loss Function")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()

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
    axes[2].set_title("Original Image (Ground Truth)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()
