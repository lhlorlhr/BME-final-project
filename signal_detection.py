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
    plt.plot(noise_levels, auc_results, marker="o")
    plt.xlabel("Noise Level")
    plt.ylabel("AUC")
    plt.title("Signal Detection Task Performance")
    plt.show()

# Load precomputed reconstructed images
reconstructed_present = np.load("reconstructed_present.npy", allow_pickle=True).item()
reconstructed_absent = np.load("reconstructed_absent.npy", allow_pickle=True).item()

# Define noise levels
noise_levels = ["Low", "Medium", "High", "Very_High"]

# Evaluate signal detection
evaluate_signal_detection(reconstructed_present, reconstructed_absent, noise_levels)


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.special import j1  # Bessel function of the first kind

# def performance_evaluation(IS, IN):
#     """
#     Perform signal-detection task using channelized Hotelling observer.

#     Args:
#         IS (numpy.ndarray): Signal-present images (pixels x images).
#         IN (numpy.ndarray): Signal-absent images (pixels x images).

#     Returns:
#         AUC (float): Area under the ROC curve.
#     """
#     assert IS.shape[0] == IN.shape[0], "The size of IS and IN should be the same."

#     # Prepare rotational symmetric frequency channels
#     Nx = int(np.sqrt(IS.shape[0]))  # Assuming square images
#     Ny = Nx
#     dx = 0.44  # Pixel size (cm)
#     dy = dx
#     x_off = Nx / 2
#     y_off = Ny / 2

#     i = np.arange(Nx)
#     j = np.arange(Ny)
#     x = (i - x_off) * dx
#     y = (j - y_off) * dy

#     pb = np.array([1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4]) / dx  # Passband edges (cycles/cm)
#     U = np.zeros((Nx * Ny, 4))  # Channel matrix

#     for c in range(4):  # Iterate over channels
#         C = np.zeros((Nx, Ny))
#         for ix in range(Nx):  # x-loop
#             for iy in range(Ny):  # y-loop
#                 r = np.sqrt(x[ix]**2 + y[iy]**2)
#                 if r > 0:
#                     C[ix, iy] = (pb[c + 1] * j1(2 * np.pi * pb[c + 1] * r) -
#                                  pb[c] * j1(2 * np.pi * pb[c] * r)) / r
#                 else:
#                     C[ix, iy] = np.pi * (pb[c + 1]**2 - pb[c]**2)

#         C /= np.linalg.norm(C, 'fro')
#         U[:, c] = C.ravel()

#     # Compute channelized Hotelling observer outputs
#     vS = U.T @ IS
#     vN = U.T @ IN

#     # Mean difference image
#     g_bar = IS.mean(axis=1) - IN.mean(axis=1)

#     # Mean difference channel outputs
#     delta_v_bar = U.T @ g_bar

#     # Intra-class scatter matrix
#     S = 0.5 * np.cov(vN) + 0.5 * np.cov(vS)

#     # Hotelling template
#     w = np.linalg.inv(S) @ delta_v_bar

#     # Apply the Hotelling template to produce outputs
#     tS = w.T @ vS
#     tN = w.T @ vN

#     # Compute AUC and ROC curve
#     data = np.concatenate([tS, tN])
#     thresholds = np.sort(np.unique(data))[::-1]

#     tpf = []  # True positive rate
#     fpf = []  # False positive rate

#     for thresh in thresholds:
#         tpf.append(np.sum(tS > thresh) / len(tS))
#         fpf.append(np.sum(tN > thresh) / len(tN))

#     tpf.append(1.0)
#     fpf.append(1.0)

#     # Compute AUC using trapezoidal integration
#     AUC = np.trapz(tpf, fpf)

#     # Plot ROC curve and mean difference image
#     plt.figure(figsize=(14, 6))

#     # ROC curve
#     plt.subplot(1, 2, 1)
#     plt.plot(fpf, tpf, linewidth=2)
#     plt.title('ROC Curve')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')

#     # Mean difference image
#     plt.subplot(1, 2, 2)
#     plt.imshow(g_bar.reshape(Nx, Ny), cmap='gray')
#     plt.colorbar()
#     plt.title(r'$\Delta \overline{\hat{f}}$')

#     plt.tight_layout()
#     plt.show()

#     return AUC
