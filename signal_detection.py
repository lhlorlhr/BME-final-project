import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1  # Bessel function of the first kind

def performance_evaluation(IS, IN):
    """
    Perform signal-detection task using channelized Hotelling observer.

    Args:
        IS (numpy.ndarray): Signal-present images (pixels x images).
        IN (numpy.ndarray): Signal-absent images (pixels x images).

    Returns:
        AUC (float): Area under the ROC curve.
    """
    assert IS.shape[0] == IN.shape[0], "The size of IS and IN should be the same."

    # Prepare rotational symmetric frequency channels
    Nx = int(np.sqrt(IS.shape[0]))  # Assuming square images
    Ny = Nx
    dx = 0.44  # Pixel size (cm)
    dy = dx
    x_off = Nx / 2
    y_off = Ny / 2

    i = np.arange(Nx)
    j = np.arange(Ny)
    x = (i - x_off) * dx
    y = (j - y_off) * dy

    pb = np.array([1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4]) / dx  # Passband edges (cycles/cm)
    U = np.zeros((Nx * Ny, 4))  # Channel matrix

    for c in range(4):  # Iterate over channels
        C = np.zeros((Nx, Ny))
        for ix in range(Nx):  # x-loop
            for iy in range(Ny):  # y-loop
                r = np.sqrt(x[ix]**2 + y[iy]**2)
                if r > 0:
                    C[ix, iy] = (pb[c + 1] * j1(2 * np.pi * pb[c + 1] * r) -
                                 pb[c] * j1(2 * np.pi * pb[c] * r)) / r
                else:
                    C[ix, iy] = np.pi * (pb[c + 1]**2 - pb[c]**2)

        C /= np.linalg.norm(C, 'fro')
        U[:, c] = C.ravel()

    # Compute channelized Hotelling observer outputs
    vS = U.T @ IS
    vN = U.T @ IN

    # Mean difference image
    g_bar = IS.mean(axis=1) - IN.mean(axis=1)

    # Mean difference channel outputs
    delta_v_bar = U.T @ g_bar

    # Intra-class scatter matrix
    S = 0.5 * np.cov(vN) + 0.5 * np.cov(vS)

    # Hotelling template
    w = np.linalg.inv(S) @ delta_v_bar

    # Apply the Hotelling template to produce outputs
    tS = w.T @ vS
    tN = w.T @ vN

    # Compute AUC and ROC curve
    data = np.concatenate([tS, tN])
    thresholds = np.sort(np.unique(data))[::-1]

    tpf = []  # True positive rate
    fpf = []  # False positive rate

    for thresh in thresholds:
        tpf.append(np.sum(tS > thresh) / len(tS))
        fpf.append(np.sum(tN > thresh) / len(tN))

    tpf.append(1.0)
    fpf.append(1.0)

    # Compute AUC using trapezoidal integration
    AUC = np.trapz(tpf, fpf)

    # Plot ROC curve and mean difference image
    plt.figure(figsize=(14, 6))

    # ROC curve
    plt.subplot(1, 2, 1)
    plt.plot(fpf, tpf, linewidth=2)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # Mean difference image
    plt.subplot(1, 2, 2)
    plt.imshow(g_bar.reshape(Nx, Ny), cmap='gray')
    plt.colorbar()
    plt.title(r'$\Delta \overline{\hat{f}}$')

    plt.tight_layout()
    plt.show()

    return AUC
