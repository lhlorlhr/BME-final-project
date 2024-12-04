% Load system matrix and projection data
load('H.mat'); % System matrix
load('g_noisy.mat'); % Noisy projection data

% Perform SVD on the system matrix
[U, S, V] = svd(H, 'econ');

% Regularization: Keep only the largest singular values
threshold = 0.1 * max(diag(S)); % Retain singular values above 10% of the maximum
S_regularized = S .* (diag(S) > threshold);

% Reconstruct the image using the regularized pseudoinverse
H_regularized_inv = V * pinv(S_regularized) * U';
f_reconstructed = H_regularized_inv * g_noisy;

% Visualize the reconstruction
figure;
imagesc(reshape(f_reconstructed, 32, 32));
colormap gray;
colorbar;
title('Regularized Reconstruction');
xlabel('x');
ylabel('y');

% Compare with original image or low-noise reconstruction
% Assuming original is stored in 'f_original.mat'
load('f_original.mat');
rmse = sqrt(mean((f_reconstructed(:) - f_original(:)).^2));
fprintf('RMSE of Regularized Reconstruction: %.4f\n', rmse);
