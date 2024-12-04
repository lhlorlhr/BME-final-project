function evaluate_denoising(noisy_image, denoised_image, low_noise_image)
    % Calculate RMSE
    rmse = sqrt(mean((double(denoised_image(:)) - double(low_noise_image(:))).^2));
    fprintf("RMSE: %.4f\n", rmse);

    % Visualization
    figure;
    subplot(1, 3, 1);
    imshow(noisy_image, []);
    title('Noisy Image');

    subplot(1, 3, 2);
    imshow(denoised_image, []);
    title('Denoised Image');

    subplot(1, 3, 3);
    imshow(low_noise_image, []);
    title('Low-Noise Image');

    sgtitle('Comparison of Noisy, Denoised, and Low-Noise Images');
end
