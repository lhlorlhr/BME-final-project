%%
function [AUCs] = performance_evaluation_multiroc(IS_label, IN_label, IS_noisy, IN_noisy, IS_denoised, IN_denoised)
%
% [AUCs] = performance_evaluation_multiroc(IS_label, IN_label, IS_noisy, IN_noisy, IS_denoised, IN_denoised)
%
% Conduct an observer study using channelized Hotelling observer.
% Generate ROC curves for label, noisy test data, and denoised test data.
%
% Inputs:
%       IS_label, IN_label: Signal-present and signal-absent label images.
%       IS_noisy, IN_noisy: Signal-present and signal-absent noisy test images.
%       IS_denoised, IN_denoised: Signal-present and signal-absent denoised test images.
%
% Outputs:
%       AUCs -- Struct containing AUCs for label, noisy, and denoised data.
%       Figure -- ROC curves for all three data types.
%

% Check input dimensions
assert(size(IS_label, 1) == size(IN_label, 1), 'The size of IS_label and IN_label should match.');
assert(size(IS_noisy, 1) == size(IN_noisy, 1), 'The size of IS_noisy and IN_noisy should match.');
assert(size(IS_denoised, 1) == size(IN_denoised, 1), 'The size of IS_denoised and IN_denoised should match.');

% Compute ROC and AUC for each dataset
[AUC_label, fpf_label, tpf_label] = compute_roc(IS_label, IN_label);
[AUC_noisy, fpf_noisy, tpf_noisy] = compute_roc(IS_noisy, IN_noisy);
[AUC_denoised, fpf_denoised, tpf_denoised] = compute_roc(IS_denoised, IN_denoised);

% Combine AUCs into a struct for output
AUCs = struct('label', AUC_label, 'noisy', AUC_noisy, 'denoised', AUC_denoised);

% Plot ROC curves
figure;
hold on;
plot(fpf_label, tpf_label, 'LineWidth', 2, 'DisplayName', 'Label ROC');
plot(fpf_noisy, tpf_noisy, 'LineWidth', 2, 'DisplayName', 'Noisy ROC');
plot(fpf_denoised, tpf_denoised, 'LineWidth', 2, 'DisplayName', 'Denoised ROC');
title('ROC Curves');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
legend('show');
grid on;
hold off;

end

% Helper function to compute ROC and AUC
function [AUC, fpf, tpf] = compute_roc(IS, IN)
    % % Prepare rotational symmetric frequency channels
    % Nx = sqrt(size(IS, 1));
    % dx = 0.44;  % Pixel size
    Nx = size(IS,1).^0.5;
    % assert(Nx == 64, 'The input image should have 64X64 pixels.')
    Ny = Nx;
    dx = 0.44; % assume pixel size is 0.44 cm
    dy = dx; % y pixel size
    x_off = Nx/2;
    y_off = Ny/2;
    i = 0:(Nx-1);
    j = 0:(Ny-1);
    x = (i-x_off)*dx;
    y = (j-y_off)*dy;
    X = [0;0];  % origin for channel grid
    pb = [1/64, 1/32, 1/16, 1/8, 1/4] / dx;  % Passband edges
    U = compute_channels(Nx, pb);

    % Compute channel outputs
    vS = U' * IS;
    vN = U' * IN;

    % compute mean difference image
    g_bar = mean(IS')-mean(IN');
    
    % compute mean difference channel outpus -- delta v bar
    delta_v_bar = U'*g_bar';
    
    % Intra-class scatter matrix
    S = .5*cov(vN') + .5*cov(vS');
    
    % Hotelling template
    w = inv(S) * delta_v_bar;
    
    % apply the Hotelling template to produce outputs
    tS = w' * vS;
    tN = w' * vN;

    % % Compute Hotelling observer template
    % g_bar = mean(IS, 2) - mean(IN, 2);
    % delta_v_bar = U' * g_bar;
    % S = 0.5 * cov(vN') + 0.5 * cov(vS');
    % reg_param = 1e-6;  % Regularization parameter
    % w = (S + reg_param * eye(size(S))) \ delta_v_bar;
    % 
    % % Apply template
    % tS = w' * vS;
    % tN = w' * vN;

    % Compute ROC curve
    data = [tS(:); tN(:)];
    xs = unique(sort(data));
    tpf = zeros(length(xs) + 1, 1);
    fpf = zeros(length(xs) + 1, 1);

    cnt = 1;
    % go backwards so that the ROC curve starts at (0,0).
    for thresh = fliplr(xs')
      tpf(cnt)=sum(tS > thresh)/length(tS);
      fpf(cnt)=sum(tN > thresh)/length(tN);
      cnt = cnt+1;
    end  
    tpf(cnt) = 1;
    fpf(cnt) = 1;
    
    AUC = trapz(fpf,tpf)
end

%     for i = 1:length(xs)
%         tpf(i) = sum(tS > xs(i)) / length(tS);  % True positive rate
%         fpf(i) = sum(tN > xs(i)) / length(tN);  % False positive rate
%     end
% 
%     tpf(end) = 1;
%     fpf(end) = 1;
% 
%     % Ensure monotonic order
%     [fpf, sort_idx] = sort(fpf);
%     tpf = tpf(sort_idx);
% 
%     % Compute AUC
%     AUC = trapz(fpf, tpf);
% end

% Helper function to compute frequency channels
function U = compute_channels(Nx, pb)
    Ny = Nx;
    dx = 0.44; % assume pixel size is 0.44 cm
    dy = dx; % y pixel size
    x_off = Nx/2;
    y_off = Ny/2;
    i = 0:(Nx-1);
    j = 0:(Ny-1);
    x = (i-x_off)*dx;
    y = (j-y_off)*dy;
    X = [0;0];  % origin for channel grid
    
    % pb = [1/64,1/32,1/16,1/8,1/4]/dx;  % edges of passbands (cycles/cm)
    C = zeros(Nx,Ny);
    U = zeros(Nx*Ny,4);
    x2 = x - X(1);
    y2 = y - X(2);
    for c=1:4
        for i=1:Nx % x-loop
           for j=1:Ny % y-loop
              r = sqrt(x2(i).^2 + y2(j)^2);
              if r > 0
                C(i,j) = (pb(c+1)*besselj(1,2*pi*pb(c+1)*r) - ...
                          pb(c)*besselj(1,2*pi*pb(c)*r))/r;
              else
                C(i,j) = pi*(pb(c+1)^2 - pb(c)^2);
              end
           end
        end
        C = C./norm(C,'fro');
        U(:,c) = C(1:(Nx*Ny))';
    end
end