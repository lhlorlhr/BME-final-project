function [AUC] = performance_evaluation(IS,IN)
%
% [AUC] = performance_evaluation(IS,IN)
%
% Conduct an observer study using channelized Hotelling observer.
% The signal-present images are given by IS, the signal-absent are images given by
% IN. Then compute the AUC, and plot the ROC curves (by simple 
% thresholding).
%
% Inputs: 
%       IS, IN -- [Number_of_Pixels X Number_of_Images] -- The input images 
%                 (Each column represents one image)
%
% Outputs:
%       AUC -- Wilcoxon area under the ROC curve
%       Figures -- ROC curve and delta f_hat_bar image.
%       (delta f_hat_bar: the difference between signal-present and signal-absent images)
%
% Edited by Zitong Yu @ Nov-16-2021
% Based on IQmodelo toolbox Apr-15-2016© the Food and Drug Administration (FDA)
% Contact: yu.zitong@wustl.edu

%% % Define the path to the predicted images and the ground truth images
pred_dir = './data/test/prediction/'; % Path to predictions (adjust accordingly)
ground_truth_signal_present_dir = './data/test/signal_present/'; % Signal-present images path
ground_truth_signal_absent_dir = './data/test/signal_absent/'; % Signal-absent images path

% Number of images for evaluation (adjust accordingly)
num_signal_present = 100; % Number of signal-present images
num_signal_absent = 100;  % Number of signal-absent images

% Initialize matrices to store the images
IS = [];
IN = [];

% Load signal-present images (ground truth)
for img_ind = 1:num_signal_present
    img = readNPY(fullfile(ground_truth_signal_present_dir, ['image', num2str(img_ind), '.npy']));
    IS = [IS, img(:)]; % Append each image as a column vector
end

% Load signal-absent images (predicted or ground truth)
for img_ind = 1:num_signal_absent
    img = readNPY(fullfile(ground_truth_signal_absent_dir, ['image', num2str(img_ind), '.npy']));
    IN = [IN, img(:)]; % Append each image as a column vector
end

% Load the predicted denoised images
for img_ind = 1:num_signal_present
    pred_img = readNPY(fullfile(pred_dir, ['image', num2str(img_ind), '.npy']));
    % Use the predicted images for evaluation
    % You can replace IS or IN with the predictions if you want to compare
    % For example:
    % IS = [IS, pred_img(:)];
end

% Run performance evaluation
[AUC] = performance_evaluation(IS, IN);

% The function will compute the AUC and plot the ROC curve along with
% delta f_hat_bar, which shows the mean difference between signal-present and signal-absent images.

%%
% Define the path to the predicted images and the ground truth images
pred_dir = './data/test/prediction/'; % Path to predictions (adjust accordingly)
ground_truth_signal_present_dir = './data/test/signal_present/'; % Signal-present images path
ground_truth_signal_absent_dir = './data/test/signal_absent/'; % Signal-absent images path

% Number of images for evaluation (adjust accordingly)
num_signal_present = 50; % Number of signal-present images
num_signal_absent = 50;  % Number of signal-absent images

% Initialize matrices to store the images
IS = [];
IN = [];

% Load signal-present images (ground truth)
for img_ind = 1:num_signal_present
    img = readNPY(fullfile(ground_truth_signal_present_dir, ['image', num2str(img_ind), '.npy']));
    IS = [IS, img(:)]; % Append each image as a column vector
end

% Load signal-absent images (predicted or ground truth)
for img_ind = 1:num_signal_absent
    img = readNPY(fullfile(ground_truth_signal_absent_dir, ['image', num2str(img_ind), '.npy']));
    IN = [IN, img(:)]; % Append each image as a column vector
end

% Load the predicted denoised images
for img_ind = 1:num_signal_present
    pred_img = readNPY(fullfile(pred_dir, ['image', num2str(img_ind), '.npy']));
    % Use the predicted images for evaluation
    % You can replace IS or IN with the predictions if you want to compare
    % For example:
    % IS = [IS, pred_img(:)];
end

% Run performance evaluation
[AUC] = performance_evaluation(IS, IN);

% The function will compute the AUC and plot the ROC curve along with
% delta f_hat_bar, which shows the mean difference between signal-present and signal-absent images.



%% prepare rotational sysmmetric frequency channels
assert(size(IS,1)==size(IN,1), 'The size of IS and IN should be same.')
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

pb = [1/64,1/32,1/16,1/8,1/4]/dx;  % edges of passbands (cycles/cm)
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

%% Compute the channelized Hotelling observer outputs

% compute channel outputs
vS = U'*IS;
vN = U'*IN;

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

%% Compute the AUC and plot the ROC curve

data=[tS(:) ; tN(:)];
xs = sort(data);
xs = unique(xs);

tpf = zeros(length(xs)+1,1);
fpf = zeros(length(xs)+1,1);

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

f = figure;
subplot(1,2,1);
plot(fpf,tpf,'LineWidth',2);
title('ROC curve');
xlabel('False positive ratio');
ylabel('True positive ratio');

subplot(1,2,2);
imagesc(reshape(mean(IS - IN, 2), Nx, Ny));colormap gray;title('$$\Delta\overline{\hat{f}}$$','interpreter','latex');
f.Position = [100,100,1400,550];
set(findall(gcf,'-property','FontSize'),'FontSize',18)
end