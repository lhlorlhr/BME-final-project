<h1>BME 470/570 Project Code</h1>

• **image_generation.py:** Python code for generating the population of objects and reconstructed noisy images with signal-present and signal-absent cases.

• **train_cnn.py:** Python code for preparing datasets, predicting images, and evaluating results.

• **DL_denoiser.py:** Python code from previous project [1] for implementing the encoder-decoder architecture and training routine.

• **performance_evaluation.m:** MATLAB code based on previous project [1] for calculating AUC and generating ROC curves.

• **performance_evaluation_multiroc.m:** MATLAB code based on previous project [1] for generating ROC curves for label, noisy, and denoised dataset.

For preparing datasets, training model, evaluating model performance in Python:

> python train_cnn.py 


For generating AUC and ROC curves in MATLAB:

> noisy_label_medium_absent = readNPY(['./Medium/test_labels_absent.npy']);
> 
> noisy_label_medium_present = readNPY(['./Medium/test_labels_present.npy']);
> 
> noisy_label_medium_present = reshape(permute(noisy_label_medium_present, [2, 3, 1]), 32*32, 100);
> 
> noisy_label_medium_absent = reshape(permute(noisy_label_medium_absent, [2, 3, 1]), 32*32, 100);
> 
> [AUC1] = performance_evaluation(noisy_label_medium_present, noisy_label_medium_absent);
>
> [AUCs1] = performance_evaluation_multiroc(noisy_label_medium_present, noisy_label_medium_absent, noisy_test_medium_present, noisy_test_medium_absent, denoised_test_medium_present, denoised_test_medium_absent)




[1] https://github.com/YuZitong/BME-570-project-code/tree/main
