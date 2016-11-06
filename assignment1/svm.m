%% Load the raw CIFAR-10 data.
clear; close all;clc;
cifar10_dir = '../../dataset/cifar-10-batches-mat';
[X_train, y_train, X_test, y_test] = load_CIFAR10(cifar10_dir); 

%% Visualize
classes = {'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'};
m = 32; n = 32;
num_classes = size(classes, 2);
samples_per_class = 7;
for i = 1:num_classes
    idxs = find(y_train == i-1);
    idxs = randsrc(1, samples_per_class, idxs');
    for j = 1:samples_per_class
        plt_idx = (j-1) * num_classes + i;
        subplot(samples_per_class, num_classes, plt_idx);
        imshow(uint8(reshape(X_train(idxs(j), :, :, :), [m, n, 3])));
        if (j == 1)
            title(classes(i))
        end
    end
end

%% Subsample the data
num_training   = 49000;
num_validation = 1000;
num_test       = 1000;

mask = num_training+1:num_training+num_validation;
X_val = X_train(mask, :, :, :);
y_val = y_train(mask);

mask = 1:num_training;
X_train = X_train(mask, :, :, :);
y_train = y_train(mask);

mask = 1:num_test;
X_test = X_test(mask, :, :, :);
y_test = y_test(mask);

%% Reshape image data into rows
X_train = reshape(X_train, size(X_train, 1),[]);
X_test =  reshape(X_test, size(X_test, 1),[]);
X_val =  reshape(X_val, size(X_val, 1),[]);

%% Preprocessing: subtract the mean image
mean_image = mean(X_train, 1);
imshow(uint8(reshape(mean_image, [32, 32, 3])));

%% bsxfun broadcasting like python
X_train = bsxfun(@minus, X_train, mean_image);
X_val   = bsxfun(@minus, X_val, mean_image);
X_test  = bsxfun(@minus, X_test, mean_image);

%% append the bias dimension of ones
X_train = horzcat(X_train, ones([size(X_train, 1), 1]))';
X_val   = horzcat(X_val, ones([size(X_val, 1), 1]))';
X_test  = horzcat(X_test, ones([size(X_test, 1), 1]))';

%% SVM Classifier

