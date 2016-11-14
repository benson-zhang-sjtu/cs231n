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
num_dev        = 500;

mask = num_training+1:num_training+num_validation;
X_val = X_train(mask, :, :, :);
y_val = y_train(mask);

mask = 1:num_training;
X_train = X_train(mask, :, :, :);
y_train = y_train(mask);

mask = 1:num_test;
X_test = X_test(mask, :, :, :);
y_test = y_test(mask);

mask = randperm(num_training, num_dev);
X_dev = X_train(mask, :, :, :);
y_dev = y_train(mask);

%% Reshape image data into rows
X_train = reshape(X_train, size(X_train, 1),[]);
X_test =  reshape(X_test, size(X_test, 1),[]);
X_val =  reshape(X_val, size(X_val, 1),[]);
X_dev = reshape(X_dev, size(X_dev, 1),[]);

%% Preprocessing: subtract the mean image
mean_image = mean(X_train, 1);
imshow(uint8(reshape(mean_image, [32, 32, 3])));

%% bsxfun broadcasting like python
X_train = bsxfun(@minus, X_train, mean_image);
X_val   = bsxfun(@minus, X_val, mean_image);
X_test  = bsxfun(@minus, X_test, mean_image);
X_dev  = bsxfun(@minus, X_dev, mean_image);

%% append the bias dimension of ones
X_train = horzcat(X_train, ones([size(X_train, 1), 1]));
X_val   = horzcat(X_val, ones([size(X_val, 1), 1]));
X_test  = horzcat(X_test, ones([size(X_test, 1), 1]));
X_dev  = horzcat(X_dev, ones([size(X_dev, 1), 1]));

%% SVM Classifier
W = randn(3073, 10) * 0.0001;
[loss, ~] = svm_loss_naive(W, X_dev, y_dev, 0.00001);
disp(sprintf('loss£º %f', loss));

%% gradient check
[~, grad] = svm_loss_naive(W, X_dev, y_dev, 0.0);
f = @(w) svm_loss_naive(w, X_dev, y_dev, 0.0);
grad_check_sparse(f, W, grad);
[loss, grad] = svm_loss_naive(W, X_dev, y_dev, 1e-2);
f = @(w) svm_loss_naive(w, X_dev, y_dev, 1e-2);
grad_check_sparse(f, W, grad);

%% Measurement
tic;
[loss_naive, grad_naive] = svm_loss_naive(W, X_dev, y_dev, 0.00001);
toc;

tic;
[loss_vectorized, ~] = svm_loss_vectorized(W, X_dev, y_dev, 0.00001);
toc

disp(sprintf('difference: %f', loss_naive - loss_vectorized));

%% 
tic;
[~, grad_naive] = svm_loss_naive(W, X_dev, y_dev, 0.00001);
toc;

tic;
[~, grad_vectorized] = svm_loss_vectorized(W, X_dev, y_dev, 0.00001);
toc

disp(sprintf('difference: %f', norm(grad_naive - grad_vectorized, 'fro')));

%%
svm = LinearSVM;
tic;
[svm, loss_hist] = svm.train(X_train, y_train, 1e-7, 5e4, 1500, 200, true);
toc

%%
figure;
plot(loss_hist); xlabel('Iteration number'); ylabel('Loss value');

%%
y_train_pred = svm.predict(X_train);
fprintf('training accuracy: %f\n', mean(y_train_pred-1 == y_train))
y_val_pred = svm.predict(X_val);
fprintf('training accuracy: %f\n', mean(y_val_pred-1 == y_val))

%%
learning_rate = [1e-7] %5e-7, 1e-8];
regularization = [3e4] %5e4, 1e5];

best_val = -1;

train_accuracy = zeros([size(learning_rate, 2), size(regularization, 2)], 'double');
val_accuracy = zeros([size(learning_rate, 2), size(regularization, 2)], 'double');
for i=1:size(learning_rate, 2)
    for j=1:size(regularization, 2)
        disp([i,j])
        lr = learning_rate(i); reg = regularization(i);
        svm = LinearSVM;
        [svm, ~] = svm.train(X_train, y_train, lr, reg, 1500, 200, false);
        y_train_pred = svm.predict(X_train);
        train_accuracy(i, j) = mean(y_train_pred-1 == y_train);
        y_val_pred = svm.predict(X_val);
        val_accuracy(i,j) = mean(y_val_pred-1 == y_val);
        if val_accuracy > best_val
            best_val = val_accuracy(i,j);
            best_svm = svm;
        end
    end
end

for i=1:size(learning_rate, 2)
    for j=1:size(regularization, 2)
        fprintf('lr %e reg %e train accuracy: %f val accuracy: %f\n', learning_rate(i),regularization(j), train_accuracy(i, j), val_accuracy(i,j));
    end
end

fprintf('best validation accuracy achieved during cross-validation: %f\n', best_val);

%%
[x_scatter, y_scatter] = meshgrid(log10(learning_rate), log10(regularization));

subplot(2,1,1)
scatter(x_scatter(:), y_scatter(:), [], train_accuracy(:))
colorbar
xlabel('log learning rate');
ylabel('log regularization strength');
title('CIFAR-10 training accuracy');

subplot(2,1,2)
scatter(x_scatter(:), y_scatter(:), [], val_accuracy(:))
colorbar
xlabel('log learning rate');
ylabel('log regularization strength');
title('CIFAR-10 validation accuracy');

%%
y_test_pred = best_svm.predict(X_test);
test_accuracy = mean(y_test_pred-1 == y_test);
fprintf('linear SVM on raw pixels final test set accuracy: %f\n', test_accuracy);

%%
W = best_svm.W(1:end-1, :);
W = reshape(W, [32, 32, 3, 10]);
w_min = min(W(:)); w_max = max(W(:));
classes = {'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'};
for i=1:10
    subplot(2,5,i);
    wimg = (squeeze(W(:,:,:,i)) - w_min) / (w_max - w_min);
    imshow(wimg);
    title(classes(i))
end
