%% Load Data, 这里要将data改成double型，不然后面会有错误。
clear; close all;
addpath('../../dataset/cifar-10-batches-mat');
for i=1:5
    fname = sprintf('data_batch_%d', i);
    load(fname); data = double(data);
    X = reshape(data, [10000,32,32,3]); X = permute(X, [1,3,2,4]); 
    if i == 1
        X_train = X;
        y_train = labels;
    else
        X_train = vertcat(X_train, X);
        y_train = vertcat(y_train, labels);
    end
end

load('test_batch'); 
X = reshape(data, [10000,32,32,3]);  X = double(X); X_test = permute(X, [1,3,2,4]);
y_test = labels;

%% Display Images
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
num_training = 5000;
X_train = X_train(1:num_training, :, :, :);
y_train = y_train(1:num_training);

num_test = 500;
X_test = X_test(1:num_test, :, :, :);
y_test = y_test(1:num_test);

%% Reshape image data into rows
X_train = reshape(X_train, size(X_train, 1),[]);
X_test =  reshape(X_test, size(X_test, 1),[]);

%% KNN 
dists = compute_distances_two_loops(X_train, X_test);

%%
k = 5; %1
y_pred = predict_labels(num_test, y_train, dists, k);
num_correct = sum((y_test == y_pred));
accuracy = num_correct / num_test;
disp(sprintf('Got %d / %d correct => accuracy: %f', num_correct, num_test, accuracy))

%% No loop
dists_no = compute_distances_no_loop(X_train, X_test);
difference = dists_no - dists;
disp(sum(difference(:)))

%% Measure
tic
dists = compute_distances_two_loops(X_train, X_test);
toc
%% Measure
tic
dists_no = compute_distances_no_loop(X_train, X_test);
toc

%% Cross-validation
num_folds = 5;
fold_size = size(X_train, 1) / num_folds;
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100];
k_to_accuracies = zeros([size(k_choices, 2), num_folds]);

X_train_folds = zeros([num_folds, fold_size, size(X_train, 2)]);
y_train_folds = zeros([num_folds, fold_size]);

for i=1:num_folds
    X_train_folds(i, :, :) = X_train(1+(i-1)*fold_size:i*fold_size, :);
    y_train_folds(i, :) = y_train(1+(i-1)*fold_size:i*fold_size);
end

%%
for i=1:size(k_choices, 2)
    for j=1:num_folds
        X_input = []; y_input = [];
        for k=1:num_folds
            if (k~=j)
                X_input = vertcat(X_input, reshape(X_train_folds(k, :, :), [fold_size, size(X_train, 2)]));
                y_input = vertcat(y_input, reshape(y_train_folds(k, :), [fold_size, 1]));
            end
        end
        dists = compute_distances_no_loop(X_input ,reshape(X_train_folds(j, :, :), [fold_size, size(X_train, 2)]));
        y_pred_fold = predict_labels(fold_size, y_input, dists, k_choices(i));
        
        num_correct = sum((reshape(y_train_folds(j, :), [fold_size, 1]) == y_pred_fold));
        accuracy = num_correct / fold_size;
        k_to_accuracies(i, j) = accuracy;
    end
end

%%
for i=1:size(k_choices, 2)
    for j=1:num_folds
        disp(sprintf('k = %d, accuracy = %f', k_choices(i), k_to_accuracies(i, j)));
    end
end

%% Plot the obeservation
figure; hold on
xlim([-20 120])
for i=1:size(k_choices, 2)
   for j=1:num_folds
       scatter(k_choices(i), k_to_accuracies(i, j), 'filled', 'b');
   end
end

accuracies_mean = mean(k_to_accuracies, 2);
accuracies_std = std(k_to_accuracies, 1 ,2);
errorbar(k_choices, accuracies_mean, accuracies_std, 'b');
title('Cross-validation on k')
xlabel('k')
ylabel('Cross-validation accuracy')

%% Pick Best k
[~, index] = max(accuracies_mean);
best_k = k_choices(index);

dists_no = compute_distances_no_loop(X_train, X_test);
y_pred = predict_labels(num_test, y_train, dists_no, best_k);
num_correct = sum((y_test == y_pred));
accuracy = num_correct / num_test;
disp(sprintf('Got %d / %d correct => accuracy: %f', num_correct, num_test, accuracy))




 

