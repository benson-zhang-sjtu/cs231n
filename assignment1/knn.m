%% Load Data
clear; close all;
addpath('../dataset/cifar-10-batches-mat')
load 'data_batch_1'
X_train = reshape(data, [10000,32,32,3]); X_train = permute(X_train, [1,3,2,4]);
y_train = labels;

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
        imshow(reshape(X_train(idxs(j), :, :, :), [m, n, 3]));
        if (j == 1)
            title(classes(i))
        end
    end
end
    