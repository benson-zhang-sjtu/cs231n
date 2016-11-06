function [X_train, y_train, X_test, y_test] = load_CIFAR10(cifar10_dir)
    for i=1:5
        fname = strcat(cifar10_dir, '/',sprintf('data_batch_%d', i));
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
    fname = strcat(cifar10_dir, '/test_batch');
    load(fname); 
    X = reshape(data, [10000,32,32,3]);  X = double(X); X_test = permute(X, [1,3,2,4]);
    y_test = labels;
end