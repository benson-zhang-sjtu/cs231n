function dists = compute_distances_no_loop(X_train, X_test)
    num_test = size(X_test, 1);
    num_train = size(X_train, 1);
    square_train = sum(X_train.^2, 2);
    square_test = sum(X_test.^2, 2);
    train = []; test = [];
    for i=1:num_train
        test = horzcat(test, square_test);
    end
    for i=1:num_test
        train = horzcat(train, square_train);
    end
    
    dists = sqrt(train' + test - 2*X_test*X_train');
end