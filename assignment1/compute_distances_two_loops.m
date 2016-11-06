function dists = compute_distances_two_loops(X_train, X_test)
    num_test = size(X_test, 1);
    num_train = size(X_train, 1);

    dists = zeros([num_test, num_train]);
    for i=1:num_test
        for j=1:num_train
            dists(i, j) = sqrt(sum((X_test(i, :) - X_train(j, :)).^2));
        end
    end
end