function y_pred = predict_labels(num_test, y_train, dists, k)
    y_pred = zeros([num_test,1]);
    for i=1:num_test
        [~, ind] = sort(dists(i, :), 2);
        closest_y = y_train(ind(1:k));
        y_pred(i) = mode(closest_y);
    end
end