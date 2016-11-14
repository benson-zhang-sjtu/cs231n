function [loss, dW] = svm_loss_naive(W, X, y, reg)
    dW = zeros(size(W), 'double');
    num_classes = size(W, 2);
    num_train = size(X, 1);
    loss = 0.0;
    for i=1:num_train
        scores = X(i,:) * W;
        correct_class_score = scores(y(i)+1);
        cnt = 0;
        for j=1:num_classes
            if j~=y(i)+1
                margin = scores(j) - correct_class_score + 1;
                if margin > 0
                    loss = loss + margin;
                    dW(:, j) = dW(:, j) + X(i, :)';
                    cnt = cnt + 1;
                end
            end
        end
        dW(:, y(i)+1) = dW(:, y(i)+1) - cnt * X(i, :)';
    end
    dW = dW / num_train + reg * W;
    loss = loss / num_train;
    loss = loss + 0.5 * reg * sum(sum(W .* W));
end