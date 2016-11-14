function [loss, dW] = svm_loss_vectorized(W, X, y, reg)
    dW = zeros(size(W), 'double');  
    num_classes = size(W, 2);
    num_train = size(X, 1);
    score = X * W; scoret = score'; 
    ind = 0:num_train-1;
    yn = double(y) + 1 + num_classes*ind';
    correct_scores = scoret(yn);
    M = bsxfun(@minus, score, correct_scores) + 1;
    loss = sum(sum(max(M, 0))) / num_train - 1 + 0.5 * reg * sum(sum(W .* W));
    
    % calculate dW
    C = double(M > 0);
    cnt = sum(C, 2) - 1; 
    for i=1:num_train
        C(i, y(i)+1) = - cnt(i);
    end
    dW = X' * C / num_train + reg * W;
end