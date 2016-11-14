classdef LinearClassifier
    properties
        W = nan;
    end
    methods(Abstract)
        [loss, grad] = loss(obj, X_batch, y_batch, reg)
    end
    methods
        function [obj, loss_history] = train(obj, X, y, learning_rate, reg, num_iters, batch_size, verbose)
            [num_train, dim] = size(X);
            num_classes = max(y) + 1;
            if isnan(obj.W)
                obj.W = 0.001 * randn(dim, num_classes);
            end
            
            loss_history = zeros([1 num_iters], 'double');
            for i=1:num_iters
                indices = randi(num_train, [1 batch_size]);
                X_batch = X(indices, :);
                y_batch = y(indices);
                [loss, grad] = obj.loss(X_batch, y_batch, reg);
                loss_history(i) = loss;
                obj.W = obj.W - learning_rate * grad;
                
                if mod(i, 100) == 0 && verbose
                    fprintf('iteration %d / %d: loss %f\n', i, num_iters, loss);
                end
            end
        end 
        
        function y_pred = predict(obj, X)
            scores = X * obj.W;
            [~, y_pred] = max(scores, [] ,2);
        end
    end
end

    