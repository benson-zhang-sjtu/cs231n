classdef LinearSVM < LinearClassifier
    methods
        function [loss, grad] = loss(obj, X_batch, y_batch, reg)
            [loss, grad] = svm_loss_vectorized(obj.W, X_batch, y_batch, reg);
        end
    end
end