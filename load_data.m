addpath('../dataset/cifar-10-batches-mat')
load 'data_batch_1'
X = reshape(data, [10000,32,32,3]);

%%
for id = 1:6
    fname = sprintf('../dataset/cifar-10-batches-mat/data_batch_%d', id);
    load fname;
    X = reshape(data, [10000, 32, 32, 3]);
    Y = labels;
end
    
