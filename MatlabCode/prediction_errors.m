function [res, err, probs] = prediction_errors(y, X, W, pd, pr_pd, parnames, probs, cls)

m = length(y);
nW_samples = size(W, 2);
err = zeros(m, 1);
predictions = zeros(m, nW_samples);
if ~exist('probs', 'var')
cls = unique(y);
end
num_cls = length(cls);
lh = zeros(m, num_cls);

if ~exist('probs', 'var')
probs = zeros(1, length(pd));
for i = 1:length(cls)
    probs(i) = mean(y == cls(i));
end
end

for nw = 1:nW_samples
    for ncls = 1:num_cls
        lh(:, ncls) = loglike_func(ones(size(y))*cls(ncls), X, probs, pd, W(:, nw), parnames);
    end
    [~, idx_cls] = max(lh, [], 2);
    try
    predictions(:, nw) = cls(idx_cls);
    catch
        disp(': (')
    end
    err(:, nw) = predictions(:, nw) ~= y;
end

res = [];
res.err = err;
res.predictions = predictions;

err = mean(err);

end