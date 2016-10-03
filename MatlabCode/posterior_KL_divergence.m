function [res, dkl] = posterior_KL_divergence(y, X, W, pd, pr_pd, parnames)
% Parametric version

SIGNIFICANCE = 0.05;
MAX_N_LOO = 1000;

% this is an LOO-estimation of KL-divergence. Since W is the sample of
% posterior parameters w|D_m, averaging over W [log(w|D)/log(w|D')]
% approximates KL(D|D'). 
m = length(y);
nW_samples = size(X, 2);
nLooSamples = min(m, MAX_N_LOO);
dkl = zeros(nW_samples, nLooSamples);

idx = 1:m;
idx_loo = randperm(m, nLooSamples);


cls = unique(y);
probs = zeros(1, length(cls));
for i = 1:length(cls)
    probs(i) = mean(y == cls(i));
end


% calc "full" likelihood and evidence:
lh = zeros(1, nW_samples);
prior = zeros(1, nW_samples);
for nw = 1:nW_samples
    lh(nw) = loglike_func(y, X, probs, pd, W(:, nw), parnames);
    prior(nw) = exp(logprior(W(:, nw), pr_pd));
end
evidence_full = log(sum(exp(lh).*prior));
    
lh1 = zeros(1, nw);
evd1 = zeros(1, nLooSamples);

for i = idx_loo
    idx1 = idx(idx ~= i);
    y1 = y(idx1);
    X1 = X(idx1, :);
    for nw = 1:nW_samples
        lh1(nw) = loglike_func(y1, X1, probs, pd, W(:, nw), parnames);
    end
    evd1(i) = log(sum(exp(lh1).*prior));   
    dkl(i) = mean(lh./lh1) - evidence_full + evd1(i);
end
%                    
[h, p, C, chi2fit] = infer_chi2_stats(dkl*2*m, ...
                     SIGNIFICANCE, ['tmp/dkl_hist_pdf_central_m', num2str(m)]);
%}


res = {};
res.dkl = dkl;
res.evidence = [evidence_full, evd1];
res.chi2stats = [h, p, C, chi2fit];

end
