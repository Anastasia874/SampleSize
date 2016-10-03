function [res, coverage, cov_length] = coverage_probability(y, X, W, pd, pr_pd, ...
                                                                    parnames)

% W is [N x M] matrix of sampled parameters, where M is the number of
% samples, N is the number of parameters
% pd [1 x N] - data likelihood functions
% parnames [1 x N] - names of pd parameters
% conf_int  - half widht of the confidence interval for the parameter of
% interest

CONFIDENCE = 1; % ratio of diviance (pars -mle) in l2 to l2 norm of mle
INTERCEPT = 0.05;
nW_samples = size(W, 2); % number of MCMC samples of posterior parameters


%SIGNIFICANCE = 0.05;

min_dist = zeros(nW_samples, 1);
coverage = zeros(nW_samples, 1);
cov_length = zeros(nW_samples, 1);

%test_errors = zeros(N_DKL_SAMPLES, length(m_vec));
%train_errors = zeros(N_DKL_SAMPLES, length(m_vec));

cls = unique(y);
probs = zeros(1, length(pd));
for i = 1:length(cls)
    probs(i) = mean(y == cls(i));
end

options = optimset('Display','off');
mle_w = fminsearch(@(w) -1*loglike_func(y, X, probs, pd, w, parnames) ...
                        - logprior(w, pr_pd), ...
                           randn(size(W, 1), 1), options);
%mle(@(w) loglike_func(y, X, probs, pd, w, parnames));


for i = 1:nW_samples
    dist_to_opt = sort(pdist2(W(:, i)', mle_w')/norm(mle_w));
        
    cov_by_d = (1:length(dist_to_opt))/length(dist_to_opt);
    idx_dist = cov_by_d == min(cov_by_d(cov_by_d > 1 - INTERCEPT));
    cov_length(i) = dist_to_opt(idx_dist);
    coverage(i) = mean(pdist2(W(:, i)', mle_w')...
                                    <= CONFIDENCE*norm(mle_w));
    min_dist(i) = min(pdist2(W(:, i)', mle_w')/norm(mle_w));
    %
end

res = {};
res.cov_prob = coverage;
res.cov_len = cov_length;
res.min_dist = min_dist;

coverage = mean(coverage);
cov_length = mean(cov_length);
disp(['Coverage probability: ', num2str(coverage), ...
    ' min dist to norm ', num2str(mean(min_dist))])


end