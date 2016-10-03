function m = average_coverage_ssd(y, X, conf_int, intercept)

% conf_int  - half widht of the confidence interval for the parameter of
% interest
% intercept is anlogous of significance in hypothesis testing
% 

N_DKL_SAMPLES = 50; % number of dkl samples
LAMBDA = 0.5; % trade-off between generative and discriminative
ALPHA = 0.01; % diagonal of the covariance matrix
N_W_SAMPLES = 5000; % number of MCMC samples of posterior parameters

M = length(y); % max sample size for dkl evaluation

m_vec = [20, 50:10:100, 150:50:500, 750, 1000];
m_vec = m_vec(m_vec <= M);

%SIGNIFICANCE = 0.05;


dkl = zeros(N_DKL_SAMPLES, length(m_vec));
min_dist = zeros(N_DKL_SAMPLES, length(m_vec));
coverage = zeros(N_DKL_SAMPLES, length(m_vec));
%test_errors = zeros(N_DKL_SAMPLES, length(m_vec));
%train_errors = zeros(N_DKL_SAMPLES, length(m_vec));

for m = m_vec
    
    for i = 1:N_DKL_SAMPLES
        idx = randperm(M, m);
        % obtain a sample of posterior parameters for the sample D_m
        [hat_wd, hat_wg, ~, ~, ~, ~] = gdl_par(y(idx), X(idx, :), LAMBDA, ALPHA);

        [wd, wg] = MonteCarloParameters(y(idx), X(idx,:), LAMBDA, ALPHA, ...
                                                          N_W_SAMPLES);
        hat_w = [hat_wd(:); hat_wg(:)]';
        coverage(i, m_vec == m) = mean(pdist2([wd, wg], hat_w)...
                                        <= conf_int*norm(hat_w));
        min_dist(i, m_vec == m) = min(pdist2([wd, wg], hat_w)/norm(hat_w));
        %save('res/dkl_dloo.mat', 'dkl', 'dkl_dloo'); 
    end
    %{                    
    [h(m_vec == m), p(m_vec == m), C(m_vec == m), chi2fit(m_vec == m)] = ...
                            infer_chi2_stats(dkl(:, m_vec == m)*2*m, ...
                            SIGNIFICANCE,...
                            ['tmp/dkl_hist_pdf_central_m', num2str(m)]);
    %}                   
end


accepted_m = mean(coverage) > 1 - intercept;
m = min(m_vec(accepted_m));


end


