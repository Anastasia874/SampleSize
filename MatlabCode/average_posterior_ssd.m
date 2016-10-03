function m = average_posterior_ssd(conf_int, intercept)

% conf_int  - half widht of the confidence interval for the parameter of
% interest
% intercept is anlogous of significance in hypothesis testing
%

mu = [0, 1]';
sigma2 = 1;
theta = 0.7;
m = 5000;

D = testData(mu, sigma2, theta, m);
X = D.X;
y = D.y;

N_DKL_SAMPLES = 2; % number of dkl samples
LAMBDA = 0.5; % trade-off between generative and discriminative
ALPHA = 0.01; % diagonal of the covariance matrix
N_W_SAMPLES = 5000; % number of MCMC samples of posterior parameters

M = length(y); % max sample size for dkl evaluation

m_vec = [20:10:100, 150:50:500, 750, 1000, 2000, 5000];
m_vec = m_vec(m_vec <= M);

%SIGNIFICANCE = 0.05;


min_dist = zeros(N_DKL_SAMPLES, length(m_vec));
cov_length = zeros(N_DKL_SAMPLES, length(m_vec));
coverage = zeros(N_DKL_SAMPLES, length(m_vec));
Aopt = zeros(N_DKL_SAMPLES, length(m_vec));
Dopt = zeros(N_DKL_SAMPLES, length(m_vec));
Gopt = zeros(N_DKL_SAMPLES, length(m_vec));
%test_errors = zeros(N_DKL_SAMPLES, length(m_vec));
%train_errors = zeros(N_DKL_SAMPLES, length(m_vec));

for m = m_vec    
    for i = 1:N_DKL_SAMPLES
        idx = randperm(M, m);
        % estimate parameters for the sample D_m
        [hat_wd, hat_wg, ~, ~, ~, ~] = gdl_par(y(idx), X(idx, :), LAMBDA, ALPHA);

        % obtain a sample of posterior parameters for the sample D_m
        [wd, wg] = MonteCarloParameters(y(idx), X(idx,:), LAMBDA, ALPHA, ...
                                                          N_W_SAMPLES);
        hat_w = [hat_wd(:); hat_wg(:)]';
        dist_to_opt = sort(pdist2([wd, wg], hat_w)/norm(hat_w));
        
        cov_by_d = (1:length(dist_to_opt))/length(dist_to_opt);
        idx_dist = cov_by_d == min(cov_by_d(cov_by_d > 1 - intercept));
        cov_length(i, m_vec == m) = dist_to_opt(idx_dist);
        coverage(i, m_vec == m) = mean(dist_to_opt <= conf_int);
        min_dist(i, m_vec == m) = min(dist_to_opt);
        
        invI = inv(X(idx, :)'*X(idx, :));
        Aopt(i, m_vec == m) = trace(invI);
        Dopt(i, m_vec == m) = det(invI);
        Gopt(i, m_vec == m) = max(diag(X(idx, :)*(invI)*X(idx, :)'));
        %save('res/dkl_dloo.mat', 'dkl', 'dkl_dloo'); 
    end
    %{                    
    [h(m_vec == m), p(m_vec == m), C(m_vec == m), chi2fit(m_vec == m)] = ...
                            infer_chi2_stats(dkl(:, m_vec == m)*2*m, ...
                            SIGNIFICANCE,...
                            ['tmp/dkl_hist_pdf_central_m', num2str(m)]);
    %}                   
end



scatterAndPlot(cov_length, {'x', m_vec, 'xlbl', 'Sample size',...
                                       'ylbl', 'Coverage length', ...
                                       'saveas', 'tmp/cov_length_m0'});
scatterAndPlot(coverage, {'x', m_vec, 'xlbl', 'Sample size',...
                            'ylbl', 'Coverage', 'avfunc', 'mean', ...
                            'saveas', 'tmp/coverage_wdispersion_m0'});
scatterAndPlot(Aopt, {'x', m_vec, 'xlbl', 'Sample size',...
                                       'ylbl', 'A-optimality', ...
                                       'saveas', 'tmp/aopt_m0'});
scatterAndPlot(Gopt, {'x', m_vec, 'xlbl', 'Sample size',...
                                       'ylbl', 'G-optimality', ...
                                       'saveas', 'tmp/gopt_m0'});
scatterAndPlot(Dopt, {'x', m_vec, 'xlbl', 'Sample size',...
                                       'ylbl', 'D-optimality', ...
                                       'saveas', 'tmp/dopt_m0'});
accepted_m = mean(coverage) > 1 - intercept;
m = min(m_vec(accepted_m));

figure;
hold on;
%semilogy(m_vec, median(cov_length), 'k-', 'linewidth', 2);
%plot(m_vec, median(cov_length), 'k-', 'linewidth', 2);
%plot(m_vec, cov_length, 'k.', 'linewidth', 2);
plot(m_vec, Gopt, 'k.', 'linewidth', 2);
plot(m_vec, median(Gopt), 'k-', 'linewidth', 2);
%plot(m_vec, mean(coverage), 'k-', 'linewidth', 2);
%plot(m_vec, coverage, 'k.', 'linewidth', 2);
xlabel('Sample size', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
ylabel('$\max$ diag. of $X^T(X^TX)^{-1}X$', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 20, 'FontName', 'Times')
axis tight;
hold off;

end


