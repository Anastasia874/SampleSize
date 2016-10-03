function [ss, h] = ParametricKLSSD(X, y, intercept)
% Parametric version, 02.09.2016
% Main idea: compare 2mKL to a given intercept since EXP(2mKL - n) tends to chi^2_n

N_DKL_SAMPLES = 50; % number of dkl samples
LAMBDA = 0.5; % trade-off between generative and discriminative
ALPHA = 0.01; % diagonal of the covariance matrix
N_W_SAMPLES = 5000; % number of MCMC samples of posterior parameters

M = length(y); % max sample size for dkl evaluation

m_vec = [20, 50:10:100, 150:50:500, 750, 1000];
m_vec = m_vec(m_vec <= M);

SIGNIFICANCE = 0.05;
%{
N_M_STEPS = 50;
m_step = round(M/2/N_M_STEPS);
if m_step < 1
    m_vec = M/2:M;
else
    m_vec = round(M/2):m_step:M;
end
if m_vec(1) > 100
    m_vec = [20, 50, 100, m_vec];
end
%}

h = zeros(length(m_vec),1);
p = zeros(length(m_vec),1);
C = zeros(length(m_vec),1);
chi2fit = zeros(length(m_vec),1);
dkl_dloo = zeros(N_DKL_SAMPLES, length(m_vec));
dkl = zeros(N_DKL_SAMPLES, length(m_vec));
test_errors = zeros(N_DKL_SAMPLES, length(m_vec));
train_errors = zeros(N_DKL_SAMPLES, length(m_vec));
%load('dkl_060916.mat');
for m = m_vec
    % obtain a sample of dkl for sample size m
    for i = 1:N_DKL_SAMPLES
        idx = randperm(M);
        idx1 = idx(1:m-1);
        idx2 = idx(1:m);
        % generate N_W_SAMPLES of w inside!     
        %{
        [dkl_dloo(i, m_vec == m)] = kl_divergence(y, X, [], N_W_SAMPLES, ...
                                                    LAMBDA, ALPHA, 'diff_loo', idx2);
        %}
        %
        [dkl(i, m_vec == m),~,post] = kl_divergence(y, X, [], N_W_SAMPLES, ...
                                                    LAMBDA, ALPHA, idx1, idx2);  
        [~, idx_max] = max(post.pw);
        w = post.w(idx_max, :);
        [test_errors(i, m_vec == m), train_errors(i, m_vec == m)] = ...
                                                   calc_loo_errors(y, X, ...
                                                   LAMBDA, ALPHA);
        %}
        %save('res/dkl_dloo.mat', 'dkl', 'dkl_dloo'); 
    end
    %{                    
    [h(m_vec == m), p(m_vec == m), C(m_vec == m), chi2fit(m_vec == m)] = ...
                            infer_chi2_stats(dkl(:, m_vec == m)*2*m, ...
                            SIGNIFICANCE,...
                            ['tmp/dkl_hist_pdf_central_m', num2str(m)]);
    %}                   
end

figure;
hold on;
semilogy(m_vec, median(dkl).*m_vec, 'k-', 'linewidth', 2);
%plot(m_vec, dkl_dloo, 'k--', 'linewidth', 2);
xlabel('Sample size', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
ylabel('$2m KL(m)$', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 20, 'FontName', 'Times')
axis tight;
hold off;
legend({'dkl', 'dkl_dloo'});

ss = 0;

end

function h = kltest(X, y, ss, nSamples, alpha)

N_BINS = 20;
for i = 1:nSamples
    idx1 = randi(length(y),ss,1);
    idx2 = randi(length(y),ss,1);
    s1 = X(idx1,:);
    s2 = X(idx2,:);
    stat = length(s1)*length(s2)*KLdiv(s1, s2, N_BINS)/(length(s1)+length(s2));
    chi1 = chi2inv(1 - alpha, 2*N_BINS);
    h = chi1 > stat & 0 < stat;
end

end

function dKL = KLdiv(X1, X2, nbins)

[N, xbars] = histcounts([X1; X2], nbins);
p1 = hist(X1, xbars)/length(X1);
p2 = hist(X2, xbars)/length(X2);

lstDKL = p1.*log(p1./p2);
idx = lstDKL == Inf | lstDKL == -Inf;
lstDKL = lstDKL.*(~idx) + idx*abs(max(lstDKL(~idx)));

dKL = nansum(lstDKL);

end