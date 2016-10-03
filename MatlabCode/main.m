function main
addpath(fullfile(pwd,'../../mlalgorithms/','Group874/Motrenko2014KL/code'));
addpath(fullfile(pwd,'../../mlalgorithms/PhDThesis/Motrenko/code/','KLdivergence'));
addpath(fullfile(pwd,'../../mlalgorithms/PhDThesis/Motrenko/code/','GenerativeDiscriminative'));
addpath(fullfile(pwd,'../../mlalgorithms/PhDThesis/Motrenko/code/','plot'));
addpath(fullfile(pwd,'../../mlalgorithms/PhDThesis/Motrenko/code/','SampleSize'));
addpath(fullfile(pwd,'../../','MATLAB/grinsted-gwmcmc-9e7a97b'));

% av_posterior experiment:
% define a range of SS values
m_vec = [20, 50, 100, 200, 1000];

% assign quality criteria
criteria = {@posterior_KL_divergence};%, @coverage_probability, @ADG_optimality};

% define data likelihood
pars1.mu = -1;
pars2.a = 5;
pars2.b = 5;
pars3.a = 10;
scale = {1, 1, 1}; % FIXIT
sampling_model.data = struct('dist_name', {'Normal', 'Gamma', 'Gamma'},...
    'probs', {0.5, 0.5, 1}, 'pars', {pars1, pars2, pars3}, 'scale', scale);

% define model
fitting_model = sampling_model;
fitting_model.prior{1} = struct('mu', []);
fitting_model.prior{1}.mu = {'Normal'};
fitting_model.prior{2} = struct('a', [], 'b', []);
fitting_model.prior{2}.a = {'Normal', pars1};
fitting_model.prior{2}.b = {'Normal'};
fitting_model.prior{3} = struct('a', []);
fitting_model.prior{3}.a = {'Normal'};
av_criteria = cell(1, length(m_vec));

% Experiment settings:
nD_samples = 4; % number of samples to generate at each m
nW_samples = 500; % number os posterior parameter samples for T(D_m) computation

%--------------------------------------------------------------------------
% Main loop:
for m = m_vec
    disp(['Sample size: ', num2str(m)])
    % returns a cell array [nD_samples x nCriteria], each cell contains a
    % structure, nFileds = num ob dimensions of the corresponding criteria,
    % fileds store ([1 x nW] or [1]) results of T(D_m; w) computations
    av_criteria{m_vec == m} = av_posterior_sampling(m, sampling_model, ...
                                fitting_model, criteria, ...
                                nD_samples, nW_samples);    
end

%--------------------------------------------------------------------------
% Other experiments
mu = [0, 1]';
sigma2 = 1;
theta = 0.7;
m = 5000;

D = testData(mu, sigma2, theta, m);

CONFIDENCE_INTERVAL = 0.1;
SIGNIFICANCE = 0.05;

average_posterior_ssd(CONFIDENCE_INTERVAL, SIGNIFICANCE)
average_coverage_ssd(D.y, D.X, CONFIDENCE_INTERVAL, SIGNIFICANCE);
ParametricKLSSD(D.X, D.y, SIGNIFICANCE);

%{
nSamples = 50;
pars = initPars('Wald', 0.05, 0.2, 0, nSamples, {0.1});
m = testSS_err(pars);
%}
%m = testSS_distPars(pars);
%[m, h, ss] = KLSampleSize(X, y, pars);
%{
for i = 1:length(ss)
    ge(i) = calcError(X, y, ss(i));
end
%}
%m = testSS_nSamples(X, y, mu, sigma2, theta);

end

function m = testSS_distPars(pars)
parsEq = pars;
parsEq.test = 'Prob';
parsEq.prob = 0;
parsEq.more = {'Sup', 0.1};
mu0 = 0;
theta = 0.7;
nIterM = 0;
for mu1 = 0:0.5:10
    nIterM = nIterM + 1;
    nIterS = 0;
    for sq_sigma = 0.1:0.1:2
        nIterS = nIterS + 1;
        mu = [mu1, mu0]';
        sigma2 = sq_sigma^2;
        D = testData(mu, sigma2, theta, 1000);
        X = D.X;
        y = D.y;
        %m.MR(nIterM, nIterS) = MRsampleSize(X, y, pars);
        %m.Wald(nIterM, nIterS) = WaldSampleSize(X, y, pars);
        try
        [ss, p, ss_vec] = KLSampleSize(X, y, pars);
        m.KL(nIterM, nIterS) = ss;
        catch
            disp('mKL')
            [~, idx] = min(abs(1 - p - pars.alpha));
            m.KL(nIterM, nIterS) = ss_vec(idx);
        end
        %m.Eq(nIterM, nIterS) = sampleSize(X, y, parsEq);
        %m.Sup(nIterM, nIterS) = sampleSize(X, y, parsEq);
    end 
end    
end

function m = testSS_err(pars)
parsEq = pars;
parsEq.test = 'Prob';
parsEq.prob = 0;
parsEq.more = {'Sup', 0.1};
mu = [1, 0]';
sigma2 = 1;
theta = 0.7;
nIter = 0;
for nSamples = 200:20:1000
        nIter = nIter + 1;
        pars.nSamples = nSamples;
        D = testData(mu, sigma2, theta, nSamples);
        X = D.X;
        y = D.y;
        m.KL(nIter) = KLSampleSize(X, y, pars);
        m.MR(nIter) = MRsampleSize(X, y, pars);
        m.Wald(nIter) = WaldSampleSize(X, y, pars);
       err(1, nIter) = calcError(X, y, round(m.KL(nIter)));
       err(2, nIter) = calcError(X, y, round(m.MR(nIter)));
       err(3, nIter) = calcError(X, y, round(m.Wald(nIter)));
       save('err_m.mat','err');
       save('m.mat','m');
end

%{
if length(s_vec) ~= length(all_s)
    count = repmat(s_vec', length(all_s), 1) == ...
        repmat(all_s, 1, length(s_vec));
    count = sum(count)';
end

m.KL = m.KL/nIterM;
err = err/nIterM;
%}
end

function m = testSS_nSamples(X, y, mu, sigma2, theta)
m0 = 50;
prob = 0.5;
nSamples = 50;
%
nIter = 1;
for nSamples = m0:5:100
    nInnerIt = 1;
    pars = initPars('Wald', 0.1, 0.2, 0, nSamples, {0.1});
    m.MR(nIter) = MRsampleSize(X, y, pars);
    m.Wald(nIter) = WaldSampleSize(X, y, pars);
    %for prob = 0.5:0.01:0.8
    pars = initPars('Prob', 0.1, 0.2, prob, nSamples, {'Eq', 0.01});
    %m.KL = KLsampleSize(X, y, pars);
    m.Eq(nIter) = sampleSize(X, y, pars);
    pars = initPars('Prob', 0.1, 0.2, prob, nSamples, {'Sup', 0.1});
    m.Sup(nIter) = sampleSize(X, y, pars);
    %nInnerIt = nInnerIt + 1;
    %end
    nIter = nIter + 1;
end
%}
%m.MR = sampleSize(X, y, pars);
%m.boot = sampleSize(X, y, pars);
%m(end )
plotResults(m, pars, mu, sigma2, theta);

end

function pars = initPars(test, alpha, beta, H0, nSamples, others)

pars.test = test;
pars.alpha = alpha;
pars.beta = beta;
pars.H0 = H0;
pars.nSamples = nSamples;

if nargin > 5
    pars.more = others;
else
    pars.more = [];
end
end