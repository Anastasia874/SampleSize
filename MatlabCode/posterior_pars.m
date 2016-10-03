function [W, data_pd, prior_pd, parnames] = posterior_pars(y, X, model, nSamples)
% fnction runs cascaded affine invariant ensemble MCMC sampler ("The MCMC hammer")
% implementated by Aslak Grinsted 
% http://www.mathworks.com/matlabcentral/fileexchange/49820-ensemble-mcmc-sampler

addpath(fullfile(pwd,'../../..','MATLAB/grinsted-gwmcmc-9e7a97b'));


% Fit the distrbutions for each class:
n_classes = numel(model.data);
data_pd = cell(1, n_classes);
probs = zeros(1, n_classes);
for ncls = 1:n_classes
    probs(ncls) = mean(y == ncls);
    data_distr = model.data(ncls);
    data_pd(ncls) = fit_distr(X(y==ncls, :), data_distr.dist_name);   
end


% Create probability distribution objects for priors
numW = zeros(1, n_classes);
prior_pd = cell(0);
parnames = cell(1, n_classes);
for ncls = 1:n_classes
    prior_distr = model.prior{ncls};
    % get names of parameters, for which priors are set
    pars = fieldnames(prior_distr);
    numW(ncls) = numel(pars);   
    parnames{ncls} = pars;
    % for each priored par create pd 
    for npar = 1:numel(pars)
        dist_name = prior_distr.(pars{npar}){1};
        % create pd:
        prior_pd{end+1} = makedist(dist_name);
        % if necessary, set pd parameters
        if numel(prior_distr.(pars{npar})) < 2
            % (if no pars specified, MATLAB defaults will be used)
            continue
        end
        dist_pars = prior_distr.(pars{npar}){2};
        dist_pars_names = fieldnames(dist_pars);
        for i = 1:numel(dist_pars_names)
        prior_pd{end}.(dist_pars_names{i}) = ...
                            dist_pars.(dist_pars_names{i});
        end
                  
    end
end

numW = sum(numW);  % number of "priored" parameters 
nWalkers = 2*numW; % number of walkers in MCMC

%{
% test logprior and loglike:
w = randn(1, numW);
lpr = logprior(w, prior_pd);
lhd = loglike(y, X, probs, data_pd, w, parnames);
%}

% Assign loglikelihood and logprior density functions
logPfuns =  {@(w)logprior(w, prior_pd), ...
             @(w)loglike_func(y, X, probs, data_pd, w, parnames)};
         
minit=rand(numW, nWalkers); % a set of starting points for the entire ensemble of walkers

% Run ensemble MCMC sampler
try_again = true;
while try_again
[parameter_walks, try_again] = gwmcmc(minit,logPfuns, nSamples);
end
W = parameter_walks(:,:);

% Plot evolution of simulated parameters:
lpr = zeros(size(W, 2), 1);
lhd = zeros(size(W, 2), 1);
for i = 1:size(W, 2)
    lpr(i) = logprior(W(:, i)', prior_pd);
    lhd(i) = loglike_func(y, X, probs, data_pd, W(:, i)', parnames);
end
%myPlot2Y(lpr, lhd, {'ylbl1', 'Prior', 'ylbl2', 'Data', 'xlbl', 'N. iterations'});

ndrop = round(size(W, 2)/5);
W = W(:, ndrop + 1:end);
%plotWPairs(wd(:,1:2),wg(:,1:2));

end

function  pd = fit_distr(data, dist_name)

n = size(data, 2);
pd = cell(1, n);
for i = 1:n
   % Use later!
   % for scale estimation, use mle with custom-defined pdf:
   %     pars =  mle(x,'pdf',@(x,v,s)chi2pdf(x/s,v)/s,'start',[1,200])
   % will return pars = [v, s]  
   pd{i} = fitdist(data(:, i), dist_name);
end

end


