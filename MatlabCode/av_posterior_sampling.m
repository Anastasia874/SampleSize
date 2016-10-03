function av_criteria = av_posterior_sampling(m, sampling_model, ...
                                fitting_model, criteria, ...
                                nD_samples, nW_samples)
                            
% Compute average values of the SSD criterion
% m = the tested sample size, scalar
% sampling_model - structure, specifies data likelihood and prior on the
% model parameters used for data sampling (true distribution)
% fitting_model - structure, specifies data likelihood and prior on the
% model parameters, used for modelling data (hypothesed distribution)
% criteria - cell array of handles/names of the SSD criteria 

N_AV_SAMPLES = 1000;
N_W_SAMPLES = 10000;
TRAIN_TEST_RATIO = 0.25;

if ~exist('nD_samples', 'var')
    nD_samples = N_AV_SAMPLES;
end
if ~exist('nW_samples', 'var')
    nW_samples = N_W_SAMPLES;
end

%m = 500;
n = 1;



n_criteria = numel(criteria);
sampled_stats = cell(nW_samples, n_criteria + 2); % + 2 for train and test errors
for i = 1:nD_samples
    % returns a structure, D.X [m x n], D.y [m x 1]
    m_train_test = round(m*(1 + TRAIN_TEST_RATIO));
    D = generate_sample(m_train_test, n, sampling_model);
    X = D.X(1:m, :);
    y = D.y(1:m);
    %plotCls(D.X, D.y, {'res', D.y, 'legend', {sampling_model.data().dist_name} });
    % returns a matrix W [N_W_SAMPLES x n] of the posterior parameters
    [W, data_pd, prior_pd, parnames] = posterior_pars(y, X, fitting_model, nW_samples);
    for j = 1:n_criteria
        sampled_stats{i, j} = feval(criteria{j}, y, X, W, data_pd, ...
                                                prior_pd, parnames);
    end
    % train errors:
    [sampled_stats{i, n_criteria + 1}, ~, probs] = prediction_errors(y, X, W, data_pd, ...
                                                        prior_pd, parnames);
    % test errors:
    sampled_stats{i, n_criteria + 2} = prediction_errors(D.y(m+1:end), ....
                           D.X(m+1:end,:), W, data_pd, prior_pd, parnames, ...
                           probs, unique(y));
end

%av_criteria = mean(sampled_stats);
av_criteria = sampled_stats;

end