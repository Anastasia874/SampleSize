function D = generate_sample(m, n, model)

% Generates a sample of size m accorging to the sampling_model and returns
% structure D with fields y and X
% m - sample size
% n - dimensionality of X
% model - structure with fields:
%   data - data is a structure array itself. Each elemet corresponds to one
%   class. Fields: dist_name, prob, pars
%   parameters - another struct, specifies sampling prior distribution



n_classes = numel(model.data);
probs = [model.data().probs];
probs = probs/sum(probs);
sizes = ceil(m*probs);

X = [];
y = [];
for ncls = 1:n_classes
    data_distr = model.data(ncls);
    % create probability distribution object:
    pd = makedist(data_distr.dist_name);
    pars = fieldnames(data_distr.pars);
    for npar = 1:numel(pars)
        pd.(pars{npar}) = data_distr.pars.(pars{npar});
    end
    % generate random numbers from dist_name:
    X = [X; data_distr.scale*random(pd, [sizes(ncls), n])];
    y = [y; ncls*ones(sizes(ncls), 1)];
end

idx = randperm(m);
D.X = X(idx, :);
D.y = y(idx, :);

end