from __future__ import print_function

import numpy as np
from scipy import stats


class PosteriorDistribution(object):

    def __init__(self, prior_covariance=np.ones(1), data_covariance=None, targets_averaging="optimal",
                 n_parameters_to_sample=500):
        self.prior_mean = np.zeros(1)
        self.prior_covariance = prior_covariance

        # self.data_mean = data_mean
        self.data_covariance = data_covariance
        self.inv_prior_cov = None
        self.inv_data_cov = None
        self.type = None
        self.targets_averaging = targets_averaging
        self.n_parameters_to_sample = n_parameters_to_sample

        self.n_feats = None
        self.X = None
        self.y = None
        self.target_range = None
        self.classes = None
        self.posterior_mean_ = None
        self.posterior_covariance_ = None
        self.problem_type = None
        self.posterior_parameters_sample = None

    def update_posterior_density(self, parameters, x):
        data = np.vstack((self.X, x))
        posterior_covariance = self.posterior_covariance(data)
        if self.targets_averaging.lower() == "optimal":
            targets = np.hstack((self.y, self.predict(x)))
            posterior_mean = self.posterior_mean(data, targets, posterior_covariance)
            return self.posterior_density_by_object(parameters, posterior_mean, posterior_covariance)

        if self.targets_averaging.lower() == "average":
            return self.posterior_density_target_averaging(parameters, posterior_covariance, data)

        raise ValueError("targets_averaging should be either 'optimal' or 'average', got {}"
                         .format(self.targets_averaging))

    def posterior_density_target_averaging(self, parameters, posterior_covariance, data):
        posterior_probs_by_y = np.zeros((len(self.target_range), len(parameters)))
        for i, y in enumerate(self.target_range):
            targets = np.hstack((self.y, y))
            posterior_mean = self.posterior_mean(data, targets, posterior_covariance)
            posterior_probs_by_y[i, :] = self.posterior_density_by_object(parameters, posterior_mean,
                                                                          posterior_covariance)
        return np.mean(posterior_probs_by_y, axis=0)

    def posterior_mean(self, *args, **kwargs):
        return self.prior_mean

    def posterior_covariance(self, *args, **kwargs):
        return self.prior_covariance

    def posterior_density_by_object(self, *args, **kwargs):
        raise AttributeError("posterior density must be overridden")

    def predict(self, x):
        pass


class GaussianPrior(PosteriorDistribution):

    def __init__(self, prior_covariance=np.ones(1), data_covariance=None, targets_averaging="optimal",
                 n_parameters_to_sample=500):
        PosteriorDistribution.__init__(self, prior_covariance, data_covariance,
                                       targets_averaging, n_parameters_to_sample)
        self.posterior_alpha = None
        self.posterior_beta = None

    def fit(self, X, y):
        if self.inv_prior_cov is None:
            self.init_model_pars(X, y)

        self.X = X
        self.y = y
        self.posterior_covariance_ = self.posterior_covariance(X)
        self.posterior_mean_ = self.posterior_mean(X, y, self.posterior_covariance_)

    def init_model_pars(self, X, y):
        if self.type == "regression":
            self.target_range = np.unique(y)
        else:
            self.target_range = self.classes
        # if self.data_mean is None:
        #     self.data_mean = np.mean(self.X, axis=1)
        self.n_feats = X.shape[1]
        self.prior_mean = np.tile(self.prior_mean, self.n_feats)

        self.data_covariance = self.data_covariance or self.prior_covariance
        self.prior_covariance, self.inv_prior_cov = adjust_covariance_dimensions(self.prior_covariance, self.n_feats)
        self.data_covariance, self.inv_data_cov = adjust_covariance_dimensions(self.prior_covariance, self.n_feats)

    def posterior_density_by_object(self, parameters, posterior_mean, posterior_cov):
        return stats.multivariate_normal.pdf(parameters, mean=posterior_mean, cov=posterior_cov)

    def posterior_density(self, data, parameters=None):
        if parameters is None:
            if self.posterior_parameters_sample is None:
                self.posterior_parameters_sample = self.sample_posterior_params()
            parameters = self.posterior_parameters_sample
        posterior_probs = np.zeros((len(parameters), len(data)))
        for i, x in enumerate(data):
            x = x[None, :]
            posterior_probs[:, i] = self.update_posterior_density(parameters, x)

        return posterior_probs

    def posterior_covariance(self, X):
        return np.linalg.inv(np.dot(self.inv_data_cov, np.dot(X.T, X)) + self.inv_prior_cov)

    def posterior_mean(self, X, y, post_cov=None):
        if post_cov is None:
            post_cov = self.posterior_covariance(X)
        return np.dot(np.dot(self.inv_prior_cov, post_cov),  np.dot(X.T, y))

    def prior_density(self, parameters):
        if parameters is None:
            if self.posterior_parameters_sample is None:
                self.posterior_parameters_sample = self.sample_posterior_params()
            parameters = self.posterior_parameters_sample
        return stats.multivariate_normal.pdf(parameters, mean=self.prior_mean, cov=self.prior_covariance)

    def sample_posterior_params(self, n_samples=None):
        n_samples = n_samples or self.n_parameters_to_sample
        return stats.multivariate_normal.rvs(size=n_samples, mean=self.posterior_mean_, cov=self.posterior_covariance_)

    def decision_function(self, X):
        return np.sum(self.posterior_mean_ * X, axis=1)

    def predict(self, X):
        if X.ndim == 1:
            raise ValueError("Error in GaussPrior.predict. Data should have 2 dimensions, got one.")
        predictions = self.decision_function(X)
        if not self.problem_type == "classification":
            return predictions

        y = np.zeros(len(X))
        y[predictions > 0] = np.ones(len(X))[predictions > 0]
        return y


class BinBetaPrior(PosteriorDistribution):

    def __init__(self, prior_covariance=np.ones(1), data_covariance=None, targets_averaging="average",
                 n_parameters_to_sample=500):
        PosteriorDistribution.__init__(self, prior_covariance, data_covariance,
                                       targets_averaging, n_parameters_to_sample)

    def posterior_density_by_object(self, parameters, posterior_mean, posterior_cov):
        pass


def adjust_covariance_dimensions(covariance, n_feats):
    if n_feats == 1:
        return covariance, 1.0 / covariance

    if len(covariance) == n_feats and covariance.ndim < 2:
        covariance = np.diag(covariance)
    elif len(covariance) == 1:
        covariance = np.diag(np.tile(covariance, n_feats))

    return covariance, np.linalg.inv(covariance)