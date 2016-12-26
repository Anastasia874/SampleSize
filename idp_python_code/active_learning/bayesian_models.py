from __future__ import print_function

import random
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample


class GaussianPrior(object):

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


class BinBetaPrior(object):
    def __init__(self, prior_a=2, prior_b=2, data_covariance=None, targets_averaging="average",
                 n_parameters_to_sample=100, save_updates=True):
        self.prior_a = prior_a
        self.prior_b = prior_b

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
        self.posterior_a_ = None
        self.posterior_b_ = None
        self.w, self.p = None, None
        self.save_updates = None
        self.coefs = []
        self.ps = []

    def posterior_density(self, data, parameters):
        predictions = self.predict(data)
        if parameters is None:
            if self.posterior_parameters_sample is None:
                self.posterior_parameters_sample = []
                state = random.getstate()
                for x in data:
                    random.setstate(state)
                    self.posterior_parameters_sample.append(self.sample_posterior_params(x))
                self.posterior_parameters_sample = np.vstack(self.posterior_parameters_sample).T

            parameters = self.posterior_parameters_sample

        posterior_probs = np.zeros((len(parameters), len(data)))
        for i, y in enumerate(predictions):
            # posterior_a_ = self.posterior_a(y, self.posterior_a_)
            # posterior_b_ = self.posterior_a(y, self.posterior_b_)
            posterior_probs[:, i] = self.posterior_density_by_object(parameters[:, i])  #, posterior_a_, posterior_b_)

        return posterior_probs

    # def posterior_density_target_averaging(self, parameters):
    #     for cls in self.classes:
    #         posterior_a_ = self.posterior_a(cls, self.posterior_a_)
    #         posterior_b_ = self.posterior_a(cls, self.posterior_b_)
    #         posterior_probs[:, i] = self.posterior_density_by_object(parameters)

    def posterior_density_by_object(self, parameters, posterior_a_=None, posterior_b_=None):
        if posterior_a_ is None:
            posterior_a_ = self.posterior_a_
        if posterior_b_ is None:
            posterior_b_ = self.posterior_b_
        return stats.beta.pdf(parameters, posterior_a_, posterior_b_)

    def posterior_a(self, data, a):
        return a + np.sum(data)

    def posterior_b(self, data, b):
        return b + len(data) - np.sum(data)

    def prior_density(self, parameters=None):
        if parameters is None:
            if self.posterior_parameters_sample is None:
                raise AttributeError("Error in prior_density: self.posterior_parameters_sample in None")
            parameters = self.posterior_parameters_sample
        return stats.beta.pdf(parameters, self.prior_a, self.prior_b)

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.w, self.p = self.optimal_w(X, y)
        self.posterior_a_ = self.posterior_a(y, self.prior_a)
        self.posterior_b_ = self.posterior_b(y, self.prior_b)

        if self.save_updates:
            self.coefs.append(self.w)
            self.ps.append(self.p)

    def optimal_w(self, X, y):
        lr = LogisticRegression().fit(X, y)
        return lr.coef_[0], lr.intercept_[0]

    def sample_posterior_params(self, x, n_samples=None):
        # Since we only know the distribution of wX | X, y, we need to bootstrap (X, y) obtain the sample of w
        n_samples = n_samples or self.n_parameters_to_sample
        posterior_sample = []

        mdl = LogisticRegression().fit(np.vstack((self.X, x)), np.hstack((self.y, self.predict(x))))
        for n in range(n_samples):
            bX, by = resample(self.X, self.y)
            if len(np.unique(by)) < len(self.classes):
                bX, by = self.resample_with_noise()
            predictions = self.predict(bX, w=mdl.coef_[0], p=mdl.intercept_[0])
            posterior_sample.append(np.mean(predictions))

        # return stats.beta.rvs(size=n_samples, a=self.posterior_a_, b=self.posterior_b_)
        return np.hstack(posterior_sample)

    def resample_with_noise(self, noise_lvl=0.07):
        std = np.std(self.X)
        noise = np.random.randn(self.X.shape[0], self.X.shape[1]) * std * noise_lvl

        return self.X + noise, self.y

    def decision_function(self, X, w=None, p=None):
        if w is None:
            w = self.w
        if p is None:
            p = self.p
        return p + np.dot(w,  X.T)

    def bernoulli_p(self):
        return float(self.posterior_a_) / float(self.posterior_a_ + self.posterior_b_)

    def predict(self, X, w=None, p=None):
        # import matplotlib.pyplot as plt
        # lr = LogisticRegression().fit(self.X, self.y)
        # plt.plot(self.predict_proba(X))
        # plt.plot(lr.predict_proba(X))
        # plt.show()
        return self.predict_proba(X, w=w, p=p) > 0.5  # self.bernoulli_p()

    def predict_proba(self, X, w=None, p=None):
        return 1 / (1 + np.exp(-self.decision_function(X, w=w, p=p)))


def adjust_covariance_dimensions(covariance, n_feats):
    if n_feats == 1:
        return covariance, 1.0 / covariance

    if len(covariance) == n_feats and covariance.ndim < 2:
        covariance = np.diag(covariance)
    elif len(covariance) == 1:
        covariance = np.diag(np.tile(covariance, n_feats))

    return covariance, np.linalg.inv(covariance)