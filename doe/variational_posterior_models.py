from __future__ import print_function
import numpy as np
from scipy import linalg
from sklearn.mixture import BayesianGaussianMixture
from scipy import stats


class VariationalRegression(object):
    """
    Discriminative likelihood: p(y|w) = Normal(w*X, beta^{-1}),
    Parameter prior: p(w|\alpha) = Normal(0, alpha^{-1})
    Hyperparameter prior: p(\alpha) = Gamma(a_0, b_0)
    """

    def __init__(self, prior_covariance=np.ones(1), data_covariance=None, targets_averaging="optimal",
                 n_parameters_to_sample=500, kernels=None, n_components=10):
        self.prior_mean = np.zeros(1)
        self.prior_covariance = prior_covariance

        # self.data_mean = data_mean
        self.data_covariance = data_covariance  # beta^{-1}
        self.inv_prior_cov = None
        self.inv_data_cov = None  # beta
        self.type = None
        self.targets_averaging = targets_averaging
        self.n_parameters_to_sample = n_parameters_to_sample
        self.gamma_a = 2
        self.gamma_b = 2


        self.n_feats = None
        self.X = None
        self.y = None
        self.target_range = None
        self.classes = None
        self.posterior_mean_ = None
        self.posterior_covariance_ = None
        self.problem_type = None
        self.posterior_parameters_sample = None
        self.kernels = kernels  # optional
        self.w = None

        self.model = BayesianGaussianMixture(n_components=n_components)


    def likelihood(self, X, y):
        likelihoods = self.model()

        return np.prod(likelihoods)

    def fit(self, X, y):
        self.model.fit(X)






