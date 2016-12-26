from __future__ import print_function

import numpy as np
import bayesian_models
from base_learner import BaseLearner


class RandomSampling(BaseLearner):

    def __init__(self, dataset_, model=None, rebuild_model_at_each_iter=True, name=""):
        BaseLearner.__init__(self, dataset_, model, rebuild_model_at_each_iter, "Random"+name)
        self.description = "RandomSampling selects new samples at random at each step"

    def query_function(self, n_samples):
        return self.get_random_unlabeled_ids(n_samples)


class LeastConfidentSampling(BaseLearner):
    def __init__(self, dataset_, model=None, rebuild_model_at_each_iter=True, name=""):
        BaseLearner.__init__(self, dataset_, model, rebuild_model_at_each_iter, "LeastConfident"+name)
        if self.dataset.type == "classification" and len(self.dataset.classes) > 2:
            raise Exception('Least confident sampling is only defined for 2-class problems, this one has {} classes'.
                            format(len(self.dataset.classes)))
        self.description = "LeastConfidentSampling chooses those samples from the pool, " \
                           "where the model is least confident"
        

    def query_function(self, n_samples):
        conf = self.model_confidence(self.dataset.unlabeled_data)
        if conf.ndim > 1:
            conf = np.max(conf, axis=1)
        else:
            conf = np.abs(0.5 - conf)
        idx_to_label = np.argsort(conf)[:n_samples]
        # if n_samples == 1:
        #     return [self.dataset.unlabeled_idx[idx_to_label]]

        return self.dataset.unlabeled_idx[idx_to_label]


class MaxEntropySampling(BaseLearner):
    def __init__(self, dataset_, model=None, rebuild_model_at_each_iter=True, name=""):
        BaseLearner.__init__(self, dataset_, model, rebuild_model_at_each_iter, "MaxEntropy"+name)
        self.description = "MaxEntropySampling chooses those samples from the pool, " \
                           "where the model the entropy of p(y|x) is the highest"

    def query_function(self, n_samples):
        probs = self.model_confidence(self.dataset.unlabeled_data)

        if probs.ndim == 1:
            probs /= np.sum(probs)  # normalize probs to [0, 1]
            probs = np.vstack((probs, 1-probs)).T
        else:
            probs /= np.sum(probs, axis=1)[:, None] # normalize probs to [0, 1]
        entropy = -np.sum(probs*np.log(probs), axis=1)

        idx_to_label = np.argsort(entropy)[-n_samples:]

        # if n_samples == 1:
        #     return [self.dataset.unlabeled_idx[idx_to_label]]

        return self.dataset.unlabeled_idx[idx_to_label]


class LindleyInformation(BaseLearner):
    def __init__(self, dataset_, model=None, rebuild_model_at_each_iter=True, update_parameters_sample=True,
                 min_prob=1e-15, name=""):
        if model is None:
            model = bayesian_models.GaussianPrior()
        BaseLearner.__init__(self, dataset_, model, rebuild_model_at_each_iter, "LindleyInformation"+name)
        self.description = "LindleyInformation chooses those samples from the pool, " \
                           "which provide maximum expected information gain"
        self.model.posterior_parameters_sample = None
        self.update_parameters_sample = update_parameters_sample
        self.min_prob = min_prob

    def query_function(self, n_samples):
        if not hasattr(self.model, "posterior_density"):
            raise AttributeError("The model must provide posterior density estimation")

        posterior_probs = self.model.posterior_density(self.dataset.unlabeled_data,
                                                       self.model.posterior_parameters_sample)
        prior_probs = self.model.prior_density(self.model.posterior_parameters_sample)
        posterior_probs = _clip_zeros(posterior_probs, self.min_prob)
        prior_probs = _clip_zeros(prior_probs, self.min_prob)
        if self.update_parameters_sample:
            self.model.posterior_parameters_sample = None
        if prior_probs.ndim == 1:
            prior_probs = np.tile(prior_probs[:, None], (1, posterior_probs.shape[1]))
        information_gain = np.sum(posterior_probs * (np.log(posterior_probs) - np.log(prior_probs)), axis=0)

        idx_to_label = np.argsort(information_gain)[-n_samples:]
        return self.dataset.unlabeled_idx[idx_to_label]


def _clip_zeros(x, min_value):
    x[x < min_value] = min_value
    return x