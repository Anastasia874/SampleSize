from __future__ import division
from __future__ import print_function

import copy
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import normalize
from sklearn.datasets import make_classification
from collections import namedtuple

probDistrj = namedtuple("probDistrj", "pd pars idx")
probDistrList = namedtuple("probDistrList", "pd_list name")
probDistrXy = namedtuple("probDistrXy", "probs distr")

MIN_LHD = 1e-18
MAX_LHD = 1e18

def sample_from_prob_distrj(n_samples, distr):
    # distr is an instance  probDistrj

    frozen_distr = distr.pd(**distr.pars)
    z = frozen_distr.rvs(size=int(n_samples))

    return z

class ProbDistr():

    def __init__(self, n_feats=1):
        self.n_feats = n_feats

    def create_distr_by_name(self, distr_name, n_dims=None, pars={}):
        """
        creates probability distribution to sample from

        :param distr_name: Distribution name (uniform, gamma, exponential, normal). If name does not belong to this list, multivariate normal distribution will be created
        If list, the function will be called recursively
        :type distr_name: string
        :param n_dims: number of features to generate
        :type n_dims: int
        :param pars: parameters, recognized by the corresponding distribution
        :type pars: dict
        :return: namedtuple which stores a list of scipy.generators in .pd_list field
        :rtype: probDistrList
        """
        if n_dims is None:
            n_dims = self.n_feats
            

        if isinstance(distr_name, list):
            # If several names are passed as a list, return a list pf PDs
            distr = [0] * len(distr_name)
            for i, d_name in enumerate(distr_name):
                distr[i] = self.create_distr_by_name(d_name, pars={"loc":i})

            return distr

        name = distr_name
        if distr_name == 'uniform':
            pd_list = [probDistrj(stats.uniform, pars, []) for i in range(n_dims)]
        elif distr_name == 'exp':
            pd_list = [probDistrj(stats.expon, pars, []) for i in range(n_dims)]
        elif distr_name == "gamma":
            pd_list = [probDistrj(stats.gamma, pars, []) for i in range(n_dims)]
        elif distr_name == "normal" or distr_name == "norm":
            pd_list = [probDistrj(stats.norm, pars, []) for i in range(n_dims)]
        else:
            if distr_name is None:
                print("Feature distribution is not specified! Using multivariate normal")
            else:
                print("Feature distribution {} is not supported. Using multivariate normal".format(distr_name))
            pd_list = [probDistrj(stats.multivariate_normal, {"mean":range(n_dims), "cov":1}, [])]
            name = "Multivariate normal"

        distr = probDistrList(pd_list, name)

        return distr



    def check_distr(self, distr, probs):
        """
        Checks that the distribution passed to class constructor is adequate

        :param distr: PD-like object, used to generate data within classes
        :type distr: probDistXy or list
        :param probs: prior probabilities of each class
        :type probs: list or ndarray
        :return: corrected probability distribution
        :rtype: probDistXy
        """

        if isinstance(distr, probDistrXy):
            return distr

        if isinstance(distr, list):
            for d in distr:
                if not isinstance(d, probDistrList):
                    print("Type of prob. distr is not supported: received {}, expected probDistrList".format(type(d)))
                    raise ValueError
            distr = probDistrXy(probs, distr)


        return distr

    def sample(self, n_samples, distr, return_idx=False):
        """
        Samples data from the specified distribution

        :param n_samples: number of samples to draw from distr
        :type n_samples: int
        :param distr: contain a number of distribution (factorization of the whole distribution)
        :type distr: probDistrList
        :param return_idx: if True, return list of indices of generated features in matrix X (one list per element of distr.pd_list)
        :type return_idx: bool
        :return: generated data matrix
        :rtype: ndarray
        """

        n_distr = len(distr.pd_list)
        z = [0] * n_distr
        idx = [0] * n_distr
        last_idx = -1
        for n in range(n_distr):
            z[n] = sample_from_prob_distrj(n_samples, distr.pd_list[n])
            if z[n].ndim == 1:
                z[n] = z[n][:, None]
            n_zfeats = z[n].shape[1]
            idx[n] = range(last_idx + 1, last_idx + 1 + n_zfeats)
            last_idx += n_zfeats

        z = np.hstack(z)

        if return_idx:
            return z, idx

        return z



class GenerativeDistribution(ProbDistr):

    def __init__(self, n_feats=1, n_cls=2, x_distr=None, x_distr_name=None, w_distr=None, probs=None):
        ProbDistr.__init__(self, n_feats)


        if x_distr_name is None and x_distr is None:
            print("Must specify either x_distr or x_distr_name for each class")
            raise ValueError

        self.probs = probs
        if self.probs is None and not n_cls is None:
            self.probs = np.ones(n_cls) / n_cls
        self.n_cls = len(self.probs)

        if not x_distr is None:
            self.x_distr = self.check_distr(x_distr, self.probs)
        else:
            self.x_distr = probDistrXy(self.probs, self.create_distr_by_name(x_distr_name))




    def sample_data(self, n_samples):
        """
        Make a sample from the probability distribution probDistrXy (self.x_distr)

        :param n_samples: sample size
        :type n_samples: int
        :return: vector of targets and feature matrix
        :rtype: ndarray, ndarray
        """

        distr = self.x_distr


        n_samples_per_cls = np.ceil(n_samples * np.array(distr.probs))
        y, X = [], []
        for cls, cls_n_samples in enumerate(n_samples_per_cls):

            y.append(np.ones(cls_n_samples) * cls)
            Xcls, idx = self.sample(cls_n_samples, distr.distr[cls], return_idx=True)
            for i in range(len(distr.distr[cls].pd_list)):
                distr.distr[cls].pd_list[i] = probDistrj(distr.distr[cls].pd_list[i].pd,
                                                         distr.distr[cls].pd_list[i].pars, idx[i])

            if Xcls.ndim == 1:
                Xcls = Xcls[:, None]

            X.append(Xcls)

        idx = range(int(sum(n_samples_per_cls)))
        np.random.shuffle(idx)
        idx = idx[:n_samples]
        y = np.hstack(y)[idx]
        X = np.vstack(X)[idx, :]

        self.x_distr = distr

        return y, X

    def make_y_labels(self, X=None, lh=None, x_distr=None):
        """
        Classifies the sample X. Label maximize data likelihood, based on specified probability distribution

        :param X: sample to classify
        :type X: ndarray
        :param lh: likelihoods, shape = (n_samples, n_cls)
        :type lh: ndarray
        :param x_distr: probability distribution object
        :type x_distr: probDistrXy
        :return: vector of most probable class labels
        :rtype: ndarray
        """

        if lh is None:
            lh = self.likelihood(X, x_distr)

        y = np.argmax(lh, axis=1)

        return y


    def likelihood(self, X, x_distr=None, y=None):
        """
        Compute likelihood of X under x_distr

        :param X: Data sample
        :type X: ndarray
        :param x_distr: each element of probDistrXy contains PD object, which factorizes probability distribution p(X|y) for one class
        :type x_distr: probDistrXy
        :param y: optional. If y is not specified, compute likelihood for all classes and return (n_samples, n_cls) matrix
        :type y: ndarray
        :return: likelihoods, (n_samples, n_cls) or (n_samples, )
        :rtype: ndarray
        """

        if x_distr is None:
            x_distr = self.x_distr

        lh = np.zeros(X.shape[0])
        lh_y_none = np.zeros((X.shape[0], len(x_distr.distr)))
        for cls, distr in enumerate(x_distr.distr):
            lh_by_class = np.zeros((X.shape[0], len(distr.pd_list)))

            for j, xdj in enumerate(distr.pd_list):
                frozen_distr = xdj.pd(**xdj.pars)
                lh_by_class[:, j] = np.squeeze(frozen_distr.pdf(X[:, xdj.idx]))

            lh_y_none[:, cls] = np.prod(lh_by_class, axis=1)

            if not y is None:
                lh[y == cls] = lh_y_none[y == cls, cls]

        if y is None:
            return lh_y_none

        return lh


    def fit_distr_pars(self, y, X, replace=True, method="Nelder-Mead"):




        probs = np.ones_like(self.probs)
        distr = [0] * self.n_cls

        for cls in range(self.n_cls):
            probs[cls] = np.mean(y == cls)
            pd_list = self.x_distr.distr[cls].pd_list
            name = self.x_distr.distr[cls].name
            for i, pd in enumerate(pd_list):

                def opt_func(w):
                    loc, scale = w[0], w[1]
                    # partial_likelihood rerurns either inf or negative values, FIXIT maybe
                    lhd =  partial_likelihood(X[pd.idx, :], pd.pd, {"loc": loc, "scale": scale})
                    res = max(-np.mean(np.log(lhd)), MIN_LHD)
                    res = min(res, MAX_LHD)
                    return res

                loc, scale = pd.pd.fit(X[y==cls, pd.idx])
                # initial_opt_func = opt_func((loc, scale))
                res = minimize(opt_func, (loc, scale), method=method, options = {'xtol': 1e-8, 'disp': True, 'maxfev':500})
                loc, scale = res.x
                pd_list[i] = probDistrj(pd.pd, {"loc":loc, "scale":scale}, pd.idx)


            distr[cls] = probDistrList(pd_list, name)

        if replace:
            self.x_distr = probDistrXy(probs, distr)

            return probDistrXy(probs, distr)


    def fit_distr_with_prior(self, y, X, prior=None, replace=True):
        """


        :param y:
        :type y:
        :param X:
        :type X:
        :param prior:
        :type prior:
        :param replace:
        :type replace:
        :return:
        :rtype:
        """

        if prior is None:
            return self.fit_distr_pars(y, X, replace)


        probs = np.ones_like(self.probs)
        distr = [0] * self.n_cls

        for cls in range(self.n_cls):
            probs[cls] = np.mean(y == cls)
            pd_list = self.x_distr.distr[cls].pd_list
            name = self.x_distr.distr[cls].name
            for i, pd in enumerate(pd_list):
                loc, scale = pd.pd.fit(X[y == cls, pd.idx])
                pd_list[i] = probDistrj(pd.pd, {"loc": loc, "scale": scale}, pd.idx)

            distr[cls] = probDistrList(pd_list, name)

        if replace:
            self.x_distr = probDistrXy(probs, distr)

            return probDistrXy(probs, distr)



def partial_likelihood(X, pd, pars):

    frozen_distr = pd(**pars)
    return np.squeeze(frozen_distr.pdf(X))


def sigmoid(X, y, w=None, n_cls=None, norm=True):

    probs = [np.exp(np.dot(X , w[:, k])) for k in range(n_cls)]
    y_probs = np.hstack([probs[int(yi)][i] for i, yi in enumerate(y)])

    if norm:
        y_probs = y_probs / np.sum(probs, axis=0)
    #mask = np.vstack([list(range(n_cls)) == yi for yi in list(y)])


    return y_probs




def generate_sample(n_samples, gen_distr=None):
    """
    Generate sample from GenerativeDistribution

    :param n_samples: Sample size
    :type n_samples: int
    :param gen_distr: instance of GenerativeDistribution class
    :return: y, X
    :rtype: ndarray, ndarray
    """

    if gen_distr is None:
        gen_distr =  GenerativeDistribution()# create_default_distr

    y, X = gen_distr.sample_data(n_samples=n_samples)
    return y, X, gen_distr

def two_class_sample(n_feats=1, n_samples=100, weights=(0.3, 0.7), std=(1,1), delta_mu=0):
    """
    :param n_feats:
    :param n_samples:
    :param weights: ratios of class objects. First is class zero
    :param std:
    :param delta_mu:
    :return:
    """
    X, Y = make_classification(n_samples=n_samples, n_features=n_feats, n_redundant=0,
                             n_clusters_per_class=1, class_sep=1.0, weights=weights)
    X = X*np.transpose(np.tile(Y*std[0] + (1-Y)*std[1], (2, 1)))
    print(np.mean(X[Y==0,:]), np.mean(X[Y==1,:]))
    return X, Y

def gen_sample_by_distr(n_samples=100, n_feats=1,  n_classes=2, weights=None, distr=None, distr_name=None):
    if not weights is None:
        weights = np.array(weights)
        n_classes = weights.shape[0]
    else:
        weights = np.ones((n_classes, 1))
    weights /= np.sum(weights)

    if distr is None and not distr_name is None:
        if distr_name == 'uniform':
            distr = stats.uniform()
        elif distr_name == 'exp':
            distr = stats.expon()
        else:
            distr = stats.multivariate_normal(mean=[0]*n_feats, cov=1)

    X = distr.rvs(size=n_samples)
    y = np.digitize(stats.uniform.rvs(size=n_samples), np.cumsum(weights))
    for i in range(n_classes):
        idx = y == i
        X[idx, :] += np.ones((np.sum(idx), n_feats))*i

    return X, y




