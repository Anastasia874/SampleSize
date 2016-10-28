from __future__ import division
from __future__ import print_function


import numpy as np
from scipy import stats
from sklearn.preprocessing import normalize
from sklearn.datasets import make_classification
from collections import namedtuple

probDistrj = namedtuple("probDistrj", "pd pars idx")
probDistrList = namedtuple("probDistrList", "pd_list name")
probDistrXy = namedtuple("probDistrXy", "probs distr")


def sample_from_prob_distrj(n_samples, distr):
    # distr is an instance  probDistrj

    frozen_distr = distr.pd(**distr.pars)
    z = frozen_distr.rvs(size=int(n_samples))

    return z

class ProbDistr():

    def __init__(self, n_feats=1):
        self.n_feats = n_feats


    def create_distr_by_name(self, distr_name, n_dims=None, pars={}):
        if n_dims is None:
            n_dims = self.n_feats
            

        if isinstance(distr_name, list):
            distr = [0] * len(distr_name)
            for i, d_name in enumerate(distr_name):
                distr[i] = self.create_distr_by_name(d_name, n_dims=None, pars={"loc":i})

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

        # if not self.check_distr(distr) :
        #     raise ValueError

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

    def make_y_labels(self, X=None, lh=None, w=None):

        if lh is None:
            lh = self.likelihood(X)

        y = np.argmax(lh, axis=1)

        return y


    def likelihood(self, X, x_distr=None, y=None):

        if x_distr is None:
            x_distr = self.x_distr

        lh = np.zeros(X.shape[0])
        lh_y_none = np.zeros((X.shape[0], len(x_distr)))
        for cls in range(len(x_distr)):
            lh_by_class = np.zeros((X.shape[0], len(x_distr[cls].pd_list)))

            for j, xdj in enumerate(x_distr[cls].pd_list):
                frozen_distr = xdj.pd(**xdj.par)
                lh_by_class[:, j] = frozen_distr.pdf(X[:, xdj.idx[j]])

            lh_y_none[:, cls] = np.prod(lh_by_class, axis=1)

            if not y is None:
                lh[y == cls] = lh_y_none[y == cls, cls]

        if y is None:
            return lh_y_none

        return lh




def sigmoid(X, y, w=None, n_cls=None, norm=True):

    probs = [np.exp(np.dot(X , w[:, k])) for k in range(n_cls)]
    y_probs = np.hstack([probs[int(yi)][i] for i, yi in enumerate(y)])

    if norm:
        y_probs = y_probs / np.sum(probs, axis=0)
    #mask = np.vstack([list(range(n_cls)) == yi for yi in list(y)])


    return y_probs




def generate_sample(n_samples, gen_distr=None):

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




