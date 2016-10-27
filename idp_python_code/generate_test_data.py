from __future__ import division
from __future__ import print_function


import numpy as np
from scipy import stats
from sklearn.preprocessing import normalize
from sklearn.datasets import make_classification



class ProbDistr():

    def __init__(self, n_feats=1):
        self.n_feats = n_feats

    # def create_distr_by_name(self, distr_name):
    #     if distr_name == 'uniform':
    #         distr = stats.uniform()
    #     elif distr_name == 'exp':
    #         distr = stats.expon()
    #     else:
    #         if distr_name is None:
    #             print("Feature distribution is not specified! Using multivariate normal")
    #         distr = stats.multivariate_normal(mean=[0] * self.n_feats, cov=1)
    #
    #     return distr

    def create_distr_by_name(self, distr_name, n_dims=None):
        if n_dims is None:
            n_dims = self.n_feats
            
        distr = {}
        distr["name"] = distr_name
        distr["mvr"] = False
        if distr_name == 'uniform':
            distr["pd"] = [stats.uniform() for i in range(n_dims)]
        elif distr_name == 'exp':
            distr["pd"] = [stats.expon() for i in range(n_dims)]
        elif distr_name == "gamma":
            distr["pd"] = [stats.gamma() for i in range(n_dims)]
        else:
            if distr_name is None:
                print("Feature distribution is not specified! Using multivariate normal")
            else:
                print("Feature distribution {} is not supported. Using multivariate normal".format(distr_name))
            distr["pd"] = stats.multivariate_normal(mean=[0] * n_dims, cov=1)
            distr["name"] = "Multivariate normal"
            distr["mvr"] = True


        return distr

    def check_distr(self, distr):

        distr_attr = list(distr.keys())

        if not isinstance(distr, dict):
            print("Probability distribution distr should be a dict")
            return False

        if not "pd" in distr_attr:
            print("Probability distribution is not defined")
            return False

        if not "mvr" in distr_attr:
            print("mvr field is not defined")
            return False

        if not distr["mvr"]:
            if not isinstance(distr["pd"], list):
                print("distr['pd'] for 'mvr = False' should contain a list of generators")
                distr["pd"] = [distr["pd"]]
        else:
            pass
            # if mvr == True, then distr["pd"] is a single scipy.stats.... prob_gen object

        return True

    def sample(self, n_samples, distr):

        if not self.check_distr(distr) :
            raise ValueError

        if distr["mvr"]:
            z = distr["pd"].rvs(size=n_samples)

        else:
            n_feats = len(distr["pd"])
            z = [0] * n_feats

            for n in range(self.n_feats):
                z[n] = distr[n].rvs(size=n_samples)
            z = np.vstack(z)


        return z



class GenerativeDistribution(ProbDistr):

    def __init__(self, n_feats=1, n_cls=2, x_distr=None, lh_func=None, x_distr_name=None, w_distr=None, probs=None):
        ProbDistr.__init__(self, n_feats)
        self.n_cls = n_cls
        self.x_distr_name = str(x_distr_name)
        self.probs = probs
        if self.probs is None:
            self.probs = np.ones(n_cls) / n_cls

        if not x_distr is None:
            self.x_distr = x_distr
        else:
            self.x_distr = self.create_distr_by_name(x_distr_name)

        if not lh_func is None:
            self.lh_func = lh_func
            self.lh_func.n_cls = n_cls
        else:
            self.lh_func = LikelihoodFunc(func=poly_sigmoid, n_cls=self.n_cls, n_feats=self.n_feats + 4, w_distr=w_distr)


    def sample_data(self, n_samples, distr=None):
        if distr is None:
            distr = self.x_distr


        X = self.sample(n_samples, distr)

        if X.ndim == 1:
            X = X[:, None]

        # self.lh_func.func(X=X)
        # transfX = self.lh_func.transformedX

        lh = self.lh_func.func(X=X, y=np.ones(X.shape[0]), norm=False)
        lh = 1/(1 + lh)
        y = label_by_hist_counts(data=lh, probs=self.probs, nbins=100)

        # lh_matrix = np.zeros((n_samples, self.n_cls))
        # for cls in range(self.n_cls):
        #     lh_matrix[:, cls] = self.lh_func.func(X=X, y=cls*np.ones(X.shape[0]), w=self.lh_func.opt_pars)
        #
        # y = np.argmax(lh_matrix, axis=1)

        return y, X




class LikelihoodFunc(ProbDistr):

    def __init__(self, func, n_cls, n_feats=None, opt_pars=None, w_distr=None, y_probs=None, intercept=True):
        ProbDistr.__init__(self, n_feats)
        self.intercept = intercept


        if not y_probs is None:
            n_cls = len(y_probs)
            y_probs = np.cumsum(y_probs) / n_cls

        self.opt_pars = opt_pars
        if not opt_pars is None:
            self.n_feats, self.n_cls = opt_pars.shape
            if not self.n_cls is None and not self.n_cls == n_cls:
                print("Number of classes {0} specified explicitly does not correspond to opt_pars shape[1]={1}"
                      .format(n_cls, self.n_cls))
                
            if not n_feats is None and not self.n_feats == n_feats:
                print("Number of features {0} specified explicitly does not correspond to opt_pars shape[0]={1}"
                      .format(n_feats, self.n_feats))
        else:
            self.n_cls = n_cls
            self.n_feats = n_feats
        
        if w_distr is None and not self.n_cls is None:
            if y_probs is None:
                y_probs = np.cumsum(np.ones(self.n_cls)/n_cls)
            self.w_distr = self.create_distr_by_name("Gamma", n_dims=1)

        elif self.n_feats is None:
            print("Specify either n_feats and n_cls, w_distr or opt_pars")
            raise ValueError
        
        if self.opt_pars is None:
            opt_pars = self.sample(n_feats, distr=self.w_distr)[:, None]
            opt_pars = np.tile(opt_pars, n_cls)
            opt_pars = normalize(opt_pars, axis=0)
            self.opt_pars = np.vstack((y_probs, opt_pars))

        
        
        #self.opt_pars = normalize(self.opt_pars, axis=0)

        self.n_feats = self.opt_pars.shape[0]
        self.func = self.make_likelihood_fn(func)
        self.transformedX = None
        self.w_distr = w_distr



    def make_likelihood_fn(self, func):
        def lh_func(X, y, w=None, norm=True):

            if w is None:
                w = self.opt_pars

            lh, transformedX = func(X=X, y=y, w=w, n_cls=self.n_cls, intercept=self.intercept, norm=norm)
            if self.transformedX is None:
                self.transformedX = X
            return lh

        return lh_func


def label_by_hist_counts(data, probs, nbins=100):

    data = np.squeeze(data)
    n_cls = len(probs)
    hist, bins = np.histogram(data, bins=nbins, density=True)

    hist = np.cumsum(hist) / np.sum(hist)
    probs = np.cumsum(np.hstack((0, probs)))
    new_bins = np.zeros(len(probs))
    for cls, prb in enumerate(probs[1:]):
        new_bins[cls + 1] = max(bins[hist <= prb])

    y = np.digitize(data, new_bins) - 1
    y[y >= n_cls] = n_cls - 1
    return y



# def poly_sigmoid(X, y=None, w=None, n_cls=None, intercept=True):
#
#     ndims = w.shape[0]
#     polyX = []
#
#     if intercept:
#         ndims = ndims -1
#         polyX = [np.ones((X.shape[0], 1))]
#
#     n_degrees = ndims // X.shape[1]
#     for nd in range(n_degrees):
#         polyX.append(np.power(X, nd))
#
#     X = np.hstack(polyX)
#
#     if y is None:
#         return None, X
#
#     probs = 1./(1 + np.exp(-np.dot(X, w)))
#     return probs, X


def poly_sigmoid(X, y=None, w=None, n_cls=None, intercept=True, norm=True):

    ndims = w.shape[0]
    polyX = []

    if intercept:
        ndims = ndims -1
        polyX = [np.ones((X.shape[0], 1))]

    n_degrees = ndims // X.shape[1]
    for nd in range(n_degrees):
        polyX.append(np.power(X, nd))

    X = np.hstack(polyX)

    if y is None:
        return None, X

    probs = [np.exp(np.dot(X , w[:, k])) for k in range(n_cls)]
    y_probs = np.hstack([probs[int(yi)][i] for i, yi in enumerate(y)])
    if norm:
        y_probs = y_probs / np.sum(probs, axis=0)
    #mask = np.vstack([list(range(n_cls)) == yi for yi in list(y)])


    return y_probs, X


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




