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
            self.lh_func = LikelihoodFunc(func=sigmoid, n_cls=self.n_cls, n_feats=self.n_feats + 1, transform="poly", w_distr=w_distr)



    def sample_data(self, n_samples, distr=None):
        if distr is None:
            distr = self.x_distr


        X = self.sample(n_samples, distr)

        if X.ndim == 1:
            X = X[:, None]

        lh = self.lh_func.func(X=X)
        self.cls_intercepts = label_by_hist_counts(data=lh, probs=self.probs, nbins=100)
        y = self.make_y_labels(lh=lh, w=self.lh_func.opt_pars)
        print("Generated class ratios {}, expected {}".format([sum(y==i) for i in range(self.n_cls)], [self.probs]))

        return y, X

    def make_y_labels(self, X=None, lh=None, w=None):

        if lh is None:
            lh = self.lh_func.func(X=X, w=w)
            
        n_cls = len(self.cls_intercepts) - 1
        y = np.digitize(lh, self.cls_intercepts) - 1
        print("Out of class samples: {}\%".format(np.mean(y >= n_cls)*100))
        y[y >= n_cls] = n_cls - 1

        return y





class LikelihoodFunc(ProbDistr):

    def __init__(self, func, n_cls, n_feats=None, opt_pars=None, w_distr=None, y_probs=None, intercept=True, transform=None):
        ProbDistr.__init__(self, n_feats)
        self.intercept = intercept


        self.transform = self.make_transform_fn(transform)


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
            opt_pars = self.sample(n_feats, distr=self.w_distr)
            opt_pars = np.squeeze(normalize(opt_pars, axis=1))
            if self.intercept:
                self.opt_pars = np.hstack((1, opt_pars))

        
        
        #self.opt_pars = normalize(self.opt_pars, axis=0)

        self.n_feats = self.opt_pars.shape[0]
        self.func = self.make_likelihood_fn(func)
        self.transformedX = None
        self.w_distr = w_distr



    def make_likelihood_fn(self, func):
        def lh_func(X, y=None, w=None, norm=True):

            if w is None:
                w = self.opt_pars

            X = self.transform(X, intercept=self.intercept)
            if self.transformedX is None:
                self.transformedX = X


            lh = func(X=X, w=w)
            return lh

        return lh_func

    def make_transform_fn(self, transform=None):

        if hasattr(transform, "__call__"):
            return transform

        if transform is None:
            transform = "identity"

        if not isinstance(transform, str):
            print("transform should be a string with function name or callable")
            raise ValueError

        if transform == "poly":
            def transform(X, n_degrees=self.n_feats, intercept=True):
                d0 = 1
                if intercept:
                    d0 = 0
                return np.hstack([np.power(X, i) for i in range(d0, n_degrees + 1)])
        if transform == "identity":
            def transform(X, intercept=True):
                if intercept:
                    X = np.hstack((np.ones(X.shape[0], 1), X))
                return X

        return transform




def label_by_hist_counts(data, probs, nbins=100):

    data = np.squeeze(data)

    hist, bins = np.histogram(data, bins=nbins, density=True)

    hist = np.cumsum(hist) / np.sum(hist)
    probs = np.cumsum(np.hstack((0, probs)))
    new_bins = np.zeros(len(probs))
    for cls, prb in enumerate(probs[1:]):
        new_bins[cls + 1] = max(bins[hist <= prb])


    return new_bins



def sigmoid(X, w=None, norm=True):

    y_probs = 1/(1 + np.exp(-np.dot(X , w)))

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




