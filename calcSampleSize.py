from scipy.stats import norm
from logistic import *
import math
import numpy as np
p0 = 0.5

def ss_equality_test(y, x, pars):
    mu1 =  x[np.nonzero(y)].mean(0)
    mu0 = x[np.nonzero(1-y)].mean(0)

    p = y.mean(0)
    z_alpha = norm.ppf(1 - pars['alpha']/2)
    z_pow = norm.ppf(pars['power'])

    ss = (z_alpha + z_pow)**2*p*(1-p)/(p-p0)**2
    return ss

def sample_size_wald(y, x, pars):
    z_alpha = norm.ppf(1 - pars['alpha']/2)
    z_pow = norm.ppf(pars['power'])

    m,n = x.shape
    X = np.ones((m, n+1))
    X[:, 1:] = x
    theta = optimize_theta(X, y)
    print(theta)
    p = y.mean(0)
    p0 = 0.5
    ss = (z_alpha + z_pow)**2*p*(1-p)/theta[1]**2
    return ss

    

def sample_size_norm(y, x, pars):
    s_size = []
    z_alpha = norm.ppf(1 - pars['alpha'])
    mu = x.mean(0)
    sigma = x.std(0)
    
    z_alpha = z_alpha*np.ones([1, len(mu)])
    s_size = (z_alpha*sigma/mu)**2
    return s_size
