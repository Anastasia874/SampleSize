from scipy.stats import norm, chi2
from logistic import *
import math, random
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

def sample_size_chi2(y, x, pars):
    alpha = pars['alpha']
    m, n = x.shape
    x = np.squeeze(np.asarray(x))
    y = np.squeeze(np.asarray(y))
    nCl = len(np.unique(np.asarray(y)))
        
    s_size = m
    N = round(pow(m, 1/3))
    nBinsArray = [nCl]
    nBinsArray.extend([N]*n)
    p, edges = np.histogramdd((y, x), bins=nBinsArray)
    idx = range(0,m)
    
    chi_min = chi2.ppf(alpha/2, 1.5*N)
    chi_max = chi2.ppf(1-alpha/2, 1.5*N)
    while s_size > N:
        chi2d = []
        for nIter in range(0, 50):
            random.shuffle(idx)
            idx1 = idx[:s_size]
            random.shuffle(idx)
            idx2 = idx[:s_size]
            chi2d.append(chi2div(y, x, idx1, idx2, edges))
        chi2d = sum(chi2d)/nIter    
        if chi2d < chi_min or chi2d > chi2_max:
            s_size += 1
            break
        s_size -= 1    

    

    return s_size    

def chi2div(y, x, idx1, idx2, edges):
    n1, edges1 = np.histogramdd([y[idx1], x[idx1]], bins=edges)
    n2, edges2 = np.histogramdd([y[idx2], x[idx2]], bins=edges)
    chi2d = sum(len(idx1)*pow(n1 - n2, 2)/(2*n2*(len(idx1)-n2)))

    #chi2 = m*(n1 - n2).^2./(2*n2.*(m-n2));

    return chi2d