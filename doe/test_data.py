from __future__ import division
from __future__ import print_function

from sklearn import mixture
from sklearn import metrics
import itertools
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
from scipy import stats
from sklearn.externals import joblib

from active_learning import *

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])


def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-9., 5.)
    plt.ylim(-3., 6.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)


def plot_data(X, Y_):

    color_iter =['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange']
    for i, y in enumerate(np.unique(Y_)):
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color_iter[i])
    plt.show()

def plot_sample(sample, labels, title, start_sample=None):
    plt.scatter(sample[labels == 0, 0], sample[labels == 0, 1], color="blue")
    plt.scatter(sample[labels == 1, 0], sample[labels == 1, 1], color="red")
    # if start_sample is not None:
    #     # plt.plot(sample[start_sample:, 0], sample[start_sample:, 1], c="k")
    #     plt.plot(sample[start_sample:, 0], sample[start_sample:, 1], c="k", ls="o")
    plt.title(title)
    plt.show()


def sample_from_prob_distr(n_samples, pd, pars):
    # distr is an instance  probDistr
    frozen_distr = pd(**pars)
    z = frozen_distr.rvs(size=int(n_samples))

    return z

def random_sample_(X, sample_size):
    idx = range(len(X))
    np.random.shuffle(idx)
    idx = idx[:sample_size]
    return idx, X[idx]


def calc_metrics(sample, X, true_labels):
    dpgmm = mixture.BayesianGaussianMixture(n_components=2,
                                            covariance_type='full').fit(sample)
    pred_labels = dpgmm.predict(X)
    v_score = metrics.v_measure_score(true_labels, pred_labels)

    return v_score, dpgmm


def one_step_choice(X, dpgmm=None):
    if dpgmm is None:
        dpgmm = mixture.BayesianGaussianMixture(n_components=2,
                                            covariance_type='full')
    dpgmm.fit(X)
    t_func = np.zeros(dpgmm.n_components)
    N_TRIES = 50
    for nc in range(dpgmm.n_components):
        # sample several data points from multivariate normal distribution, with parameters of nc component
        samples = sample_from_prob_distr(N_TRIES, stats.multivariate_normal, {"mean":dpgmm.means_[nc],
                                                                       "cov":dpgmm.covariances_[nc]})
        lhd = np.zeros(N_TRIES)
        for i in range(N_TRIES):
            dpgmm_nc = copy.deepcopy(dpgmm)
            # fit distribution to augmented data
            dpgmm_nc.fit(np.vstack((X, samples[i])))
            lhd[i] = dpgmm_nc.lower_bound_ # variational bound of data likelihood

        t_func[nc] = np.mean(lhd) # expected likelihood ratio

    nc = np.argmax(t_func) # select component to sample from

    return nc, t_func[nc]


def reassign_labels(true_labels, pred_labels, eps=0.1):
    if metrics.v_measure_score(true_labels, pred_labels) > 0.5 + eps and\
                    np.mean(true_labels==pred_labels) < 0.5 - eps:
        true_labels = np.logical_not(true_labels)

    return true_labels

# Number of samples per component

MIN_SAMPLE_SIZE = 25
MAX_SAMPLE_SIZE = 250
N_ITERS = 100
n_samples = 1000

# Generate random sample, two components
np.random.seed(0)
C = np.array([[0., -0.5], [1.7, .4]])
X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
          .7 * np.random.randn(n_samples, 2) + np.array([-2, 1])]
true_labels = np.hstack((np.zeros(n_samples), np.ones(n_samples)))
idx = range(len(X))
np.random.shuffle(idx)
X, true_labels = X[idx], true_labels[idx]
idx = range(len(X) - MIN_SAMPLE_SIZE)

start_sample = X[:MIN_SAMPLE_SIZE]
start_labels = true_labels[:MIN_SAMPLE_SIZE]

X = X[MIN_SAMPLE_SIZE:]
true_labels = true_labels[MIN_SAMPLE_SIZE:]
true_dpgmm = mixture.BayesianGaussianMixture(n_components=2,
                                covariance_type='full').fit(X)

if metrics.f1_score(true_labels, true_dpgmm.predict(X)) < 0.5:
    true_labels = reassign_labels(true_labels, true_dpgmm.predict(X))

T = np.zeros((N_ITERS, MAX_SAMPLE_SIZE))
v_score = np.zeros((N_ITERS, MAX_SAMPLE_SIZE))
rd_v_score = np.zeros((N_ITERS, MAX_SAMPLE_SIZE))
cmp_v_score = np.zeros(N_ITERS)

for i in range(N_ITERS):
    print("Iteration number {}".format(i))
    future_random_sample, future_random_labels = true_dpgmm.sample(MAX_SAMPLE_SIZE - MIN_SAMPLE_SIZE)
    shfl = np.arange(MAX_SAMPLE_SIZE - MIN_SAMPLE_SIZE)
    np.random.shuffle(shfl)
    future_random_sample, future_random_labels = future_random_sample[shfl], future_random_labels[shfl]
    future_random_sample = np.vstack((start_sample, future_random_sample))
    future_random_labels = np.hstack((start_labels, future_random_labels))
    sample, labels = start_sample, start_labels
    random_sample, random_labels = start_sample, start_labels
    sample_size = MIN_SAMPLE_SIZE
    np.random.shuffle(idx)
    complete_random_sample = X[idx[:MAX_SAMPLE_SIZE]]
    cmp_v_score[i], _ = calc_metrics(complete_random_sample, X, true_labels)

    while sample_size < MAX_SAMPLE_SIZE:

        # calc quality metrics for random sampling:
        rd_v_score[i, sample_size], _ = calc_metrics(random_sample, X, true_labels)

        # calc quality metrics for one-step-ahead sampling
        v_score[i, sample_size], dpgmm = calc_metrics(sample, X, true_labels)
        nc, T[i, sample_size] = one_step_choice(sample, dpgmm)

        # sample data point from the true distribution of the selected component
        new_sample = sample_from_prob_distr(1, stats.multivariate_normal, {"mean": true_dpgmm.means_[nc],
                                                                           "cov": true_dpgmm.covariances_[nc]})

        # sample data point from the true distribution at random
        # new_random_sample, new_rd_lbl = true_dpgmm.sample()
        # print("Assigned label {}, probability {}.".format(new_rd_lbl, true_dpgmm.predict_proba(new_random_sample)))
        # random_sample, random_labels = np.vstack((random_sample, new_random_sample)), np.hstack((random_labels, new_rd_lbl))
        random_sample, random_labels = future_random_sample[:sample_size], future_random_labels[:sample_size]

        sample = np.vstack((sample, new_sample))
        labels = np.hstack((labels, nc))


        sample_size += 1

res = {}
res["X"] = X
res['true_labels'] = true_labels
res['true_dpgmm'] = true_dpgmm

res['sample'] = sample
res['labels'] = labels
res['random_sample'] = random_sample
res['random_labels'] = random_labels

res['rd_v_score'] = rd_v_score
res['v_score'] = v_score
res['cmp_v_score'] = cmp_v_score
res['T'] = T

joblib.dump(res, "res.pkl")


plt.plot(range(MIN_SAMPLE_SIZE, MAX_SAMPLE_SIZE), np.mean(v_score, axis=0)[MIN_SAMPLE_SIZE:], label="one-step")
plt.plot(range(MIN_SAMPLE_SIZE, MAX_SAMPLE_SIZE), np.mean(rd_v_score, axis=0)[MIN_SAMPLE_SIZE:], label="random")
plt.plot(range(MIN_SAMPLE_SIZE, MAX_SAMPLE_SIZE), np.mean(cmp_v_score) *np.ones(MAX_SAMPLE_SIZE - MIN_SAMPLE_SIZE), label="cmp random")
plt.legend(loc="best")
plt.xlabel("Sample size")
plt.ylabel("V-score")
plt.show()
plt.plot(range(MIN_SAMPLE_SIZE, MAX_SAMPLE_SIZE),np.mean(T, axis=0)[MIN_SAMPLE_SIZE:])
plt.xlabel("Sample size")
plt.ylabel("KL-loss")
plt.show()


plot_sample(sample, labels, "One_step", MIN_SAMPLE_SIZE)
plot_sample(random_sample, random_labels, "Random", MIN_SAMPLE_SIZE)




    # plot_data(sample, true_labels[idx])



def random_sampling(X):
    v_score, v_train = [], []
    sample_size_values = range(25, 1000, 50)
    for sample_size in sample_size_values:
        v_score.append([])
        v_train.append([])
        for i in range(20):
            idx, sample = random_sample_(X, sample_size)
            dpgmm = mixture.BayesianGaussianMixture(n_components=2,
                                                covariance_type='full').fit(sample)
            pred_labels = dpgmm.predict(X)
            v_train[-1].append(metrics.v_measure_score(true_labels[idx], pred_labels[idx]))
            v_score[-1].append(metrics.v_measure_score(true_labels, pred_labels))

            new_sample, sample = one_step_choice(sample, dpgmm)

        #plot_data(sample, true_labels[idx])
        v_score[-1] = np.mean(v_score[-1])
        v_train[-1] = np.mean(v_train[-1])

    plt.plot(sample_size_values, v_score)
    plt.plot(sample_size_values, v_train)
    plt.show()
