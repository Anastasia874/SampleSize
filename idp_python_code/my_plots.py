import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import itertools


def plot_2dim_data_clf(data, labeled_data=None, labels=None, ax=None, dim0=0, dim1=1):
    color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                                  'darkorange'])
    if data.X.ndim < 2:
        raise ValueError("plot_2dim_data only valid for 2D data, got {}".format(data.X.ndim))

    if ax is None:
        f, ax = plt.subplots(nrows=1, ncols=1)

    y = data.y
    X = data.X[:, [dim0, dim1]]

    for cls, col in zip(data.classes, color_iter):
        if not np.any(y == cls):
            continue
        ax.scatter(X[y == cls, 0], X[y == cls, 1], 0.8, color=col, label=str(cls))
        if not np.any(labels == cls) or labels is None:
            continue
        ax.scatter(labeled_data[labels == cls, 0], labeled_data[labels == cls, 1],
                   s=80, facecolors='none', edgecolors=col)

    return ax


def plot_1d_data_reg(data, labeled_data=None, labels=None, ax=None, dim0=0, dim1=1, col='navy'):

    X = data.X[:, dim0]
    y = data.y

    if ax is None:
        f, ax = plt.subplots(nrows=1, ncols=1)

    ax.scatter(X, y, color=col, label="Unlabeled data")
    if hasattr(data, "coef"):
        line = np.sum(data.coef * data.X, axis=1)
        ax.plot(X, line, label="True model")

    if labeled_data is not None and len(labels) > 0:
        ax.scatter(labeled_data[:, dim0], labels,
                   s=80, facecolors='none', edgecolors="c", label="Labeled data")

    return ax


# def plot_2dim_data_reg(data, labeled_data, labels, ax, dim0=0, dim1=1, col='navy'):
#     if data.X.ndim < 2:
#         raise ValueError("plot_2dim_data only valid for 2D data, got {}".format(data.X.ndim))
#
#     y = data.y
#     X = data.X[:, [dim0, dim1]]
#
#     X, Y =
#
#     return ax


def plot_data(y, x, intercept=False):
    if x.ndim == 1 or x.shape[1] == 1:
        x = np.squeeze(x)
        plot_1d_data(y, x)
        return None

    plt.title("First two dims of the data sample", fontsize='small')
    if not intercept:
        plt.scatter(x[:, 0], x[:, 1], marker='o', c=y)
    elif x.shape[1] > 2:
        plt.scatter(x[:, 1], x[:, 2], marker='o', c=y)
    plt.show()

    # idx = np.argsort(y)
    # data = np.hstack((y, x))
    # plt.imshow(data[idx, :], interpolation="none", aspect="auto")
    # plt.colorbar()


def plot_1d_data(y, x):
    colors = ["b", "g", "r", "c", "m", "k", "y"]
    plt.title("1d sampe", fontsize='small')

    hist_range = (min(x), max(x))
    for cls in np.unique(y):
        plt.scatter(x[y == cls], np.zeros_like(x[y == cls]), marker='x', c=colors[cls])
        hist, bin_edges = np.histogram(x[y==cls], density=True, bins="auto", range=hist_range)
        bins = (bin_edges[0:-1] + bin_edges[1:])/2
        plt.plot(bins, hist, c=colors[cls], lw=2)

    plt.tight_layout()
    plt.xlabel("Feature values")
    plt.show()


def plot_mean_std(X, std, mean):
    plt.plot(mean)
    for i in range(np.shape(X)[0]):
        plt.plot(X[i, :], ls='.')
    plt.show()


def plot_mcmc_hists(samples, prior_func, post_func):
    ''' plot sampled and true posterior (when posterior is known)'''

    priors = [eval(prior_func, w) for w in samples]
    posteriors = [eval(post_func, w) for w in samples]
    nmcmc = len(samples)//2
    plt.figure(figsize=(12, 9))
    plt.hist(priors, 40, histtype='step', normed=True, linewidth=1, label='Distribution of prior samples')
    plt.hist(samples[nmcmc:], 40, histtype='step', normed=True, linewidth=1, label='Distribution of posterior samples')
    plt.plot(samples, posteriors, c='red', linestyle='--', alpha=0.5, label='True posterior')
    plt.xlim([0,1])
    plt.legend(loc='best')
    plt.show()


def plot_mcmc_convergene(samples, labels):
    # Convergence of multiple chains
    ls = ['-', '--', ':']
    c = ['k', 'b', 'r']
    for i in range(len(samples)):
        plt.plot(samples[i], ls=ls[i], c=c[i], label=labels[i])
    plt.legend(loc='best')
    plt.ylim([0, 1])
    plt.show()


def plot_kl_hist(hist_vals, hist_bins, df, name):

    hist_vals /= np.sum(hist_vals)
    pdf = stats.chi2.pdf(hist_bins, df)
    fig = plt.figure()
    plt.bar(hist_bins, hist_vals, width=0.1, label='Observed KL')
    plt.plot(hist_bins, hist_vals, ls='--',  c='k', lw=2, label='Emp. pdf')
    plt.plot(hist_bins, pdf, ls='-',  c='r', lw=2, label='True pdf')

    plt.legend(loc='best')
    fig.savefig(name)

    return np.mean(np.power(pdf - hist_vals, 2))
