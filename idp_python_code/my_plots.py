import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy import stats
import itertools


def plot_selection_results(data, learners):
    n_subplots = len(learners) + 1
    n_rows = n_subplots // 2 + n_subplots % 2
    f, ax = plt.subplots(n_rows, 2, figsize=(12, 3.5*n_rows))
    plt.subplots_adjust(right=0.9)

    if n_rows == 1:
        ax = ax[None, :]

    ax[0, 0] = plot_2dim_data(data, None, None, ax[0, 0])
    ax[0, 0].set_title("Data")
    row, col = 0, 1
    for learner in learners:
        ax[row, col] = plot_2dim_data(data, learner.labeled_instances, learner.labels, ax[row, col])
        ax[row, col].set_title(learner.name)
        col = (col + 1) % 2
        row += (col + 1) % 2
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.show()
    return f


def plot_2dim_data(data, labeled_data, labels, ax, dim0=0, dim1=1):
    if data.type == "classification":
        return plot_2dim_data_clf(data, labeled_data, labels, ax, dim0, dim1)

    if data.type == "regression":
        return plot_1d_data_reg(data, labeled_data, labels, ax, dim0, dim1)
    return ax


def plot_clf_metrics(results):
    f, ax = plt.subplots(2, 2, figsize=(12, 7))
    plt.subplots_adjust(right=0.75, left=0.1)

    for learner, res in list(results.items()):
        sizes, f1, auc, sensitivity, specificity = [], [], [], [], []
        f1_std, auc_std, sensitivity_std, specificity_std = [], [], [], []
        for size, d in list(res.items()):
            sizes.append(size)
            f1.append(np.mean(d["f1"]))
            f1_std.append(np.std(d["f1"]))
            auc.append(np.mean(d["auc"]))
            auc_std.append(np.std(d["auc"]))
            sensitivity.append(np.mean(d["sensitivity"]))
            sensitivity_std.append(np.std(d["sensitivity"]))
            specificity.append(np.mean(d["specificity"]))
            specificity_std.append(np.std(d["sensitivity"]))

        idx = np.argsort(sizes)
        sizes = np.array(sizes)[idx]
        f1 = np.array(f1)[idx]
        auc = np.array(auc)[idx]
        sensitivity = np.array(sensitivity)[idx]
        specificity = np.array(specificity)[idx]
        l = ax[0, 0].plot(sizes, f1, lw=2, label=learner)
        ax[0, 0].fill_between(sizes, f1 - f1_std, f1 + f1_std, alpha=0.1, color=l[0]._color)
        l = ax[0, 1].plot(sizes, auc, lw=2, label=learner)
        ax[0, 1].fill_between(sizes, auc - auc_std, auc + auc_std, alpha=0.1, color=l[0]._color)
        l = ax[1, 0].plot(sizes, sensitivity, lw=2, label=learner)
        ax[1, 0].fill_between(sizes, sensitivity - sensitivity_std, sensitivity + sensitivity_std,
                              alpha=0.1, color=l[0]._color)
        l = ax[1, 1].plot(sizes, specificity, lw=2, label=learner)
        ax[1, 1].fill_between(sizes, specificity - specificity_std, specificity + specificity_std,
                              alpha=0.1, color=l[0]._color)

    ax[0, 0].set_title("F1 score")
    ax[0, 1].set_title("AUC")
    ax[1, 0].set_title("Sensitivity")
    ax[1, 1].set_title("Specificity")

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.show()
    return f


def plot_reg_metrics(results):
    f, ax = plt.subplots(1, 3, figsize=(15, 4))
    plt.subplots_adjust(right=0.8, left=0)

    for learner, res in list(results.items()):
        sizes, mae, mse, r2 = [], [], [], []
        mae_std, mse_std, r2_std = [], [], []
        for size, d in list(res.items()):
            sizes.append(size)
            mae.append(np.mean(d["mae"]))
            mse.append(np.mean(d["mse"]))
            r2.append(np.mean(d["r2"]))
            mae_std.append(np.std(d["mae"]))
            mse_std.append(np.std(d["mse"]))
            r2_std.append(np.std(d["r2"]))

        idx = np.argsort(sizes)
        sizes = np.array(sizes)[idx]
        mae = np.array(mae)[idx]
        mse = np.array(mse)[idx]
        r2 = np.array(r2)[idx]
        l = ax[0].plot(sizes, mae, label=learner)
        ax[0].fill_between(sizes, mae + mae_std, mae + mae_std, alpha=0.1, color=l[0]._color)
        l = ax[1].plot(sizes, mse, label=learner)
        ax[0].fill_between(sizes, mse + mse_std, mse + mse_std, alpha=0.1, color=l[0]._color)
        l = ax[2].plot(sizes, r2, label=learner)
        ax[0].fill_between(sizes, mse + mse_std, mse + mse_std, alpha=0.1, color=l[0]._color)

    ax[0].set_title("MAE")
    ax[1].set_title("MSE")
    ax[2].set_title("R2")

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return f


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


# # Init only required for blitting to give a clean slate.
# def init():
#     line.set_ydata(np.ma.array(x, mask=True))
#     return line,

def animate_clf_results(data, learner, dim0=0, dim1=1):
    fig, ax = plt.subplots()

    ax = plot_2dim_data_clf(data, ax=ax, dim0=dim0, dim1=dim1)
    line = ax.plot(data.X[:, dim0], np.ones(len(data.X)) * np.mean(data.X[:, dim1]))

    def animate(coef_p):
        coef, p = coef_p[0], coef_p[1]
        line.set_ydata(np.dot(coef, data.X.T) - p)  # update the data
        return line,

    frames = zip(learner.coefs, learner.ps)
    ani = animation.FuncAnimation(fig, animate, frames,  # init_func=init,
                                  interval=25, blit=True)
    plt.show()


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
