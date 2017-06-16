from __future__ import division
from __future__ import print_function

import numpy as np

from sklearn.externals import joblib
from sklearn.model_selection import KFold
from sklearn import mixture
from sklearn import metrics
import matplotlib.pyplot as plt

def plot_sample(sample, labels, title, start_sample=None):
    plt.scatter(sample[labels == 0, 0], sample[labels == 0, 1], color="blue")
    plt.scatter(sample[labels == 1, 0], sample[labels == 1, 1], color="red")
    if start_sample is not None:
        plt.plot(sample[start_sample:, 0], sample[start_sample:, 1], c="k")
    plt.title(title)
    plt.show()

def calc_metrics(sample, X, true_labels):
    dpgmm = mixture.BayesianGaussianMixture(n_components=2,
                                            covariance_type='full').fit(sample)
    pred_labels = dpgmm.predict(X)
    v_score = metrics.v_measure_score(true_labels, pred_labels)
    return v_score, dpgmm


def average_random(X, Y, n_samples, n_cvs=100):

    idx = range(len(X))
    v_score = np.zeros(n_cvs)
    for n in range(n_cvs):
        np.random.shuffle(idx)
        train_idx, val_idx = idx[:n_samples], idx[:n_samples]
        X_train, X_val = X[train_idx], X[val_idx]
        #y_train, y_val = Y[train_idx], Y[val_idx]

        v_score[n], _ = calc_metrics(X_val, X, Y)


    return np.mean(v_score)







MIN_SAMPLE_SIZE = 25


res = joblib.load("res.pkl")

X = res["X"]
true_labels = res['true_labels']
true_dpgmm = res['true_dpgmm']

sample = res['sample']
labels = res['labels']
MAX_SAMPLE_SIZE = len(sample)

random_sample = res['random_sample']
random_labels = res['random_labels']

random_score = average_random(X, true_labels, MAX_SAMPLE_SIZE, n_cvs=100)


rd_v_score = res['rd_v_score']
v_score = res['v_score']

plt.imshow(rd_v_score, interpolation="None", aspect='auto')
plt.show()
plt.imshow(v_score, interpolation="None", aspect='auto')
plt.show()

rd_v_score = np.mean(rd_v_score, axis=0)[MIN_SAMPLE_SIZE:]
v_score = np.mean(v_score, axis=0)[MIN_SAMPLE_SIZE:]
# rd_v_score = np.zeros(len(sample) - MIN_SAMPLE_SIZE)
# v_score = np.zeros(len(sample) - MIN_SAMPLE_SIZE)
# for i in range(MIN_SAMPLE_SIZE, len(sample)):
#     rd_v_score[i - MIN_SAMPLE_SIZE],_ = calc_metrics(random_sample[:i], X, true_labels)
#     # calc quality metrics for one-step-ahead sampling
#     v_score[i - MIN_SAMPLE_SIZE],_ = calc_metrics(sample[:i], X, true_labels)

plt.plot(range(MIN_SAMPLE_SIZE, MAX_SAMPLE_SIZE), v_score, label="one-step")
plt.plot(range(MIN_SAMPLE_SIZE, MAX_SAMPLE_SIZE), rd_v_score, label="random")
plt.plot(range(MIN_SAMPLE_SIZE, MAX_SAMPLE_SIZE), random_score*np.ones(MAX_SAMPLE_SIZE-MIN_SAMPLE_SIZE), label="random")
plt.legend(loc="best")
plt.xlabel("Sample size")
plt.ylabel("V-score")
plt.show()
# plt.plot(range(MIN_SAMPLE_SIZE, MAX_SAMPLE_SIZE),np.mean(T, axis=0)[MIN_SAMPLE_SIZE:])
# plt.xlabel("Sample size")
# plt.ylabel("KL-loss")
# plt.show()