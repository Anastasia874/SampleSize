from __future__ import print_function

import copy
import numpy as np
import my_plots

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import det_generate_test_data
from generate_test_data import GenerativeDistribution, ProbDistr, generate_sample


N_CLS = 3
N_FEATS = 2
STD = 0.5
ND_SAMPLES = 5
SAMPLE_SIZE = 100
TRAIN_TEST_RATIO = 0.75

x_distr = []
for cls in range(N_CLS):
    x_distr.append(ProbDistr(n_feats=N_FEATS).create_distr_by_name("uniform", pars={"loc":cls, "scale":STD}))

sampling_model = GenerativeDistribution(n_feats=N_FEATS, probs=[0.25, 0.5, 0.25], x_distr=x_distr)
y, X, sampling_model = generate_sample(n_samples=500, gen_distr=sampling_model)
# my_plots.plot_data(y, X)

sampling_prob_distr = sampling_model.x_distr
for cls, cls_distr in enumerate(sampling_prob_distr.distr):
    print("Probability distribution for class {}".format(cls))
    for j, pd in enumerate(cls_distr.pd_list):
        print("N.prob {}: {}, parameters: {}".format(j, pd.pd.name, pd.pars))


fitting_model = copy.deepcopy(sampling_model)

for SAMPLE_SIZE in range(25, 200, 50):
    f1, f1_true = [], []
    print("Sample size {}.".format(SAMPLE_SIZE))
    for nD in range(ND_SAMPLES):
        y, X, sampling_model = generate_sample(n_samples=int(SAMPLE_SIZE * (1 + TRAIN_TEST_RATIO)), gen_distr=sampling_model)
        X_test, X_train = X[:SAMPLE_SIZE, :], X[SAMPLE_SIZE:]
        y_test, y_train = y[:SAMPLE_SIZE], y[SAMPLE_SIZE:]

        y_predicted = sampling_model.make_y_labels(X_test)
        f1_true.append(f1_score(y_true=y_test, y_pred=y_predicted, average="macro"))

        fitting_model.fit_distr_pars(y=y_train, X=X_train)
        y_predicted = fitting_model.make_y_labels(X_test)
        f1.append(f1_score(y_true=y_test, y_pred=y_predicted, average="macro"))

    print("Optimal {}, extimated {}".format(np.mean(f1_true), np.mean(f1)))




# N_SPLITS = 5
# skf = StratifiedKFold(n_splits=N_SPLITS)
# f1, f1_true = [], []
# for train_index, test_index in skf.split(X, y):
#     X_test, X_train = X[test_index], X[train_index]
#     y_test, y_train = y[test_index], y[train_index]
#
#     y_predicted = sampling_model.make_y_labels(X_test)
#     f1_true.append(f1_score(y_true=y_test, y_pred=y_predicted, average="macro"))
#
#     fitting_model.fit_distr_pars(y=y_train, X=X_train)
#     y_predicted = fitting_model.make_y_labels(X_test)
#     f1.append(f1_score(y_true=y_test, y_pred=y_predicted, average="macro"))




print("That's all")