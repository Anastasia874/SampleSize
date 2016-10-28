from __future__ import print_function

import numpy as np
import my_plots

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import det_generate_test_data
from generate_test_data import GenerativeDistribution, ProbDistr, generate_sample


N_CLS = 3
N_FEATS = 2
STD = 0.5#np.linspace(0.5, 1.5, N_CLS)

x_distr = []
for cls in range(N_CLS):
    x_distr.append(ProbDistr(n_feats=N_FEATS).create_distr_by_name("normal", pars={"loc":cls, "scale":STD}))

sampling_model = GenerativeDistribution(n_feats=N_FEATS, probs=[0.25, 0.5, 0.25], x_distr=x_distr)
y, X, sampling_model = generate_sample(n_samples=500, gen_distr=sampling_model)
my_plots.plot_data(y, X)

sampling_prob_distr = sampling_model.x_distr
for cls, cls_distr in enumerate(sampling_prob_distr.distr):
    print("Probability distribution for class {}".format(cls))
    for j, pd in enumerate(cls_distr.pd_list):
        print("N.prob {}: {}, parameters: {}".format(j, pd.pd.name, pd.pars))

####################################################################################################
sampling_model = det_generate_test_data.GenerativeDistribution(n_feats=1, probs=[0.25, 0.5, 0.25])
y, X, sampling_model = det_generate_test_data.generate_sample(n_samples=500, gen_distr=sampling_model)

#my_plots.plot_data(y, X)
my_plots.plot_data(y, sampling_model.lh_func.transformedX)


N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS)
f1 = []
for train_index, test_index in skf.split(X, y):
    X_test = X[test_index]
    y_test = y[test_index]

    y_predicted = sampling_model.make_y_labels(X_test)
    f1.append(f1_score(y_true=y_test, y_pred=y_predicted, average="macro"))




print("That's all")