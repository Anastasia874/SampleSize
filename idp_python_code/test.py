from __future__ import print_function

import my_plots

from sklearn.linear_model import LogisticRegression
from generate_test_data import GenerativeDistribution, LikelihoodFunc, generate_sample


sampling_model = GenerativeDistribution(n_feats=1, probs=[0.25, 0.5, 0.25])
y, X, sampling_model = generate_sample(n_samples=500, gen_distr=sampling_model)
my_plots.plot_data(y, X)
my_plots.plot_data(y, sampling_model.lh_func.transformedX)



print("That's all")