from __future__ import print_function

import my_plots
from generate_test_data import GenerativeDistribution, LikelihoodFunc, generate_sample


sampling_model = GenerativeDistribution(n_cls=3, n_feats=1)
y, X, sampling_model = generate_sample(n_samples=500, gen_distr=sampling_model)
my_plots.plot_data(y, X)



print("That's all")