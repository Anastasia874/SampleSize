"""
    Based on Curious Snake: Active Learning in Python by Byron C Wallace
    ----
    CuriousSnake is distributed under the modified BSD licence
    Copyright (c)  2009,  byron c wallace
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in the
          documentation and/or other materials provided with the distribution.
        * Neither the name of Tufts Medical Center nor the
          names of its contributors may be used to endorse or promote products
          derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY byron c wallace 'AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL byron wallace BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from __future__ import print_function
import os
import copy
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from collections import defaultdict

import dataset
import my_plots
from active_learning import query_strategies
from active_learning import bayesian_models


def run_experiments_hold_out(n_samples=1000, n_test_samples=500, out_path="tmp", upto=None, step_size=25,
                             initial_size=4, batch_size=5, pick_balanced_initial_set=True,
                             num_runs=10):
    """
    This method demonstrates how to use the active learning framework, and is also a functional routine for comparing 
    learners. Basically,
    a number of runs will be performed, the active learning methods will be evaluated at each step, and results will be 
    reported. The results
    for each run will be dumped to a text files, which then can be combined (e.g., averaged), elsewhere, or you can use 
    the results_reporter
    module to aggregate and plot the output.

    @parameters
    --
    out_path -- this is a directory under which all of the results will be dumped.
    upto -- active learning will stop when upto examples have been labeled. if this is None, upto will default to the 
    total unlabeled pool available
    initial_size -- the size of 'bootstrap' set to use prior to starting active learning (for the initial models)
    batch_size -- the number of examples to be labeled at each iteration in active learning -- optimally, 1
    step_size -- results will be reported every time another step_size examples have been labeled
    pick_balanced_initial_set -- if True, the initial train dataset will be built over an equal number (initial_size/2) 
    of both classes.
    num_runs -- this many runs will be performed
    report_results -- if true, the results_reporter module will be used to generate output.
    """
    data, learners = None, []
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    model = bayesian_models.BinBetaPrior(same_posterior_params=True)
    for run in range(num_runs):
        print("\n********\non run {}".format(run))

        num_labels_so_far = initial_size  # set to initial size for first iteration

        if not os.path.isdir(out_path):
            os.mkdir(out_path)

        data = dataset.build_gaussian_mixture_dataset(n_samples=n_samples)
        #test_data = dataset.build_gaussian_mixture_dataset(n_samples=n_test_samples)
        #data = dataset.build_linear_regression_dataset(n_samples=n_samples, n_features=3, n_informative=1, noise=15)
        #data, test_data = data.train_test_split(n_test_samples=n_test_samples)
        test_data = data
        total_num_examples = len(data.X)

        if upto is None:
            upto = total_num_examples

        hold_out_size = len(test_data.X)

        print("using {} out of {} instances for test set".format(hold_out_size, total_num_examples))

        # Set up the learners, add to list. Here is where you would instantiate new learners.
        learners = [query_strategies.LindleyInformation(dataset_=data, rebuild_model_at_each_iter=False, model=model),
                    # query_strategies.RandomSampling(dataset_=data, rebuild_model_at_each_iter=False, model=None, name="LogR"),
                    query_strategies.RandomSampling(dataset_=data, rebuild_model_at_each_iter=False, model=model),
                    query_strategies.LeastConfidentSampling(dataset_=data, rebuild_model_at_each_iter=False, model=model),
                    query_strategies.MaxEntropySampling(dataset_=data, rebuild_model_at_each_iter=False, model=model)]

        output_files = [open("{}//{}_{}.txt".format(out_path, learner.name, run), 'w') for learner in learners]

        # arbitrarily pick the initial ids from the first learner
        initial_f = learners[0].get_random_unlabeled_ids
        init_size = num_labels_so_far
        if pick_balanced_initial_set:
            initial_f = learners[0].pick_balanced_initial_training_set
            init_size = int(num_labels_so_far / 2.0)  # equal number from both classes

        # Again, you could call *.initial_f on any learner -- it just returns the ids to label initially. these will
        # be the same for all learners.
        init_ids = initial_f(init_size)

        # label instances and build initial models
        for learner in learners:
            learner.label_instances(init_ids)
            learner.rebuild_model()

        # report initial results, to console and file.
        results = report_results(learners, test_data, num_labels_so_far, output_files, results)
        print(results.keys())
        first_iter = True
        while num_labels_so_far <= upto - step_size:
            #
            # the main active learning loop
            #
            cur_step_size = step_size
            cur_batch_size = batch_size
            if first_iter:
                # here we account for the initial labeled dataset size. for example, suppose
                # the step_size is set to 25 (we want to report results every 25 labels),
                # but the initial size was 2; then we want to label 23 on the first iteration
                # so that we report results when 25 total labels have been provided
                cur_step_size = step_size - num_labels_so_far if num_labels_so_far <= step_size \
                    else step_size - (num_labels_so_far - step_size)
                # in general, step_size is assumed to be a multiple of batch_size, for the first iteration,
                # when we're catching up to to the step_size (as outlined above), we set the
                # batch_size to 1 to make sure this condition holds.
                cur_batch_size = 1
                first_iter = False

            for learner in learners:
                learner.active_learn(cur_step_size, batch_size=cur_batch_size)

            num_labels_so_far += cur_step_size
            print("\n*** labeled {} examples out of {} so far ***".format(num_labels_so_far, upto))

            results = report_results(learners, test_data, num_labels_so_far, output_files, results)
        # close files
        [f.close() for f in output_files]

    report_final_results(data, learners, results, out_path)


def report_final_results(data, learners, results, out_path):
    for learner in learners:
        if hasattr(learner.model, "coefs") and hasattr(learner, "ps"):
            my_plots.animate_clf_results(data, learner.model)
    f_selected_data = my_plots.plot_selection_results(data, learners)
    if data.type == "classification":
        f_metrics = my_plots.plot_clf_metrics(results)
        f_metrics.savefig(os.path.join(out_path, "clf_results.png"))
        plt.close(f_metrics)
    if data.type == "regression":
        f_metrics = my_plots.plot_reg_metrics(results)
        f_metrics.savefig(os.path.join(out_path, "reg_results.png"))
        plt.close(f_metrics)

    f_selected_data.savefig(os.path.join(out_path, "selection_results.png"))
    plt.close(f_selected_data)
    # FIXIT add .tex report


def report_results(learners, test_dataset, cur_size, output_files, results):
    """
    Writes results for the learners, as evaluated over the test_dataset, to the console and the parametric
    output files.
    """
    learner_index = 0
    for learner in learners:
        print("\n Results for {} @ {} labeled examples:".format(learner.name, len(learner.dataset.labeled_data)))
        res = evaluate_learner_with_holdout(learner, cur_size, test_dataset, results[learner.name], test_dataset.type)
        results[learner.name] = copy.deepcopy(res)
        # write_out_results(results[learner.name], output_files[learner_index], cur_size) # FIXIT
        learner_index += 1

    return results


def evaluate_learner_with_holdout(learner, num_labels, test_set, results, type):
    """
    If you're not considering a "finite pool" problem, this is the correct way to evaluate the trained classifiers.

    :param learner: the learner to be evaluated
    :type learner:
    :param num_labels: how many labels have been provided to the learner thus far
    :type num_labels:
    :param test_set: the set of examples to be used for evaluation
    :type test_set:
    :return:
    :rtype:
    """
    print("evaluating learner over {} instances.".format(len(learner.dataset.X)))

    X = test_set.X
    true_labels = test_set.y
    predictions = learner.predict(X)
    new_results = _evaluate_predictions(predictions, true_labels, type)
    for k, v in list(new_results.items()):
        results[num_labels][k].append(v)

    return results


def _evaluate_predictions(predictions, true_labels, type):
    if type == "classification":
        return _evaluate_clf_predictions(predictions, true_labels)
    if type == "regression":
        return _evaluate_reg_predictions(predictions, true_labels)


def _evaluate_reg_predictions(predictions, true_labels):
    results = dict()

    results["mae"] = metrics.mean_absolute_error(true_labels, predictions)
    results["mse"] = metrics.mean_squared_error(true_labels, predictions)
    results["r2"] = metrics.r2_score(true_labels, predictions)

    for k in results.keys():
        print("{}: {}".format(k, results[k]))

    return results


def _evaluate_clf_predictions(predictions, true_labels):
    results = dict()
    conf_mat = metrics.confusion_matrix(true_labels, predictions)
    results["confusion_matrix"] = conf_mat
    results["tp"] = conf_mat[0, 0]
    results["tn"] = np.sum(np.diag(conf_mat)) - conf_mat[0, 0]
    results["fn"] = np.sum(conf_mat[0, 1:])
    results["fp"] = np.sum(conf_mat[1:, 0])
    results["accuracy"] = metrics.accuracy_score(true_labels, predictions)

    try:
        results["auc"] = metrics.roc_auc_score(true_labels, predictions)
    except:
        results["auc"] = 0.5
    results["f1"] = metrics.f1_score(true_labels, predictions)

    if float(results["tp"]) == 0:
        results["sensitivity"] = 0
    else:
        results["sensitivity"] = float(results["tp"]) / float(results["tp"] + results["fn"])
    if results["tn"] == 0:
        results["specificity"] = 0
    else:
        results["specificity"] = float(results["tn"]) / float(results["tn"] + results["fp"])

    for k in results.keys():
        if k not in ["confusion_matrix", "tp", "fp", "tn", "fn"]:
            print("{}: {}".format(k, results[k]))

    return results


# def write_out_results(results, outf, size):
#     write_these_out = [results[size][k] for k in ["size", "accuracy", "sensitivity", "specificity"]]
#     outf.write(",".join([str(s) for s in write_these_out]))
#     outf.write("\n")


if __name__ == "__main__":
    run_experiments_hold_out(n_samples=1000, num_runs=20, upto=200, initial_size=20)
