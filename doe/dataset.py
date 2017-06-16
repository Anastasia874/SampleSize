"""
    dataset.py
    Byron C Wallace
    Tufts Medical Center
        
    This module contains methods and classes for parsing
    and manipulating datasets.
"""
import random
import numpy as np
from sklearn import datasets


def build_gaussian_mixture_dataset(n_samples=1000, means=None, covariance=None):
    """
    builds and returns a dataset
    """

    # Generate random sample, two components
    # np.random.seed(0)
    covariance = covariance or np.array([[0., -0.5], [1.7, .4]])
    means = means or [-2, 1]

    X = np.r_[np.dot(np.random.randn(n_samples, 2), covariance),
              .7 * np.random.randn(n_samples, 2) + np.array(means)]
    y = np.hstack((np.zeros(n_samples), np.ones(n_samples)))

    return Dataset(X, y, name="Test data (gaussian mixture)", type="classification")


def build_dataset_from_gaussian_quantiles(mean=None, cov=1.0, n_samples=100, n_features=2, n_classes=2):
    X, y = datasets.make_gaussian_quantiles(n_samples=n_samples, mean=mean, cov=cov, n_features=n_features,
                                            n_classes=n_classes)

    return Dataset(X, y, name="Test data (gaussian quantiles)", type="classification")


def build_clf_dataset(n_samples=100, n_features=5, n_informative=2, n_redundant=2, n_repeated=0, n_classes=2,
                           n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, shift=0.0, scale=1.0):

    X, y = datasets.make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                                        n_redundant=n_redundant, n_repeated=n_repeated, n_classes=n_classes,
                                        n_clusters_per_class=n_clusters_per_class, weights=weights, flip_y=flip_y,
                                        class_sep=class_sep, shift=shift, scale=scale)

    return Dataset(X, y, name="Test data (sklearn classification)", type="classification")


def build_linear_regression_dataset(n_samples=1000, n_features=1, n_informative=1, noise=10):
    X, y, coef = datasets.make_regression(n_samples=n_samples, n_features=n_features,
                                          n_informative=n_informative, noise=noise,
                                          coef=True)

    data = Dataset(X, y, name="Test data (linear regression)", type="regression")
    data.coef = coef
    data.plot_data()
    return data

        
class Dataset:
    """
    This class represents a set of data. It is comprised mainly of a dictionary mapping
    ids to feature vectors, and various operations.
    """
    negative_class = 1

    def __init__(self, X, y, data=None, labeled_num=10, name="", type=None, init_labeled=False):

        self.X = np.array(X)
        self.y = np.array(y)
        self.name = name
        self.type = type

        if isinstance(data, Dataset):
            self.X = data.X
            self.y = data.y
            self.name = data.name

        n_samples = X.shape[0]
        if n_samples != y.shape[0]:
            raise ValueError("Dataset arguments error: X and y should have the same number of instances, got {} and {}".
                             format(X.shape[0], y.shape[0]))
        all_idx = range(n_samples)

        self.X, self.y = self.X[all_idx], self.y[all_idx]
        self.classes = np.unique(self.y)
        self.labeled_idx = np.zeros(0)
        self.unlabeled_idx = np.array(range(len(X)))
        self.labeled_data = np.zeros((0, self.X.shape[1]))
        self.labels = np.zeros(0)

        if init_labeled:
            self.labeled_idx = all_idx[:labeled_num]
            self.unlabeled_idx = all_idx[labeled_num:]
            self.unlabeled_data = self.X[self.unlabeled_idx]
            self.labeled_data = self.X[self.labeled_idx]
            self.labels = self.y[self.labeled_idx]

    def label_selected_instances(self, ids_to_label, error_rate=0.0):
        """ Remove and return the instances with ids in ids_to_remove """
        self.unlabeled_idx = list(self.unlabeled_idx)
        [self.unlabeled_idx.remove(id_) for id_ in ids_to_label]
        self.unlabeled_idx = np.array(self.unlabeled_idx)
        self.labeled_idx = np.hstack((self.labeled_idx, ids_to_label))

        flip_coin = np.random.rand(len(ids_to_label)) < error_rate
        self.unlabeled_data = self.X[self.unlabeled_idx]

        self.labeled_data = np.vstack((self.labeled_data, self.X[ids_to_label]))
        new_labels, noisy_labels = self.y[ids_to_label], self.y[ids_to_label] + flip_coin  # FIXIT
        self.labels = np.hstack((self.labels, noisy_labels))

        return noisy_labels, new_labels

    def pick_random_instances_from_cls(self, cls, k, indices=True):
        cls_ids = self.get_list_of_ids_from_cls(cls)

        if not len(cls_ids) >= k:
            raise Exception("not enough positive examples in dataset!")

        ids = random.sample(cls_ids, k)
        if indices:
            return ids

        return self.X[ids]

    def pick_random_instances(self, k, indices=True):
        cls_ids = self.unlabeled_idx

        if not len(cls_ids) >= k:
            raise Exception("not enough examples in dataset!")

        ids = random.sample(cls_ids, k)
        if indices:
            return ids

        return self.X[ids]

    def labeled_instances(self):
        return self.labeled_data, self.labels

    def unlabeled_instances(self):
        return self.unlabeled_data

    def get_list_of_ids_from_cls(self, cls):
        ids = np.nonzero(self.y == cls)[0]

        return ids

    def get_examples_from_cls(self, cls):
        return self.get_list_of_ids_from_cls(cls)

    def number_of_cls_examples(self, cls):
        """
        Counts and returns the number of examples from cls in this dataset.
        """
        return len(self.get_examples_from_cls(cls=cls))

    def train_test_split(self, test_ratio=0.25, n_test_samples=None):
        if n_test_samples is not None:
            n_test_samples = np.round(len(self.y) * test_ratio)

        train_data = Dataset(self.X[:-n_test_samples], self.y[:-n_test_samples], name=self.name, type=self.type)
        test_data = Dataset(self.X[-n_test_samples:], self.y[-n_test_samples:], name=self.name, type=self.type)

        return train_data, test_data

    def plot_data(self):
        import my_plots
        if self.type == "regression":
            my_plots.plot_1d_data_reg(self)
        if self.type == "classification":
            my_plots.plot_2dim_data_clf(self)