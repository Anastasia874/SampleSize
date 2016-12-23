"""    
    Based on Byron Curious Snake (base_learner.py) by C Wallace
    --
    This module contains the BaseLearner class, which you can subclass  to implement your own 
    (pool-based) active learning strategy. BaseLearner itself can actually be used directly; it implements
    the 'random' strategy, i.e., it picks examples for the expert to label at random. 
"""
import copy
import random
import dataset 
import numpy as np


class BaseLearner(object):
    """
    Base learner class. Sub-class this object to implement your own learning strategy.
    """ 

    def __init__(self, dataset_=None, model=None, rebuild_model_at_each_iter=True, name="Base"):
        """

        :param dataset_: structure with input data
        :type dataset_: Dataset
        :param model: classification model
        :type model: model object
        :param rebuild_model_at_each_iter: defines if the model will be refitted at each call to active_learn
        :type rebuild_model_at_each_iter: bool

        """
        if not isinstance(dataset_, dataset.Dataset):
            raise TypeError('Input data should be of type dataset.Dataset, got {}'.format(type(dataset_)))
            
        self.dataset = copy.deepcopy(dataset_)
        self.model = copy.deepcopy(model)
        self.name = name
        self.description = ""
        self.instances = []
        self.labeled_instances = []
        self.labels = []
        self.rebuild_model_at_each_iter = rebuild_model_at_each_iter

        if model is None:
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression()
        if hasattr(self.model, "problem_type"):
            self.model.problem_type = self.dataset.type
        if hasattr(self.model, "classes"):
            self.model.classes = self.dataset.classes

    def active_learn(self, num_examples_to_label, batch_size=5):
        """
        Core active learning routine. Here the learner uses its query function to select a number of examples
        to label at each step, until the total number of examples requested has been labeled. The model will be updated
        at each iteration.

        :param num_examples_to_label: number of examples to be labeled
        :type num_examples_to_label: int
        :param batch_size: max number of examples that can be labeled at one step
        :type batch_size: int
        :return: None
        """
        labeled_so_far = 0
        while labeled_so_far < num_examples_to_label:
            example_ids_to_label = self.query_function(batch_size)
            # now remove the selected examples from the unlabeled sets and put them in the labeled sets.
            # if not ids are returned -- ie., if a void query_function is used --
            # it is assumed the query function took care of labeling the examples selected. 
            if example_ids_to_label is not None:
                self.label_instances(example_ids_to_label)

            if self.rebuild_model_at_each_iter:
                self.rebuild_model()   

            labeled_so_far += batch_size

        self.rebuild_model()

    def predict(self, X):
        """ 
        This defines how we will predict labels for new examples. We use a simple ensemble voting
        strategy if there are multiple feature spaces. If there is just one feature space, this just
        uses the 'predict' function of the model.
        """
        return self.model.predict(X)
                
    def query_function(self, k):
        """ Overwrite this method with query function of choice (e.g., SIMPLE) """
        raise Exception("no query function provided!")

    def label_all_data(self):
        """
        Labels all the examples in the training set
        """
        inst_ids = self.dataset.unlabeled_idx
        self.label_instances(inst_ids)

    def label_instances(self, ids_to_label, error_rate=0.0):
        """
        Imitates labeling process

        :param ids_to_label: indices to label
        :type ids_to_label: list
        :param error_rate: probability of mislabeling a sample
        :type error_rate: float
        :return: None
        """

        new_labels, true_labels = self.dataset.label_selected_instances(ids_to_label, error_rate)
        self.labeled_instances = self.dataset.labeled_data
        self.labels = np.hstack((self.labels, new_labels))
        return new_labels, true_labels

    def pick_balanced_initial_training_set(self, k, indices=True):
        """
        Picks k examples of each class.
        """
        if self.dataset.type == "regression":
            return self.dataset.pick_random_instances(k, indices)

        cls_x_to_label = []
        for cls in self.dataset.classes:
            cls_x_to_label.append(self.dataset.pick_random_instances_from_cls(cls, k, indices))

        if indices:
            all_x_to_label = np.hstack(cls_x_to_label)
        else:
            all_x_to_label = np.vstack(cls_x_to_label)

        np.random.shuffle(all_x_to_label)
        return all_x_to_label

    def get_labeled_instance_ids(self):
        return self.dataset.labeled_instances()

    def get_random_unlabeled_ids(self, n_ids):
        return random.sample(self.dataset.unlabeled_idx, n_ids)

    def model_confidence(self, inputs):
        """
        If the model provides some confidence score, this returns prediction confidence for the inputs

        :param inputs: data matrix
        :type inputs: numpy.ndarray
        :return: confidence scores
        :rtype: numpy.ndarray
        """
        if hasattr(self.model, 'model'):
            return self.model.predict_proba(inputs)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(inputs)
        if hasattr(self.model, 'decision_function'):
            return self.model.decision_function(inputs) # FIXIT
        raise AttributeError('Model should provide confidence score, {} does not'.format(self.model.__name__))

    def rebuild_model(self):
        self.model.fit(self.labeled_instances, self.labels)
        # raise Exception, "No model provided! (BaseLearner)"

    # def unlabel_instances(self, inst_ids):
    #     for inst_index in range(len(self.labeled_datasets[0].instances)):
    #         if self.labeled_datasets[0].instances[inst_index].id in inst_ids:
    #             for unlabeled_dataset, labeled_dataset in zip(self.datasets, self.labeled_datasets):
    #                 labeled_dataset.instances[inst_index].lbl = labeled_dataset.instances[inst_index].label
    #                 labeled_dataset.instances[inst_index].has_synthetic_label = False
    #
    #     # now remove the instances and place them into the unlabeled set
    #     for unlabeled_dataset, labeled_dataset in zip(self.datasets, self.labeled_datasets):
    #         unlabeled_dataset.add_instances(labeled_dataset.remove_instances(inst_ids))


def _arg_max(ls, f):
    """ Returns the index for x in ls for which f(x) is maximal w.r.t. the rest of the list """
    return_index = 0
    max_val = f(ls[0])
    for i in range(len(ls)-1):
        if f(ls[i+1]) > max_val:
            return_index = i
            max_val = f(ls[i+1])
    return return_index
