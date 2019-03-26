# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6
"""
__synopsis__    : Matching Networks for Extreme Classification.

__description__ : Metrics (Precision@k and NDCG@k) for Matching Networks for Extreme Classification.
__project__     : MNXC
__author__      : Vishwak, Samujjwal Ghosh
__version__     : "0.1"
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root
                  directory of this source tree.

__classes__     : Metrics

__variables__   :

__methods__     : precision_at_k, dcg_score_at_k, ndcg_score_at_k
"""

import numpy as np
import torch
from logger.logger import logger

seed_val = 0
# random.seed(seed_val)
# np.random.seed(seed_val)
# torch.manual_seed(seed_val)
# torch.cuda.manual_seed_all(seed=seed_val)


def precision_at_k(actuals, predictions, k=5, pos_label=1):
    """
    Function to evaluate the precision @ k for a given
    ground truth vector and a list of predictions (between 0 and 1).

    Args:
        actuals : np.array consisting of multi-hot encoding of label vector
        predictions : np.array consisting of predictive probabilities for every label.
        k : Value of k. Default: 5
        pos_label : Value to consider as positive. Default: 1

    Returns:
        precision @ k for a given ground truth - prediction pair.
    """
    assert len(actuals) == len(predictions), "P@k: Length mismatch: len(actuals) [{}] == [{}] len(predictions)" \
        .format(len(actuals), len(predictions))

    ## Converting to Numpy as it has supported funcions.
    if torch.is_tensor(actuals):
        logger.debug("'actuals' is of [{}] type. Converting to Numpy.".format(type(actuals)))
        actuals = actuals.numpy()
        logger.debug(actuals)
    if torch.is_tensor(predictions):
        logger.debug("'predictions' is of [{}] type. Converting to Numpy.".format(type(predictions)))
        predictions = predictions.data.numpy()
        logger.debug(predictions)

    n_pos_vals = (actuals == pos_label).sum()
    desc_order = np.argsort(predictions, -k)  # [::-1] reverses array
    matches = np.take(actuals, desc_order[:, :k])  # taking the top indices
    relevant_preds = (matches == pos_label).sum()

    return relevant_preds / min(n_pos_vals, k)


def dcg_score_at_k(actuals, predictions, k=5, pos_label=1):
    """
    Function to evaluate the Discounted Cumulative Gain @ k for a given
    ground truth vector and a list of predictions (between 0 and 1).

    Args:
        actuals : np.array consisting of multi-hot encoding of label
                       vector
        predictions : np.array consisting of predictive probabilities for
                      every label.
        k : Value of k. Default: 5
        pos_label : Value to consider as positive. Default: 1

    Returns:
        DCG @ k for a given ground truth - prediction pair.
    """
    assert len(actuals) == len(predictions), "DCG@k: Length mismatch: len(actuals) [{}] == [{}] len(predictions)" \
        .format(len(actuals), len(predictions))

    desc_order = np.argsort(predictions)[::-1]  # ::-1 reverses array
    actuals = np.take(actuals, desc_order[:k])  # the top indices
    gains = 2 ** actuals - 1

    discounts = np.log2(np.arange(1, len(actuals) + 1) + 1)
    return np.sum(gains / discounts)


def ndcg_score_at_k(actuals, predictions, k=5, pos_label=1):
    """
    Function to evaluate the Discounted Cumulative Gain @ k for a given
    ground truth vector and a list of predictions (between 0 and 1).

    Args:
        actuals : np.array consisting of multi-hot encoding of label
                       vector
        predictions : np.array consisting of predictive probabilities for
                      every label.
        k : Value of k. Default: 5
        pos_label : Value to consider as positive. Default: 1

    Returns:
        NDCG @ k for a given ground truth - prediction pair.
    """
    dcg_at_k = dcg_score_at_k(actuals, predictions, k, pos_label)
    best_dcg_at_k = dcg_score_at_k(actuals, actuals, k, pos_label)
    return dcg_at_k / best_dcg_at_k


class Metrics:
    """ Initializes an Metrics object. """
    def __init__(self, cuda_available=None, use_cuda=False):
        if cuda_available is None:
            self.cuda_available = torch.cuda.is_available()
        else:
            self.cuda_available = cuda_available

        self.use_cuda = use_cuda

    def precision_k_hot(self, actuals, predictions, k=1, pos_label=1):
        """
        Calculates precision of actuals multi-hot vectors and predictions probabilities of shape: (batch_size, Number of samples, Number of categories).

        :param actuals: 3D torch.tensor consisting of multi-hot encoding of label vector of shape: (batch_size, Number of samples, Number of categories)
        :param predictions: torch.tensor consisting of predictive probabilities for every label: (batch_size, Number of samples, Number of categories)
        :param k: Value of k. Default: 1
        :param pos_label: Value to consider as positive in Multi-hot vector. Default: 1

        :return: Precision @ k for a given ground truth - prediction pair.for a batch of samples.
        """
        ## Top k probabilities
        preds_indices = torch.argsort(predictions,dim=2, descending=True)
        preds_desc = preds_indices[:,:,:k]

        # com_labels = []  # batch_size, Number of samples
        precision_batch = 0
        for i in np.arange(predictions.shape[0]):  # (batch_size, Number of samples, Number of categories)
            precision_samples = 0
            for j in np.arange(predictions.shape[1]):
                precision_elm = 0
                for l in np.arange(preds_desc.shape[2]):
                    if actuals[i,j,preds_desc[i,j,l].item()] == pos_label:  # Checking if top index positions are 1.
                        precision_elm += 1
                precision_samples += precision_elm / preds_desc.shape[2]
            precision_batch += precision_samples / predictions.shape[1]
        precision = precision_batch / predictions.shape[0]
        return precision#, com_labels


if __name__ == '__main__':
    cls_count = 3
    multi_hot = np.random.randint(2, size=(2, 2, cls_count))  # Generates integers till 2-1, i.e. [0 or 1]
    logger.debug(multi_hot)
    indices = [[[0], [0, 2]], [[1], [0, 2]]]  # Generates integers till [cls_count]
    logger.debug(indices)
    # indices = np.random.randint(cls_count, size=(1, 2, cls_count))  # Generates integers till [cls_count]
    proba = np.random.rand(2, 2, cls_count)
    logger.debug(proba)

    test_metrics = Metrics()
    proba_t = torch.from_numpy(proba)
    multi_hot_t = torch.from_numpy(multi_hot)
    precision, com_labels = test_metrics.precision_k_hot(multi_hot_t, proba_t, k=2)
    logger.debug(precision)
    logger.debug(com_labels)

    # logger.debug(proba)
    # logger.debug(proba.shape)
    # logger.debug(multi_hot)
    # logger.debug(multi_hot.shape)
    # np_val = precision_at_k(multi_hot, proba, k=1)
    # logger.debug(np_val.shape)
    # logger.debug(np_val)
    # logger.debug(type(np_val))
    #
    # logger.debug(a_t)
    # logger.debug(a_t.shape)
    # logger.debug(b_t)
    # logger.debug(b_t.shape)
    #
    # torch_val = precision_at_k(b_t, a_t)
    # logger.debug(torch_val.shape)
    # logger.debug(torch_val)
    # logger.debug(type(torch_val))
