# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6
"""
__synopsis__    : Matching Networks for Extreme Classification.
__description__ : Metrics (Precision@k and NDCG@k) for Matching Networks for Extreme Classification.
__project__     : MNXC
__author__      : Vishwak
__version__     : "0.1"
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root directory of this source tree.

__classes__     :

__variables__   :

__methods__     :
"""

import numpy as np
import torch
from logger.logger import logger


def precision_at_k(ground_truth, predictions, k=5, pos_label=1):
    """
    Function to evaluate the precision @ k for a given
    ground truth vector and a list of predictions (between 0 and 1).

    Args:
        ground_truth : np.array consisting of multi-hot encoding of
                       label vector
        predictions : np.array consisting of predictive probabilities
                      for every label.
        k : Value of k. Default: 5
        pos_label : Value to consider as positive. Default: 1

    Returns:
        precision @ k for a given ground truth - prediction pair.
    """
    assert len(ground_truth) == len(predictions), "P@k: Length mismatch: len(ground_truth) [{}] == [{}] len(predictions)"\
        .format(len(ground_truth), len(predictions))

    if torch.is_tensor(ground_truth):
        logger.debug("'ground_truth' is of [{}] type. Converting to Numpy.".format(type(ground_truth)))
        ground_truth = ground_truth.numpy()
        logger.debug(ground_truth)
    if torch.is_tensor(predictions):
        logger.debug("'predictions' is of [{}] type. Converting to Numpy.".format(type(predictions)))
        predictions = predictions.data.numpy()
        logger.debug(predictions)

    # if len(predictions)>len(ground_truth):
    #     logger.debug("Length of 'predictions'[{}] is more than length of 'ground_truth'[{}]. "
    #                  "Reducing the length of 'predictions'.".format(len(predictions),len(ground_truth)))
    #     predictions = predictions[:len(ground_truth)]

    n_pos_vals = (ground_truth == pos_label).sum()
    desc_order = np.argsort(predictions)[::-1]  # [::-1] reverses array
    ground_truth = np.take(ground_truth, desc_order[:k])  # taking the top indices
    relevant_preds = (ground_truth == pos_label).sum()

    return relevant_preds / min(n_pos_vals, k)


def dcg_score_at_k(ground_truth, predictions, k=5, pos_label=1):
    """
    Function to evaluate the Discounted Cumulative Gain @ k for a given
    ground truth vector and a list of predictions (between 0 and 1).

    Args:
        ground_truth : np.array consisting of multi-hot encoding of label
                       vector
        predictions : np.array consisting of predictive probabilities for
                      every label.
        k : Value of k. Default: 5
        pos_label : Value to consider as positive. Default: 1

    Returns:
        DCG @ k for a given ground truth - prediction pair.
    """
    assert len(ground_truth) == len(predictions), "DCG@k: Length mismatch: len(ground_truth) [{}] == [{}] len(predictions)"\
        .format(len(ground_truth), len(predictions))

    desc_order = np.argsort(predictions)[::-1]  # ::-1 reverses array
    ground_truth = np.take(ground_truth, desc_order[:k])  # the top indices
    gains = 2 ** ground_truth - 1

    discounts = np.log2(np.arange(1, len(ground_truth) + 1) + 1)
    return np.sum(gains / discounts)


def ndcg_score_at_k(ground_truth, predictions, k=5, pos_label=1):
    """
    Function to evaluate the Discounted Cumulative Gain @ k for a given
    ground truth vector and a list of predictions (between 0 and 1).

    Args:
        ground_truth : np.array consisting of multi-hot encoding of label
                       vector
        predictions : np.array consisting of predictive probabilities for
                      every label.
        k : Value of k. Default: 5
        pos_label : Value to consider as positive. Default: 1

    Returns:
        NDCG @ k for a given ground truth - prediction pair.
    """
    dcg_at_k = dcg_score_at_k(ground_truth, predictions, k, pos_label)
    best_dcg_at_k = dcg_score_at_k(ground_truth, ground_truth, k, pos_label)
    return dcg_at_k / best_dcg_at_k


if __name__ == '__main__':
    a = np.random.rand(3,2)
    b = np.ones((3,2))
    logger.debug(a)
    logger.debug(a.shape)
    logger.debug(b)
    logger.debug(b.shape)
    np_val = precision_at_k(b,a)
    logger.debug(np_val.shape)
    logger.debug(np_val)
    logger.debug(type(np_val))

    a_t = torch.from_numpy(a)
    b_t = torch.from_numpy(b)
    # logger.debug(a_t)
    logger.debug(a_t.shape)
    # logger.debug(b_t)
    logger.debug(b_t.shape)

    torch_val = precision_at_k(b_t,a_t)
    logger.debug(torch_val.shape)
    logger.debug(torch_val)
    logger.debug(type(torch_val))
