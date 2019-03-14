# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6
"""
__synopsis__    : Produces pdfs over the support set classes for the target set sample.

__description__ : Produces pdfs over the support set classes for the target set sample.
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : "0.1"
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root
                  directory of this source tree.

__classes__     : Attn

__variables__   :

__methods__     :
"""

import torch
import numpy as np
import torch.nn as nn

from logger.logger import logger
from models_orig import PairCosineSim as C


class Attn(nn.Module):
    def __init__(self):
        super(Attn, self).__init__()

    def forward(self, similarities, support_set_y):
        """
        Produces pdfs over the support set classes for the target samples.

        :param similarities: A tensor with cosine similarities of size [sequence_length, batch_size]
        :param support_set_y: A tensor with the one hot vectors of the targets for each support set image
                                                                            [sequence_length,  batch_size, num_classes]
        :return: Softmax pdf
        """
        # logger.debug(("similarities.shape: ",similarities.shape))
        # logger.debug(("support_set_y.shape: ",support_set_y.shape))
        # logger.debug(("support_set_y: ",support_set_y))
        softmax = nn.Softmax()
        softmax_similarities = softmax(similarities)
        # logger.debug(("softmax_similarities.shape: ",softmax_similarities.shape))
        softmax_similarities = softmax_similarities.unsqueeze(1)
        # logger.debug(("softmax_similarities.unsqueeze(1).shape: ",softmax_similarities.shape))
        # preds = softmax_similarities.unsqueeze(1).bmm(support_set_y).squeeze()
        preds = softmax_similarities.bmm(support_set_y)
        # logger.debug(("preds.shape: ",preds.shape))
        preds = preds.squeeze()
        # logger.debug(("preds.squeeze().shape: ",preds.shape))
        # logger.debug(("preds: ",preds))
        return preds


if __name__ == '__main__':

    a = torch.tensor([[[1., 0.4],
                       [1., 1.]],
                      [[1., 0.4],
                       [0., 1.5]],
                      [[1., 0.4],
                       [1., 1.5]]])

    b = torch.tensor([[[1., 0.4],
                       [0., 1.5]],
                      [[1., 0.4],
                       [1., 1.5]]])
    # a = torch.ones(1,4,7)
    # logger.debug(a)
    logger.debug(a.shape)
    # b = torch.ones(1,2,7)
    # logger.debug(b)
    logger.debug(b.shape)
    test_DN = C.PairCosineSim()
    sim = test_DN(a, b)
    logger.debug(sim.shape)
    logger.debug("sim: {}".format(sim))

    # y = torch.tensor([[1., 0.],
    #                   [0., 1.],
    #                   [1., 0.]])
    y = torch.tensor([[[1., 0.],
                       [0., 1.]],
                      [[0., 1.],
                       [1., 0.]]])
    # y = torch.ones(2, 2, 5)
    # y = torch.ones(1, 4, 5)
    logger.debug("y.shape: {}".format(y.shape))
    logger.debug("y: {}".format(y))
    test_attn = Attn()
    result = test_attn(sim, y)
    logger.debug("Attention: {}".format(result))
    logger.debug("Attention.shape: {}".format(result.shape))
    # result = test_attn.forward2(sim, y)
    # logger.debug("Attention: {}".format(result))
    # logger.debug("Attention.shape: {}".format(result.shape))
